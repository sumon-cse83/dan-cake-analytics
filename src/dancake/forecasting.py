"""Layer 6a: predictive analytics - monthly demand forecasting (regression).

Problem: forecast next month's units for each of the 52 products.
Protocol:
  * train on 2020-2024, test on Jan 2025 - Sep 2026 (21 unseen months);
  * hyper-parameters chosen with expanding-window validation (validate on 2022, 2023, 2024);
  * two test modes - one-step-ahead (lags use actual history) and recursive (the model
    must feed on its own predictions for 21 months, as it would in real planning);
  * models compared against naive and seasonal-naive baselines (a model that cannot beat
    "same as last year" has no business value).
"""
from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from . import config as C
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features, monthly_tables
from .utils import plt, save_json, savefig, to_latex_table

warnings.filterwarnings("ignore")
COUNT_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "roll3_mean", "roll6_mean", "roll12_mean", "roll3_std"]
OTHER_NUM = [c for c in NUMERIC_FEATURES if c not in COUNT_FEATURES]


def preprocessor(scale: bool) -> ColumnTransformer:
    """log1p for count-like features (multiplicative demand), median imputation, optional scaling."""
    count = [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("imp", SimpleImputer(strategy="median"))]
    other = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        count.append(("sc", StandardScaler())); other.append(("sc", StandardScaler()))
    return ColumnTransformer([("cnt", Pipeline(count), COUNT_FEATURES), ("num", Pipeline(other), OTHER_NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
                             verbose_feature_names_out=False)


def model_zoo():
    """name -> (estimator, scaled inputs?, search space). Spaces are deliberately modest."""
    return {
        "Linear Regression": (LinearRegression(), True, {}),
        "Decision Tree": (DecisionTreeRegressor(random_state=C.SEED), False,
                          {"m__max_depth": [4, 6, 8, 10, 14], "m__min_samples_leaf": [5, 10, 20, 40]}),
        "Random Forest": (RandomForestRegressor(random_state=C.SEED, n_jobs=-1), False,
                          {"m__n_estimators": [300, 500], "m__max_depth": [8, 12, None],
                           "m__min_samples_leaf": [2, 5, 10], "m__max_features": [0.33, 0.5, 0.8]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=C.SEED), False,
                              {"m__n_estimators": [200, 400, 700], "m__learning_rate": [0.02, 0.05, 0.1],
                               "m__max_depth": [2, 3, 4], "m__subsample": [0.7, 1.0]}),
        "XGBoost": (XGBRegressor(random_state=C.SEED, n_jobs=4, tree_method="hist", verbosity=0), False,
                    {"m__n_estimators": [300, 600, 900], "m__learning_rate": [0.02, 0.05, 0.1],
                     "m__max_depth": [2, 3, 4, 6], "m__subsample": [0.7, 0.9], "m__colsample_bytree": [0.6, 0.8, 1.0],
                     "m__min_child_weight": [1, 5, 10], "m__reg_lambda": [1, 5]}),
        "K-Nearest Neighbors": (KNeighborsRegressor(), True,
                                {"m__n_neighbors": [3, 5, 10, 20, 40], "m__weights": ["uniform", "distance"]}),
        "Support Vector Regression": (SVR(), True,
                                      {"m__C": [0.3, 1, 3, 10], "m__epsilon": [0.01, 0.05, 0.1], "m__gamma": ["scale", 0.01, 0.05]}),
    }


def year_folds(df: pd.DataFrame, years=(2022, 2023, 2024)):
    """Expanding-window folds: train on all months before year y, validate on year y."""
    yr = df.month.dt.year.values
    return [(np.where(yr < y)[0], np.where(yr == y)[0]) for y in years]


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    mse = mean_squared_error(y, p)
    return {"MAE": mean_absolute_error(y, p), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y, p),
            "MAPE": float(np.mean(np.abs(y - p) / np.maximum(y, 1)) * 100),
            "WAPE": float(np.abs(y - p).sum() / y.sum() * 100)}


def recursive_forecast(model, units_hist: pd.DataFrame, price: pd.DataFrame, meta: pd.DataFrame, months) -> pd.DataFrame:
    """Forecast several months ahead, feeding each month's predictions back in as history."""
    U = units_hist.copy()
    preds = {}
    for m in months:
        U.loc[m] = np.nan
        f = build_features(U, price, meta)
        rows = f[f.month == m]
        p = np.clip(np.expm1(model.predict(rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES])), 0, None)
        U.loc[m, rows.Product_ID.values] = p
        preds[m] = pd.Series(p, index=rows.Product_ID.values)
    return pd.DataFrame(preds).T


def run(d: pd.DataFrame) -> dict:
    out: dict = {}
    units, price, meta = monthly_tables(d)
    F = build_features(units, price, meta)
    F = F[F.lag_1.notna()]                                     # need at least one month of history
    X_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train = F[F.month <= C.TRAIN_END].reset_index(drop=True)
    test = F[F.month >= C.TEST_START].reset_index(drop=True)
    ytr = np.log1p(train.units)
    out["panel"] = {"series": int(F.Product_ID.nunique()), "train_rows": len(train), "test_rows": len(test),
                    "train_months": int(train.month.nunique()), "test_months": int(test.month.nunique()),
                    "n_features": len(X_cols)}
    folds = year_folds(train)

    # ---------------------------------------------------------------- baselines
    rows, fitted, cv_scores = [], {}, {}
    for name, col in [("Naive (last month)", "lag_1"), ("Seasonal naive (same month last year)", "lag_12")]:
        p = test[col].fillna(test.lag_1)
        rows.append({"Model": name, **metrics(test.units, p), "fit_s": 0.0, "kind": "baseline"})

    # ---------------------------------------------------------------- ML models with tuning
    tuned_params = {}
    for name, (est, scale, space) in model_zoo().items():
        pipe = Pipeline([("prep", preprocessor(scale)), ("m", est)])
        t0 = time.time()
        if space:
            n_iter = min(25, int(np.prod([len(v) for v in space.values()])))
            search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=folds, scoring="neg_root_mean_squared_error",
                                        random_state=C.SEED, n_jobs=-1 if name not in ("Random Forest", "XGBoost") else 2)
            search.fit(train[X_cols], ytr)
            pipe = search.best_estimator_
            cv_scores[name] = float(-search.best_score_)
            tuned_params[name] = {k.replace("m__", ""): (v if not isinstance(v, np.generic) else v.item())
                                  for k, v in search.best_params_.items()}
        else:
            sc = []
            for tr, va in folds:
                pipe.fit(train.loc[tr, X_cols], ytr[tr])
                sc.append(np.sqrt(mean_squared_error(ytr[va], pipe.predict(train.loc[va, X_cols]))))
            cv_scores[name] = float(np.mean(sc))
            pipe.fit(train[X_cols], ytr)
            tuned_params[name] = {}
        fit_s = time.time() - t0
        p = np.clip(np.expm1(pipe.predict(test[X_cols])), 0, None)
        rows.append({"Model": name, **metrics(test.units, p), "fit_s": fit_s, "kind": "ml"})
        fitted[name] = pipe
        test[f"pred_{name}"] = p
        print(f"   [forecast] {name:28s} test RMSE {rows[-1]['RMSE']:7.1f}  WAPE {rows[-1]['WAPE']:5.2f}%  ({fit_s:.0f}s)")

    res = pd.DataFrame(rows)
    res["CV RMSE (log)"] = res.Model.map(cv_scores)
    ml = res[res.kind == "ml"].sort_values("RMSE")
    best = ml.Model.iloc[0]
    out["best_model"] = best
    out["one_step"] = res.drop(columns="kind").to_dict("records")
    out["tuned_params"] = tuned_params
    base_rmse = res.loc[res.Model.str.startswith("Seasonal"), "RMSE"].iloc[0]
    naive_rmse = res.loc[res.Model.str.startswith("Naive"), "RMSE"].iloc[0]
    out["best_vs_seasonal_naive_pct"] = float((1 - ml.RMSE.iloc[0] / base_rmse) * 100)
    out["best_vs_naive_pct"] = float((1 - ml.RMSE.iloc[0] / naive_rmse) * 100)
    t = res.sort_values(["kind", "RMSE"], ascending=[True, True]).drop(columns=["kind"])
    t = t[["Model", "MAE", "MSE", "RMSE", "R2", "MAPE", "WAPE", "CV RMSE (log)", "fit_s"]]
    t.columns = ["Model", "MAE", "MSE", "RMSE", "R$^2$", "MAPE %", "WAPE %", "CV RMSE", "Fit (s)"]
    to_latex_table(t, "forecast_models", "Monthly product-demand forecasting: test-set performance "
                   "(one-step-ahead, Jan 2025--Sep 2026)", "tab:forecast",
                   fmt={"MAE": "{:,.1f}", "MSE": "{:,.0f}", "RMSE": "{:,.1f}", "R$^2$": "{:.3f}", "MAPE %": "{:.2f}",
                        "WAPE %": "{:.2f}", "CV RMSE": "{:.4f}", "Fit (s)": "{:.1f}"}, font=r"\footnotesize",
                   note="Errors in units per product-month. CV RMSE is the mean validation RMSE on the log scale over the "
                        "expanding-window folds 2022, 2023 and 2024 used for tuning. Baselines need no fitting.")

    # ---------------------------------------------------------------- recursive multi-step test (honest planning scenario)
    test_months = sorted(test.month.unique())
    hist = units[units.index <= C.TRAIN_END]
    rec_rows = []
    rec_preds = {}
    for name in ml.Model.head(4):
        R = recursive_forecast(fitted[name], hist, price, meta, test_months)
        rp = R.stack().rename("pred").reset_index().rename(columns={"level_0": "month", "level_1": "Product_ID"})
        mm = test[["month", "Product_ID", "units"]].merge(rp, on=["month", "Product_ID"])
        rec_rows.append({"Model": name, **metrics(mm.units, mm.pred)})
        rec_preds[name] = mm
    sn = test.lag_12.fillna(test.lag_1)        # seasonal naive is naturally multi-step up to 12 months
    # for months 13-21 ahead the seasonal naive must reuse 2024 values again (24-month lag)
    u24 = units.shift(24).stack(future_stack=True).rename("lag24").reset_index().rename(columns={"level_0": "month", "level_1": "Product_ID"})
    u24.columns = ["month", "Product_ID", "lag24"]
    tt = test.merge(u24, on=["month", "Product_ID"], how="left")
    sn_multi = np.where(tt.month <= pd.Timestamp("2025-12-01"), tt.lag_12, tt.lag24)
    rec_rows.append({"Model": "Seasonal naive (2024 pattern repeated)", **metrics(tt.units, pd.Series(sn_multi).fillna(tt.lag_1))})
    rec = pd.DataFrame(rec_rows).sort_values("RMSE")
    out["recursive"] = rec.to_dict("records")
    best_rec = rec.Model.iloc[0] if not rec.Model.iloc[0].startswith("Seasonal") else rec.Model.iloc[1]
    out["best_recursive_model"] = best_rec
    rt = rec[["Model", "MAE", "RMSE", "R2", "MAPE", "WAPE"]].copy()
    rt.columns = ["Model", "MAE", "RMSE", "R$^2$", "MAPE %", "WAPE %"]
    to_latex_table(rt, "forecast_recursive", "Recursive 21-month forecast from December 2024 (no actual data used after "
                   "training)", "tab:recursive", fmt={"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "R$^2$": "{:.3f}",
                                                       "MAPE %": "{:.2f}", "WAPE %": "{:.2f}"})

    # ---------------------------------------------------------------- company-level: statistical time-series models
    total_units = units.sum(axis=1)
    total_sales = d.groupby(d.Date.dt.to_period("M").dt.to_timestamp()).Net_Sales_BDT.sum()
    tr_u, te_u = total_units[:C.TRAIN_END], total_units[C.TEST_START:]
    hw = ExponentialSmoothing(tr_u, trend="add", seasonal="mul", seasonal_periods=12).fit(optimized=True)
    hw_p = hw.forecast(len(te_u))
    sar = SARIMAX(np.log(tr_u), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    sar_p = np.exp(sar.forecast(len(te_u)))
    ml_total = rec_preds[best_rec].groupby("month").pred.sum()
    ml_onestep_total = test.groupby("month")[f"pred_{best}"].sum()
    comp = []
    for name, p in [("Holt-Winters (additive trend, multiplicative season)", hw_p.values),
                    ("SARIMA(1,1,1)(0,1,1)$_{12}$ on log units", sar_p.values),
                    (f"{best_rec} recursive, summed over products", ml_total.values),
                    (f"{best} one-step, summed over products", ml_onestep_total.values),
                    ("Seasonal naive (2024 repeated)", np.r_[tr_u[-12:].values, tr_u[-12:].values][:len(te_u)])]:
        mt = metrics(te_u.values, p)
        comp.append({"Model": name, "MAE": mt["MAE"], "RMSE": mt["RMSE"], "MAPE": mt["MAPE"]})
    comp = pd.DataFrame(comp).sort_values("MAPE")
    out["company_level"] = comp.to_dict("records")
    ct = comp.copy(); ct.columns = ["Model (company-total monthly units)", "MAE", "RMSE", "MAPE %"]
    to_latex_table(ct, "forecast_company", "Company-level monthly units forecast, test period Jan 2025--Sep 2026",
                   "tab:company", fmt={"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE %": "{:.2f}"}, col_align="lrrr")

    # ---------------------------------------------------------------- figures: comparison & actual vs predicted
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.35, 1]})
    r = res.sort_values("RMSE")
    ax[0].barh(r.Model, r.RMSE, color=[C.ACCENT if m == best else (C.PALETTE[9] if k == "baseline" else C.PRIMARY)
                                        for m, k in zip(r.Model, r.kind)])
    ax[0].invert_yaxis(); ax[0].set_title("Test RMSE by model"); ax[0].set_xlabel("RMSE (lower is better)")
    ax[1].scatter(test.units, test[f"pred_{best}"], s=8, alpha=.5, color=C.PRIMARY)
    lim = [min(test.units.min(), test[f"pred_{best}"].min()) * 0.9, test.units.max() * 1.1]
    ax[1].plot(lim, lim, "k--", lw=.8, label="perfect forecast")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlim(lim); ax[1].set_ylim(lim); ax[1].legend()
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter   # plain, non-overlapping log ticks
    for axis in (ax[1].xaxis, ax[1].yaxis):
        axis.set_major_locator(FixedLocator([t for t in (100, 200, 500, 1000, 2000, 5000) if lim[0] <= t <= lim[1]]))
        axis.set_major_formatter(ScalarFormatter()); axis.set_minor_locator(NullLocator())
    ax[1].set_title("Best model: actual vs predicted"); ax[1].set_xlabel("actual units"); ax[1].set_ylabel("predicted units")
    savefig(fig, "ml_forecast_comparison")

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(total_units.index, total_units / 1e3, color="black", lw=1.4, label="actual")
    ax.plot(te_u.index, hw_p / 1e3, color=C.PALETTE[2], lw=1.4, ls="--", label="Holt-Winters")
    ax.plot(te_u.index, sar_p / 1e3, color=C.PALETTE[3], lw=1.2, ls=":", label="SARIMA")
    ax.plot(ml_total.index, ml_total / 1e3, color=C.ACCENT, lw=1.6, label=f"{best_rec} (recursive)")
    ax.axvline(pd.Timestamp(C.TEST_START), color="grey", lw=.8); ax.text(pd.Timestamp(C.TEST_START), ax.get_ylim()[1] * .98, "  test period", va="top", fontsize=8)
    ax.set_ylabel("units ('000)"); ax.set_title("Company monthly units: actual vs forecasts on the unseen test period"); ax.legend(ncol=4, fontsize=8)
    savefig(fig, "ml_company_forecast_test")

    # ---------------------------------------------------------------- explainability
    bm = fitted[best]
    pi = permutation_importance(bm, test[X_cols], np.log1p(test.units), n_repeats=10, random_state=C.SEED,
                                scoring="neg_root_mean_squared_error")
    imp = pd.Series(pi.importances_mean, index=X_cols).sort_values(ascending=False)
    out["permutation_importance"] = imp.head(12).to_dict()
    shap_imp = None
    xgb_name = "XGBoost"
    xp = fitted[xgb_name]
    Xt = xp.named_steps["prep"].transform(test[X_cols])
    import xgboost as xgb
    contrib = xp.named_steps["m"].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]
    names = xp.named_steps["prep"].get_feature_names_out()
    shap_imp = pd.Series(np.abs(contrib).mean(0), index=names).sort_values(ascending=False)
    out["xgb_shap_importance"] = shap_imp.head(12).to_dict()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    a = imp.head(12)[::-1]; ax[0].barh(a.index, a.values, color=C.PRIMARY)
    ax[0].set_title("Permutation importance (best model)"); ax[0].set_xlabel("RMSE increase when shuffled")
    b = shap_imp.head(12)[::-1]; ax[1].barh(b.index, b.values, color=C.ACCENT)
    ax[1].set_title("XGBoost mean |SHAP value|"); ax[1].set_xlabel("mean |contribution| (log units)")
    savefig(fig, "ml_feature_importance")

    # ---------------------------------------------------------------- future: Oct 2026 - Sep 2027
    last = units.index.max()
    fut_months = pd.date_range(last + pd.offsets.MonthBegin(1), periods=C.FORECAST_HORIZON, freq="MS")
    final = model_zoo()[best_rec]
    final_pipe = Pipeline([("prep", preprocessor(final[1])), ("m", fitted[best_rec].named_steps["m"])])
    F_all = F.reset_index(drop=True)
    final_pipe.fit(F_all[X_cols], np.log1p(F_all.units))           # refit on all data, same hyper-parameters
    joblib.dump(final_pipe, C.MODELS / "demand_forecaster.joblib")
    Rf = recursive_forecast(final_pipe, units, price, meta, fut_months)
    # empirical prediction interval from recursive test errors at each horizon (log ratio quantiles)
    mm = rec_preds[best_rec].copy()
    mm["h"] = mm.month.rank(method="dense").astype(int)
    mm["lr"] = np.log1p(mm.units) - np.log1p(mm.pred)
    tot = mm.groupby("month").agg(u=("units", "sum"), p=("pred", "sum"))
    tot_lr = np.log(tot.u / tot.p)
    lo_q, hi_q = float(np.quantile(tot_lr, .1)), float(np.quantile(tot_lr, .9))
    # net revenue per unit: last 3 months per product
    ym = d.Date.dt.to_period("M").dt.to_timestamp()
    rpu = d[ym > last - pd.DateOffset(months=3)].groupby("Product_ID", observed=True).apply(
        lambda g: g.Net_Sales_BDT.sum() / g.Quantity.sum(), include_groups=False)
    rpu.index = rpu.index.astype(str)
    fut = Rf.stack().rename("units_ml").reset_index(); fut.columns = ["month", "Product_ID", "units_ml"]
    fut = fut.merge(meta[["Product_Name", "Category"]], left_on="Product_ID", right_index=True)

    # Statistical models on the company totals (best at aggregate level in the test) with simulated intervals
    hw_u = ExponentialSmoothing(total_units, trend="add", seasonal="mul", seasonal_periods=12).fit(optimized=True)
    hw_s = ExponentialSmoothing(total_sales, trend="add", seasonal="mul", seasonal_periods=12).fit(optimized=True)
    sims_u = hw_u.simulate(C.FORECAST_HORIZON, repetitions=2000, error="mul", random_state=C.SEED)
    sims_s = hw_s.simulate(C.FORECAST_HORIZON, repetitions=2000, error="mul", random_state=C.SEED)
    ftot = pd.DataFrame({"units_ml": fut.groupby("month").units_ml.sum(),
                         "units": hw_u.forecast(C.FORECAST_HORIZON).values,
                         "units_lo": np.quantile(sims_u, .1, axis=1), "units_hi": np.quantile(sims_u, .9, axis=1),
                         "net_sales": hw_s.forecast(C.FORECAST_HORIZON).values,
                         "net_sales_lo": np.quantile(sims_s, .1, axis=1), "net_sales_hi": np.quantile(sims_s, .9, axis=1)})
    ftot.index.name = "month"
    # Top-down reconciliation: product shares from the ML model, totals from Holt-Winters
    # (so product plans add up exactly to the company plan).
    fut["share"] = fut.units_ml / fut.groupby("month").units_ml.transform("sum")
    fut["units"] = fut.share * fut.month.map(ftot.units)
    fut["value_ml"] = fut.units * fut.Product_ID.map(rpu)
    fut["net_sales"] = fut.value_ml / fut.groupby("month").value_ml.transform("sum") * fut.month.map(ftot.net_sales)
    fut = fut.drop(columns="value_ml")
    fut.to_csv(C.TABLES / "future_product_forecast.csv", index=False)
    ftot.reset_index().to_csv(C.TABLES / "future_total_forecast.csv", index=False)
    out["future"] = {"start": str(fut_months[0].date()), "end": str(fut_months[-1].date()),
                     "units": float(ftot.units.sum()), "units_ml_raw": float(ftot.units_ml.sum()),
                     "net_sales": float(ftot.net_sales.sum()),
                     "net_sales_lo": float(np.quantile(sims_s.sum(axis=0), .1)),
                     "net_sales_hi": float(np.quantile(sims_s.sum(axis=0), .9)),
                     "prev12_units": float(total_units[-12:].sum()), "prev12_sales": float(total_sales[-12:].sum()),
                     "peak_month": str(ftot.net_sales.idxmax().date()),
                     "interval_ml_lo_pct": float((np.exp(lo_q) - 1) * 100), "interval_ml_hi_pct": float((np.exp(hi_q) - 1) * 100)}
    fu = out["future"]
    fu["growth_units_pct"] = (fu["units"] / fu["prev12_units"] - 1) * 100
    fu["growth_units_ml_raw_pct"] = (fu["units_ml_raw"] / fu["prev12_units"] - 1) * 100
    fu["growth_sales_pct"] = (fu["net_sales"] / fu["prev12_sales"] - 1) * 100
    top_fut = fut.groupby("Category").net_sales.sum().sort_values(ascending=False)
    fu["top_category"] = top_fut.index[0]
    ft = ftot.reset_index()
    ft["Month"] = ft.month.dt.strftime("%b %Y")
    ft = ft[["Month", "units_ml", "units", "units_lo", "units_hi", "net_sales", "net_sales_lo", "net_sales_hi"]]
    for c in ["net_sales", "net_sales_lo", "net_sales_hi"]:
        ft[c] = ft[c] / 1e6
    ft.columns = ["Month", "Units ML raw", "Units (final)", "80% low", "80% high", "Net sales (M)", "80% low ", "80% high "]
    to_latex_table(ft, "future_forecast", "Twelve-month forward forecast, October 2026 -- September 2027", "tab:future",
                   fmt={c: "{:,.0f}" for c in ["Units ML raw", "Units (final)", "80% low", "80% high"]} |
                       {c: "{:.2f}" for c in ["Net sales (M)", "80% low ", "80% high "]}, font=r"\footnotesize",
                   note=f"Units ML raw = sum of the {best_rec} product forecasts before reconciliation. Final totals come "
                        "from Holt-Winters models of company units and net sales (BDT million); intervals are the 10th and "
                        "90th percentiles of 2,000 simulated future paths. Product forecasts are scaled to these totals.")

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    hist_s = total_sales[-36:] / 1e6
    ax.plot(hist_s.index, hist_s, color="black", lw=1.4, label="actual net sales")
    ax.plot(ftot.index, ftot.net_sales / 1e6, color=C.ACCENT, lw=1.8, marker="o", ms=3, label="forecast (Holt-Winters, reconciled)")
    ax.fill_between(ftot.index, ftot.net_sales_lo / 1e6, ftot.net_sales_hi / 1e6, color=C.ACCENT, alpha=.18, label="80% interval")
    ax.set_ylabel("BDT million"); ax.set_title("Net sales outlook, Oct 2026 - Sep 2027"); ax.legend(fontsize=8)
    savefig(fig, "ml_future_forecast")

    # data for the dashboard
    test.to_parquet(C.DASHBOARD_DATA / "forecast_test_predictions.parquet", index=False)
    rec_preds[best_rec].to_parquet(C.DASHBOARD_DATA / "forecast_test_recursive.parquet", index=False)
    fut.to_parquet(C.DASHBOARD_DATA / "forecast_future_products.parquet", index=False)
    ftot.reset_index().to_parquet(C.DASHBOARD_DATA / "forecast_future_total.parquet", index=False)
    res.to_parquet(C.DASHBOARD_DATA / "forecast_model_scores.parquet", index=False)
    save_json(out, "forecasting")
    print(f"[forecast] best one-step {best}, best recursive {best_rec}; "
          f"improvement vs seasonal naive {out['best_vs_seasonal_naive_pct']:.1f}%")
    return out
