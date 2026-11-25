"""Layer 6d: automated anomaly detection.

(a) Daily business monitoring: which days had unusual sales once trend and season are removed?
    Robust z-score of the residual (median/MAD) is compared with an Isolation Forest.
(b) Transaction screening: an Isolation Forest scores every order. Because the real file has no
    labelled errors, its ability is measured on a copy with 1,000 injected errors of known type
    (the same "damage a copy, keep the answer key" approach used in the preprocessing scripts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

from . import config as C
from .data import flag_outliers_log_iqr
from .utils import plt, save_json, savefig, to_latex_table


def daily_anomalies(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    day = d.groupby("Date").agg(net=("Net_Sales_BDT", "sum"), orders=("Quantity", "size"),
                                units=("Quantity", "sum"), disc=("Discount_%", "mean"))
    day = day.asfreq("D").fillna(0)
    # expected level: centred 31-day rolling median (trend) x month-of-year seasonal factor
    trend = day.net.rolling(31, center=True, min_periods=10).median()
    ratio = day.net / trend
    seas = ratio.groupby(ratio.index.month).transform("median")
    day["expected"] = trend * seas
    day["resid"] = np.log(day.net.clip(lower=1) / day.expected)
    med, mad = day.resid.median(), (day.resid - day.resid.median()).abs().median()
    day["robust_z"] = 0.6745 * (day.resid - med) / mad
    day["flag_z"] = (day.robust_z.abs() > 3.5).astype(int)
    feats = pd.DataFrame({"resid": day.resid, "orders_r": np.log(day.orders.clip(lower=1) / day.orders.rolling(31, center=True, min_periods=10).median()),
                          "upo": day.units / day.orders.clip(lower=1), "disc": day.disc}).fillna(0)
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=C.SEED).fit(feats)
    day["iso_score"] = -iso.score_samples(feats)
    day["flag_iso"] = (iso.predict(feats) == -1).astype(int)
    day["flag"] = ((day.flag_z + day.flag_iso) > 0).astype(int)
    day["direction"] = np.where(day.resid > 0, "spike", "dip")
    info = {"days": int(len(day)), "flag_z": int(day.flag_z.sum()), "flag_iso": int(day.flag_iso.sum()),
            "flag_both": int(((day.flag_z == 1) & (day.flag_iso == 1)).sum()), "flag_any": int(day.flag.sum())}
    return day, info


def inject(d: pd.DataFrame, rng) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Copy the data and plant 1,000 realistic errors of four kinds."""
    x = d[["Product_ID", "Quantity", "Unit_Selling_Price_BDT", "List_Price_BDT", "Discount_%", "Net_Sales_BDT"]].copy()
    x["Product_ID"] = x.Product_ID.astype(str)
    idx = rng.choice(len(x), 1000, replace=False)
    kinds = np.repeat(["price x10 (typo)", "quantity x20 (extra zero)", "discount 30-45%", "price at 10% of list"], 250)
    for i, k in zip(idx, kinds):
        if k.startswith("price x10"):
            x.iat[i, 2] *= 10
        elif k.startswith("quantity"):
            x.iat[i, 1] *= 20
        elif k.startswith("discount"):
            x.iat[i, 4] = rng.uniform(30, 45)
        else:
            x.iat[i, 2] = x.iat[i, 3] * 0.1
    x["Net_Sales_BDT"] = x.Quantity * x.Unit_Selling_Price_BDT
    label = np.zeros(len(x), int); label[idx] = 1
    kind = pd.Series("normal", index=x.index); kind.iloc[idx] = kinds
    return x, label, kind


def transaction_features(x: pd.DataFrame) -> pd.DataFrame:
    g = x.groupby("Product_ID")
    return pd.DataFrame({
        "log_qty_rel": np.log1p(x.Quantity) - g.Quantity.transform(lambda s: np.log1p(s).median()),
        "price_vs_list": np.log(x.Unit_Selling_Price_BDT / x.List_Price_BDT),
        "discount": x["Discount_%"],
        "log_value_rel": np.log(x.Net_Sales_BDT) - g.Net_Sales_BDT.transform(lambda s: np.log(s).median()),
    })


def run(d: pd.DataFrame) -> dict:
    out: dict = {}
    day, info = daily_anomalies(d)
    out["daily"] = info
    top = day[day.flag == 1].sort_values("robust_z", key=np.abs, ascending=False).head(10).reset_index()
    t = pd.DataFrame({"Date": top.Date.dt.strftime("%d %b %Y"), "Net sales (BDT)": top.net, "Expected (BDT)": top.expected,
                      "Deviation %": (top.net / top.expected - 1) * 100, "Robust z": top.robust_z,
                      "Flagged by": np.where((top.flag_z == 1) & (top.flag_iso == 1), "both",
                                             np.where(top.flag_z == 1, "robust z", "Isolation Forest"))})
    to_latex_table(t, "daily_anomalies", "Ten most unusual trading days (seasonally adjusted)", "tab:dailyanom",
                   fmt={"Net sales (BDT)": "{:,.0f}", "Expected (BDT)": "{:,.0f}", "Deviation %": "{:+.1f}", "Robust z": "{:+.2f}"},
                   col_align="lrrrrl")
    out["top_days"] = t.to_dict("records")
    day.reset_index().to_parquet(C.DASHBOARD_DATA / "daily_anomalies.parquet", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(day.index, day.net / 1e3, lw=.5, color="#9DB4CC", label="daily net sales")
    ax.plot(day.index, day.expected / 1e3, lw=1.2, color=C.PRIMARY, label="expected (trend x season)")
    f = day[day.flag == 1]
    ax.scatter(f.index, f.net / 1e3, color=C.ACCENT, s=16, zorder=3, label=f"flagged days ({len(f)})")
    ax.set_ylabel("BDT '000"); ax.set_title("Daily sales monitoring and detected anomalies"); ax.legend(fontsize=8, ncol=3)
    savefig(fig, "anomaly_daily")

    # (b) transaction screening, evaluated on injected errors
    rng = np.random.default_rng(C.SEED)
    x, label, kind = inject(d, rng)
    feats = transaction_features(x)
    iso = IsolationForest(n_estimators=300, max_samples=50_000, contamination=0.005, random_state=C.SEED, n_jobs=-1).fit(feats)
    pred_iso = (iso.predict(feats) == -1).astype(int)
    pred_iqr = flag_outliers_log_iqr(x.Quantity, x.Product_ID).astype(int).values
    pv = feats.price_vs_list
    pred_rule = ((pv.abs() > np.log(1.5)) | (x["Discount_%"] > 20)).astype(int).values      # simple business rules
    rows = []
    for name, p in [("IQR on log quantity (per product)", pred_iqr), ("Business rules (price, discount)", pred_rule),
                    ("Isolation Forest (4 features)", pred_iso), ("Rules OR Isolation Forest", np.maximum(pred_rule, pred_iso))]:
        r = {"Method": name, "Flagged": int(p.sum()), "Precision": precision_score(label, p), "Recall": recall_score(label, p)}
        for k in ["price x10 (typo)", "quantity x20 (extra zero)", "discount 30-45%", "price at 10% of list"]:
            r[k] = float(p[(kind == k).values].mean())
        rows.append(r)
    res = pd.DataFrame(rows)
    out["transaction_detection"] = res.to_dict("records")
    tt = res.copy()
    tt.columns = ["Method", "Flagged", "Precision", "Recall", "Price x10", "Qty x20", "Disc. 30-45%", "Price 10%"]
    to_latex_table(tt, "anomaly_detection", "Detection of 1,000 injected transaction errors in 500,000 orders",
                   "tab:anomaly", fmt={"Flagged": "{:,.0f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "Price x10": "{:.2f}",
                                       "Qty x20": "{:.2f}", "Disc. 30-45%": "{:.2f}", "Price 10%": "{:.2f}"},
                   font=r"\footnotesize", note="The last four columns give the share of each error type that was detected. "
                                              "250 errors of each type were planted in a copy of the data.")
    # score the real data and store the top suspicious orders for the dashboard
    real = d[["Product_ID", "Quantity", "Unit_Selling_Price_BDT", "List_Price_BDT", "Discount_%", "Net_Sales_BDT"]].copy()
    real["Product_ID"] = real.Product_ID.astype(str)
    rf = transaction_features(real)
    score = -iso.score_samples(rf)
    susp = d.assign(anomaly_score=score).nlargest(300, "anomaly_score")
    susp[["Transaction_ID", "Date", "Product_ID", "Product_Name", "Region", "Channel", "Quantity", "Unit_Selling_Price_BDT",
          "Discount_%", "Net_Sales_BDT", "anomaly_score"]].astype({"Product_ID": str, "Product_Name": str, "Region": str,
                                                                    "Channel": str}).to_parquet(
        C.DASHBOARD_DATA / "suspicious_orders.parquet", index=False)
    out["real_flagged_iso"] = int((iso.predict(rf) == -1).sum())
    save_json(out, "anomaly")
    print(f"[anomaly] daily flagged {info['flag_any']}; transaction recall IF {res.Recall.iloc[2]:.2f}, combined {res.Recall.iloc[3]:.2f}")
    return out
