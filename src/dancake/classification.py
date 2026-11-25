"""Layer 6b: predictive analytics - classification of large orders.

Question: at the moment an order is entered, can we tell whether it will be a *large order*
(>= 10 units)? Useful for stock allocation and delivery planning.
Only order-time information is used (who, where, which product, price and calendar).
Train 2020-2024, test 2025-2026; the decision threshold is tuned on 2024 (validation),
never on the test set.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config as C
from .features import pack_grams
from .utils import plt, save_json, savefig, to_latex_table

warnings.filterwarnings("ignore")
CAT = ["Channel", "Region", "Customer_Type", "Category"]
NUM = ["log_price", "Discount_%", "price_vs_list", "pack_grams", "is_gift_pack", "month_sin", "month_cos",
       "is_december", "day_of_week", "product_age", "chan_region_rate", "product_rate"]


def order_features(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    for c in CAT:
        X[c] = d[c].astype(str)
    X["Product_ID"] = d.Product_ID.astype(str)
    X["log_price"] = np.log(d.Unit_Selling_Price_BDT)
    X["Discount_%"] = d["Discount_%"]
    X["price_vs_list"] = d.Unit_Selling_Price_BDT / d.List_Price_BDT
    packs = d.Pack_Size.astype(str)
    X["pack_grams"] = packs.map({p: pack_grams(p) for p in packs.unique()}).fillna(0)
    X["is_gift_pack"] = packs.str.contains(r"x\s*\d+|12 pcs").astype(int)
    X["month_sin"] = np.sin(2 * np.pi * d.Month / 12)
    X["month_cos"] = np.cos(2 * np.pi * d.Month / 12)
    X["is_december"] = (d.Month == 12).astype(int)
    X["day_of_week"] = d.Date.dt.dayofweek
    X["product_age"] = d.Year - d.Product_Launch_Year
    X["Year"] = d.Year
    return X


def add_target_encoding(tr: pd.DataFrame, ytr, others: list[pd.DataFrame], m: float = 50):
    """Smoothed large-order rate per channel x region and per product, learned on the training rows only."""
    gm = ytr.mean()
    for key, col in [(tr.Channel + "|" + tr.Region, "chan_region_rate"), (tr.Product_ID, "product_rate")]:
        g = pd.DataFrame({"k": key, "y": ytr}).groupby("k").y.agg(["mean", "count"])
        enc = (g["count"] * g["mean"] + m * gm) / (g["count"] + m)
        tr[col] = key.map(enc)
        for o in others:
            ok = o.Channel + "|" + o.Region if col == "chan_region_rate" else o.Product_ID
            o[col] = ok.map(enc).fillna(gm)


def prep(scale: bool):
    num = [("sc", StandardScaler())] if scale else []
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
                              ("num", Pipeline(num) if num else "passthrough", NUM)], verbose_feature_names_out=False)


def zoo():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"), True, None),
        "Decision Tree": (DecisionTreeClassifier(max_depth=8, min_samples_leaf=200, class_weight="balanced",
                                                 random_state=C.SEED), False, None),
        "Random Forest": (RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=100, max_features=0.4,
                                                 class_weight="balanced_subsample", n_jobs=-1, random_state=C.SEED), False, None),
        "Gradient Boosting (hist)": (HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                                     l2_regularization=1.0, random_state=C.SEED), False, None),
        "XGBoost": (XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                  min_child_weight=20, tree_method="hist", n_jobs=4, random_state=C.SEED,
                                  eval_metric="logloss", verbosity=0), False, None),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1), True, 60_000),
        "Support Vector Machine (linear)": (CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3), True, None),
    }


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    f1 = [f1_score(y, p >= t) for t in grid]
    return float(grid[int(np.argmax(f1))])


def run(d: pd.DataFrame) -> dict:
    out: dict = {}
    X = order_features(d)
    y = (d.Quantity >= C.LARGE_ORDER_UNITS).astype(int)
    tr_mask, te_mask = X.Year <= 2024, X.Year >= 2025
    fit_mask, val_mask = X.Year <= 2023, X.Year == 2024
    out["positive_rate_train"], out["positive_rate_test"] = float(y[tr_mask].mean()), float(y[te_mask].mean())
    out["n_train"], out["n_test"] = int(tr_mask.sum()), int(te_mask.sum())

    rows, curves, cms = [], {}, {}
    rng = np.random.default_rng(C.SEED)
    for name, (est, scale, sub) in zoo().items():
        t0 = time.time()
        # --- step 1: fit on 2020-2023, pick threshold on 2024
        Xf, Xv = X[fit_mask].copy(), X[val_mask].copy()
        yf, yv = y[fit_mask], y[val_mask]
        if sub:
            idx = rng.choice(len(Xf), sub, replace=False); Xf, yf = Xf.iloc[idx], yf.iloc[idx]
        add_target_encoding(Xf, yf, [Xv])
        pipe = Pipeline([("prep", prep(scale)), ("m", est)])
        pipe.fit(Xf, yf)
        thr = best_threshold(yv, pipe.predict_proba(Xv)[:, 1])
        # --- step 2: refit on all of 2020-2024 and score the untouched test years
        Xtr, Xte = X[tr_mask].copy(), X[te_mask].copy()
        ytr, yte = y[tr_mask], y[te_mask]
        if sub:
            idx = rng.choice(len(Xtr), sub, replace=False); Xtr, ytr = Xtr.iloc[idx], ytr.iloc[idx]
        add_target_encoding(Xtr, ytr, [Xte])
        pipe = Pipeline([("prep", prep(scale)), ("m", est)])
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        yhat = (p >= thr).astype(int)
        order = np.argsort(-p)
        top10 = yte.values[order[: len(p) // 10]].mean()
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, yhat), "Precision": precision_score(yte, yhat),
                     "Recall": recall_score(yte, yhat), "F1": f1_score(yte, yhat), "ROC-AUC": roc_auc_score(yte, p),
                     "PR-AUC": average_precision_score(yte, p), "Threshold": thr,
                     "Lift@10%": top10 / yte.mean(), "Time (s)": time.time() - t0})
        curves[name] = roc_curve(yte, p)[:2]
        cms[name] = confusion_matrix(yte, yhat)
        print(f"   [classify] {name:32s} AUC {rows[-1]['ROC-AUC']:.3f}  F1 {rows[-1]['F1']:.3f}  ({rows[-1]['Time (s)']:.0f}s)")
        if name == "XGBoost":
            xgb_pipe, Xte_x = pipe, Xte
    res = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    base_acc = 1 - y[te_mask].mean()
    out["majority_baseline_accuracy"] = float(base_acc)
    out["results"] = res.to_dict("records")
    out["best_model"] = res.Model.iloc[0]
    to_latex_table(res, "classification_models", r"Large-order classification ($\geq$10 units): test-set performance, "
                   "2025--2026", "tab:classification",
                   fmt={k: "{:.3f}" for k in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]} |
                       {"Threshold": "{:.2f}", "Lift@10%": "{:.2f}", "Time (s)": "{:.0f}"}, font=r"\footnotesize",
                   note=f"Positive rate in the test years: {y[te_mask].mean():.1%}; a classifier that always predicts "
                        f"`not large' would reach {base_acc:.1%} accuracy, so accuracy alone is misleading. Thresholds "
                        "maximise F1 on the 2024 validation year. KNN uses a 60,000-row training sample. Lift@10\\% is the "
                        "large-order rate among the 10\\% highest-scored orders divided by the overall rate.")

    best = res.Model.iloc[0]
    # two panels (the metrics of all models are in the table next to the figure)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.25, 1]})
    for name, (fpr, tpr) in curves.items():
        ax[0].plot(fpr, tpr, lw=1.6 if name == best else 1, label=f"{name} ({res.set_index('Model').loc[name, 'ROC-AUC']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.7); ax[0].set_title("ROC curves (test)"); ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    cm = cms[best]
    ax[1].imshow(cm, cmap="Blues"); ax[1].grid(False)
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[1].set_xticks([0, 1], ["pred. normal", "pred. large"]); ax[1].set_yticks([0, 1], ["normal", "large"])
    ax[1].set_title(f"Confusion matrix: {best}")
    savefig(fig, "ml_classification")

    # explainability: XGBoost gain importance aggregated back to the original variables
    booster = xgb_pipe.named_steps["m"]
    names = xgb_pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(booster.feature_importances_, index=names)
    agg = imp.groupby(lambda s: next((c for c in CAT if s.startswith(c + "_")), s)).sum().sort_values(ascending=False)
    out["xgb_importance"] = agg.to_dict()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    a = agg.head(12)[::-1]; ax.barh(a.index, a.values, color=C.PRIMARY)
    ax.set_title("What drives a large order? (XGBoost gain)"); ax.set_xlabel("share of total gain")
    savefig(fig, "ml_classification_importance")

    # Large-order probability by channel (business reading of the model)
    dd = d[te_mask].assign(p=xgb_pipe.predict_proba(Xte_x)[:, 1], y=y[te_mask].values)
    out["large_rate_by_channel"] = dd.groupby("Channel", observed=True).y.mean().to_dict()
    save_json(out, "classification")
    print(f"[classify] best {best} AUC {res['ROC-AUC'].iloc[0]:.3f}")
    return out
