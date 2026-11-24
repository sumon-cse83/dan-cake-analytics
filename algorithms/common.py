"""Shared helpers for the 9 algorithm lessons (01_linear_regression.py ... 09_kmeans_clustering.py).

Every lesson needs the same data and the same scoring, so that code lives here once.
Read this file first - it is short - then open the lessons in order.

Data comes from Step 6 (06_feature_engineering.py):
    features_train_2020_2024.csv   orders 2020-2024  -> used to LEARN
    features_new_2025_2026.csv     orders 2025-2026  -> used only to TEST (the "future")

Two prediction problems are used in the lessons:
    A. REGRESSION     (a number)  : how many units will a product sell in a month?      -> load_monthly()
    B. CLASSIFICATION (a yes/no)  : will an order be LARGE (10 units or more)?          -> load_orders()
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent          # .../algorithms
ROOT = HERE.parent                              # .../dan_cake_project
OUT = HERE / "outputs"                          # charts and results go here
OUT.mkdir(exist_ok=True)
SEED = 42                                       # same "random" numbers every run -> repeatable results

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# Order-level input columns chosen in Step 6
ORDER_FEATS = json.load(open(ROOT / "feature_list.json"))["kept_features"]
# Product-month input columns (everything here is known BEFORE the month starts)
MONTH_FEATS = ["month", "is_december", "product_age_years", "log_price", "discount", "pack_grams",
               "units_lag_1", "units_lag_12", "units_roll3_prev"]


def read_orders():
    """All orders: (train 2020-2024, new 2025-2026), each with a 'year' column."""
    tr = pd.read_csv(ROOT / "features_train_2020_2024.csv", parse_dates=["Date"])
    te = pd.read_csv(ROOT / "features_new_2025_2026.csv", parse_dates=["Date"])
    for d in (tr, te):
        d["year"] = d["Date"].dt.year
    return tr, te


def load_monthly():
    """Problem A table: one row per product per month. Target column = 'units'."""
    tr, te = read_orders()
    both = pd.concat([tr, te])
    both["month"] = both["Date"].dt.month
    pm = both.groupby(["Product_ID", "month_index"]).agg(
        units=("Quantity", "sum"),                      # TARGET: total units that month
        year=("year", "first"),
        month=("month", "first"),
        is_december=("is_december", "first"),
        product_age_years=("product_age_years", "first"),
        log_price=("log_price", "mean"),
        discount=("Discount_%", "mean"),
        pack_grams=("pack_grams", "first"),
        units_lag_1=("units_lag_1", "first"),           # units last month
        units_lag_12=("units_lag_12", "first"),         # units same month last year
        units_roll3_prev=("units_roll3_prev", "first"),  # average of the previous 3 months
    ).reset_index()
    pm = pm.dropna(subset=["units_lag_12"]).reset_index(drop=True)   # first 12 months have no "last year"
    med = pm.loc[pm.year <= 2024, MONTH_FEATS].median()             # fill gaps with TRAINING medians only
    pm[MONTH_FEATS] = pm[MONTH_FEATS].fillna(med)
    return pm


def load_orders(n=15_000):
    """Problem B table: a random sample of n training orders + n test orders. Target column = 'large_order'.

    Why a sample? SVM and KNN become very slow on 500,000 rows. Every lesson uses the SAME sample, so the
    algorithms are compared fairly.
    """
    tr, te = read_orders()
    tr = tr.sample(n, random_state=SEED)
    te = te.sample(n, random_state=SEED)
    med = tr[ORDER_FEATS].median()                      # empty cells (e.g. no history yet) -> training median
    tr[ORDER_FEATS] = tr[ORDER_FEATS].fillna(med)
    te[ORDER_FEATS] = te[ORDER_FEATS].fillna(med)
    return pd.concat([tr, te]).reset_index(drop=True)


def split_years(df):
    """Time-based split. Never shuffle time data - the model must not see the future.

    fit   2020-2023 : learn, while we are still TUNING settings
    val   2024      : check settings (validation)  -> choose the best setting here, never on the test
    train 2020-2024 : learn again with the chosen setting
    test  2025-2026 : final, honest score - looked at once
    """
    return {"fit": df[df.year <= 2023], "val": df[df.year == 2024],
            "train": df[df.year <= 2024], "test": df[df.year >= 2025]}


def regression_scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": round(mean_absolute_error(y, p), 2),          # average error, in units
            "RMSE": round(mean_squared_error(y, p) ** 0.5, 2),   # like MAE but punishes big misses more
            "MAPE_%": round(float(np.mean(np.abs(y - p) / y) * 100), 2),   # average error in percent
            "R2": round(r2_score(y, p), 3)}                      # 1 = perfect, 0 = no better than the average


def classification_scores(y, pred, score=None):
    d = {"accuracy": round(accuracy_score(y, pred), 3),                          # share of correct answers
         "precision": round(precision_score(y, pred, zero_division=0), 3),       # of predicted "large", % truly large
         "recall": round(recall_score(y, pred, zero_division=0), 3),             # of truly large, % we caught
         "F1": round(f1_score(y, pred, zero_division=0), 3)}                     # balance of precision and recall
    if score is not None:
        d["ROC_AUC"] = round(roc_auc_score(y, score), 3)   # ranking quality: 0.5 = guessing, 1 = perfect
    return d


def show(results: dict, title: str):
    """Print {model name: scores dict} as a table."""
    print(f"\n{title}")
    print(pd.DataFrame(results).T.to_string())


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    print("Saved chart: algorithms/outputs/" + name)


def save_result(algorithm, problem, scores):
    """Add this lesson's final test score to algorithms/outputs/summary.json (to compare all lessons)."""
    f = OUT / "summary.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data.setdefault(problem, {})[algorithm] = scores
    f.write_text(json.dumps(data, indent=2))
    print(f"Saved result: algorithms/outputs/summary.json  [{problem} / {algorithm}]")
