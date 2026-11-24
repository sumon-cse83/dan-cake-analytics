# %% [markdown]
# # Step 7 – Predictive analytics (hands-on)  — proposal section 7.4
# Uses the files made by Step 6: features_train_2020_2024.csv and features_new_2025_2026.csv
# Run it:  python 07_predictive_analytics.py      (or click "Run Cell" above each "# %%" in VS Code)
#
# Three prediction problems, and the algorithms tried on each:
#   A. REGRESSION     - how many units will each product sell next month?
#                       Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM (SVR), KNN
#   B. CLASSIFICATION - will an order be a LARGE order (10 or more units)?
#                       Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM (SVC), KNN
#   C. CLUSTERING     - which products behave alike? (no answer column, so it is "unsupervised")
#                       K-Means
# RULE for A and B: train on 2020-2024, test on 2025-2026. The model never sees the test years while learning,
# exactly like real life (you train on the past and predict the future).

# %% 0. Imports and settings
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

SEED = 42
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

train = pd.read_csv("features_train_2020_2024.csv", parse_dates=["Date"])
new = pd.read_csv("features_new_2025_2026.csv", parse_dates=["Date"])
FEATS = json.load(open("feature_list.json"))["kept_features"]
print("train rows:", len(train), "| test rows:", len(new), "| features:", len(FEATS))


# %% 1. Helper: wrap a model so missing values are filled and (if needed) features are scaled
# Linear models, SVM and KNN measure DISTANCES or use weights, so a column in thousands (units_lag_1)
# would dominate a column between 0 and 1 (price_vs_list). StandardScaler puts every column on the same scale.
# Tree models (Decision Tree, Random Forest, Boosting, XGBoost) split on one column at a time -> no scaling needed.
def build(model, scale):
    steps = [SimpleImputer(strategy="median")]          # median of the TRAINING data fills empty cells
    if scale:
        steps.append(StandardScaler())
    return make_pipeline(*steps, model)


# %% 2. PROBLEM A - Regression: monthly units per product
# Turn orders into one row per product per month. The target is that month's total units.
both = pd.concat([train, new])
both["month"] = both["Date"].dt.month
pm = both.groupby(["Product_ID", "month_index"]).agg(
    units=("Quantity", "sum"),
    year=("Date", lambda s: s.dt.year.iloc[0]),
    month=("month", "first"),
    is_december=("is_december", "first"),
    product_age_years=("product_age_years", "first"),
    log_price=("log_price", "mean"),
    discount=("Discount_%", "mean"),
    pack_grams=("pack_grams", "first"),
    units_lag_1=("units_lag_1", "first"),          # last month's units      (known before the month starts)
    units_lag_12=("units_lag_12", "first"),        # same month last year    (known)
    units_roll3_prev=("units_roll3_prev", "first"),  # average of last 3 months (known)
).reset_index()
pm = pm.dropna(subset=["units_lag_12"])           # the first 12 months of each product have no "last year"
REG_FEATS = ["month", "is_december", "product_age_years", "log_price", "discount", "pack_grams",
             "units_lag_1", "units_lag_12", "units_roll3_prev"]
tr_m, te_m = pm[pm.year <= 2024], pm[pm.year >= 2025]
print(f"\nA. product-month rows: train {len(tr_m)}, test {len(te_m)}, products {pm.Product_ID.nunique()}")

# SVR also needs the TARGET scaled (units are in the thousands) -> TransformedTargetRegressor does that and undoes it.
reg_models = {
    "Linear Regression": build(LinearRegression(), scale=True),
    "Decision Tree": build(DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=SEED), scale=False),
    "Random Forest": build(RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED), scale=False),
    "Gradient Boosting": build(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED), scale=False),
    "XGBoost": build(XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=SEED), scale=False),
    "SVM (SVR)": TransformedTargetRegressor(build(SVR(C=10, epsilon=0.05), scale=True), transformer=StandardScaler()),
    "KNN": build(KNeighborsRegressor(n_neighbors=10, weights="distance"), scale=True),
}

def reg_scores(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "MAPE_%": float(np.mean(np.abs(y - p) / y) * 100), "R2": r2_score(y, p)}

reg_rows = [{"model": "Baseline: same as last month", **reg_scores(te_m.units, te_m.units_lag_1)}]
for name, model in reg_models.items():
    t = time.time()
    model.fit(tr_m[REG_FEATS], tr_m.units)
    reg_rows.append({"model": name, **reg_scores(te_m.units, model.predict(te_m[REG_FEATS])), "seconds": time.time() - t})
reg = pd.DataFrame(reg_rows).sort_values("MAE")
print("\nA. Regression results on 2025-2026 (lower MAE / RMSE / MAPE is better, higher R2 is better):")
print(reg.round(3).to_string(index=False))
# A model is only useful if it beats the simple baseline ("next month = this month").


# %% 3. PROBLEM B - Classification: is this a large order (Quantity >= 10)?
print("\nB. share of large orders - train:", round(train.large_order.mean(), 3), "| test:", round(new.large_order.mean(), 3))
# SVM and KNN get very slow on hundreds of thousands of rows (SVM time grows roughly with rows squared).
# So every model is trained on the SAME random 15,000 training orders and tested on the SAME 15,000 test orders -
# a fair comparison. (Once the best model is chosen you can retrain it on all rows.)
N = 15_000                                     # raise it for slightly better scores; SVM time grows fast
tr_o = train.sample(N, random_state=SEED)
te_o = new.sample(N, random_state=SEED)
pos_weight = (tr_o.large_order == 0).sum() / (tr_o.large_order == 1).sum()   # for XGBoost: how much rarer is class 1

clf_models = {
    "Logistic Regression": build(LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True),
    "Decision Tree": build(DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=SEED), scale=False),
    "Random Forest": build(RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=SEED), scale=False),
    "Gradient Boosting": build(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED), scale=False),
    "XGBoost": build(XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                   scale_pos_weight=pos_weight, eval_metric="logloss", random_state=SEED), scale=False),
    "SVM (SVC)": build(SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED), scale=True),
    "KNN": build(KNeighborsClassifier(n_neighbors=25), scale=True),
}

def clf_scores(y, pred, score):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "F1": f1_score(y, pred), "ROC_AUC": roc_auc_score(y, score)}

majority = int(tr_o.large_order.mean() >= 0.5)
clf_rows = [{"model": "Baseline: always the most common class",
             **clf_scores(te_o.large_order, np.full(len(te_o), majority), np.zeros(len(te_o)))}]
for name, model in clf_models.items():
    t = time.time()
    model.fit(tr_o[FEATS], tr_o.large_order)
    pred = model.predict(te_o[FEATS])
    # ROC-AUC needs a SCORE (how sure the model is), not just 0/1. SVC gives decision_function instead of probabilities.
    score = model.predict_proba(te_o[FEATS])[:, 1] if hasattr(model, "predict_proba") else model.decision_function(te_o[FEATS])
    clf_rows.append({"model": name, **clf_scores(te_o.large_order, pred, score), "seconds": time.time() - t})
clf = pd.DataFrame(clf_rows).sort_values("ROC_AUC", ascending=False)
print("\nB. Classification results on 2025-2026 (higher is better; ROC-AUC 0.5 = guessing, 1.0 = perfect):")
print(clf.round(3).to_string(index=False))
# Accuracy alone can fool you when one class is common: the baseline can score high accuracy with F1 = 0.

# Which features did the best tree model rely on?
best_tree = clf_models["Random Forest"][-1]
imp = pd.Series(best_tree.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\nB. Random Forest - top 5 features:\n", imp.head(5).round(3).to_string())


# %% 4. PROBLEM C - K-Means clustering: group products by how they sell (training years only)
y23 = train[train.Date.dt.year == 2023].groupby("Product_ID").Quantity.sum()
y24 = train[train.Date.dt.year == 2024].groupby("Product_ID").Quantity.sum()
prof = train.groupby("Product_ID").agg(
    avg_price=("log_price", lambda s: np.exp(s).mean()),
    total_units=("Quantity", "sum"),
    avg_order_qty=("Quantity", "mean"),
    avg_discount=("Discount_%", "mean"),
    large_order_share=("large_order", "mean"),
)
prof["growth_2024"] = (y24 / y23 - 1).reindex(prof.index).fillna(0)
Xc = StandardScaler().fit_transform(prof)          # K-Means uses distance -> scaling is a must

# Choose k: try 2..8 clusters, keep the one with the best silhouette score (-1 bad ... +1 well separated)
sil = {k: silhouette_score(Xc, KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Xc))
       for k in range(2, min(9, len(prof)))}
best_k = max(sil, key=sil.get)
print("\nC. silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, "-> best k =", best_k)
km = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit(Xc)
prof["cluster"] = km.labels_
print("\nC. Average profile of each cluster (read it to give each cluster a business name):")
print(prof.groupby("cluster").mean().round(2).assign(products=prof.cluster.value_counts().sort_index()).to_string())
# How to read it: if one cluster holds only a few products, check what makes them different (here: much lower
# total_units, i.e. products launched recently with fewer months of sales). That is a real finding, but if you
# want groups by SELLING STYLE instead, drop total_units / growth_2024 from `prof` and run this cell again.


# %% 5. One picture with all three results
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
r = reg.set_index("model")["MAE"].sort_values()
ax[0].barh(r.index, r.values, color=["#2a9d8f" if "Baseline" not in i else "#999" for i in r.index])
ax[0].invert_yaxis(); ax[0].set_title("A. Monthly units per product\nMAE on 2025-26 (lower is better)")
c = clf.set_index("model")["ROC_AUC"].sort_values(ascending=False)
ax[1].barh(c.index, c.values, color=["#e76f51" if "Baseline" not in i else "#999" for i in c.index])
ax[1].invert_yaxis(); ax[1].set_xlim(0.4, 1); ax[1].axvline(0.5, ls="--", c="k", lw=0.8)
ax[1].set_title("B. Large order (>=10 units)\nROC-AUC on 2025-26 (higher is better)")
xy = PCA(n_components=2, random_state=SEED).fit_transform(Xc)       # squeeze 6 columns into 2 just to draw them
sc = ax[2].scatter(xy[:, 0], xy[:, 1], c=prof.cluster, cmap="tab10", s=60)
ax[2].legend(*sc.legend_elements(), title="cluster"); ax[2].set_title(f"C. K-Means product clusters (k={best_k})\nPCA view")
ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2")
plt.tight_layout()
plt.savefig("predictive_model_comparison.png", dpi=120)
print("\nSaved chart: predictive_model_comparison.png")


# %% 6. Save the numbers
out = {"regression": reg.round(4).to_dict("records"), "classification": clf.round(4).to_dict("records"),
       "kmeans": {"silhouette": {str(k): round(v, 4) for k, v in sil.items()}, "best_k": int(best_k),
                  "product_cluster": prof["cluster"].to_dict()}}
json.dump(out, open("predictive_results.json", "w"), indent=2)
print("Saved: predictive_results.json")
print(f"\nSUMMARY: best regression = {reg.iloc[0].model} (MAE {reg.iloc[0].MAE:.1f} units) | "
      f"best classifier = {clf.iloc[0].model} (ROC-AUC {clf.iloc[0].ROC_AUC:.3f}) | K-Means k = {best_k}")
