# %% [markdown]
# # Algorithm 8 – K-Nearest Neighbours (KNN)
# QUESTION (problem B): will an order be LARGE (10+ units)?   (Step 6 also does problem A)
# IDEA: no real "training". To predict a new order, find the k MOST SIMILAR past orders (smallest distance)
#       and let them vote (classification) or take their average (regression).
#       k small -> follows every detail, noisy (overfits).  k large -> smooth, may miss real patterns.
# Good:  very simple idea, no assumptions about the shape of the data.
# Weak:  needs scaling; slow to PREDICT on big data (compares with every stored row); weak with many columns.
# Run:   python algorithms/08_knn.py

# %% Step 1. Load the data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (MONTH_FEATS, ORDER_FEATS, classification_scores, load_monthly, load_orders,
                    regression_scores, save_result, savefig, show, split_years)

s = split_years(load_orders())
train, test = s["train"], s["test"]
print("rows: train", len(train), "| test", len(test))

# %% Step 2. Why scaling matters (KNN is all about distance)
def auc_2024(model):
    model.fit(s["fit"][ORDER_FEATS], s["fit"]["large_order"])
    return roc_auc_score(s["val"]["large_order"], model.predict_proba(s["val"][ORDER_FEATS])[:, 1])

print(f"\nk=25 without scaling: AUC on 2024 = {auc_2024(KNeighborsClassifier(n_neighbors=25)):.3f}")
print(f"k=25 with scaling   : AUC on 2024 = {auc_2024(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25))):.3f}")
# Without scaling, "distance" is almost only units_lag_1 (values in thousands) - other columns are ignored.

# %% Step 3. Choose k on 2024
ks = [1, 3, 5, 11, 25, 51, 101, 201, 401]
rows = []
for k in ks:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    m.fit(s["fit"][ORDER_FEATS], s["fit"]["large_order"])
    rows.append({"k": k,
                 "AUC_learned_data": roc_auc_score(s["fit"]["large_order"], m.predict_proba(s["fit"][ORDER_FEATS])[:, 1]),
                 "AUC_2024": roc_auc_score(s["val"]["large_order"], m.predict_proba(s["val"][ORDER_FEATS])[:, 1])})
tune = pd.DataFrame(rows)
print("\n" + tune.round(3).to_string(index=False))
best_k = int(tune.loc[tune.AUC_2024.idxmax(), "k"])
print("best k:", best_k)
# If the best k is the LAST value tried, the score might still improve with a larger k - try a longer list.
# Here a very large k works best: order size is mostly noise, so averaging over many neighbours helps.
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(tune.k, tune.AUC_learned_data, "o-", label="data it learned from (2020-23)")
ax.plot(tune.k, tune.AUC_2024, "o-", label="unseen data (2024)")
ax.set_xscale("log"); ax.axvline(best_k, ls="--", c="grey")
ax.set_xlabel("k (number of neighbours)"); ax.set_ylabel("ROC-AUC"); ax.legend()
ax.set_title("k=1 memorises (perfect on its own data); larger k generalises")
savefig(fig, "08_choose_k.png")

# %% Step 4. Final model on 2020-2024, test on 2025-2026
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
knn.fit(train[ORDER_FEATS], train["large_order"])
prob = knn.predict_proba(test[ORDER_FEATS])[:, 1]
thr = train["large_order"].mean()     # like lesson 5: with a rare class, 0.5 rarely says "large"
results = {
    "Baseline (always 'small')": classification_scores(test.large_order, np.zeros(len(test), int), np.zeros(len(test))),
    "KNN, threshold 0.5": classification_scores(test.large_order, (prob >= 0.5).astype(int), prob),
    f"KNN, threshold {thr:.2f}": classification_scores(test.large_order, (prob >= thr).astype(int), prob),
}
show(results, "Test results 2025-2026 (higher = better):")

# %% Step 5. Look inside one prediction: who are the neighbours?
order = test.iloc[[0]]
dist, idx = knn[-1].kneighbors(knn[0].transform(order[ORDER_FEATS]), n_neighbors=10)
nb = train.iloc[idx[0]][["Date", "Product_ID", "Quantity", "large_order", "channel_region_avg_qty"]].assign(distance=dist[0].round(2))
print(f"\nNew order: {order.Product_ID.iloc[0]}, real quantity {order.Quantity.iloc[0]}. Its 10 closest past orders:")
print(nb.to_string(index=False))
print(f"-> {nb.large_order.mean():.0%} of these 10 were large (the model uses all {best_k} neighbours: P(large) = {prob[0]:.2f})")

# %% Step 6. KNN for numbers (problem A): average the neighbours' units
sm = split_years(load_monthly())
best = None
for k in (3, 5, 10, 20, 40):
    m = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance"))
    m.fit(sm["fit"][MONTH_FEATS], sm["fit"]["units"])
    mae = mean_absolute_error(sm["val"].units, m.predict(sm["val"][MONTH_FEATS]))
    print(f"KNN regressor k={k:<2}: MAE on 2024 = {mae:.1f}")
    if best is None or mae < best[1]:
        best = (k, mae)
knr = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=best[0], weights="distance"))
knr.fit(sm["train"][MONTH_FEATS], sm["train"]["units"])
reg_results = {
    "Baseline (= last month)": regression_scores(sm["test"].units, sm["test"].units_lag_1),
    f"KNN Regressor (k={best[0]})": regression_scores(sm["test"].units, knr.predict(sm["test"][MONTH_FEATS])),
}
show(reg_results, "Problem A - monthly units, test 2025-2026 (lower MAE = better):")
# weights="distance": closer neighbours count more than far ones.

# %% Step 7. Save the final scores
save_result("KNN", "B_classification_large_order", results[f"KNN, threshold {thr:.2f}"])
save_result("KNN", "A_regression_monthly_units", reg_results[f"KNN Regressor (k={best[0]})"])
