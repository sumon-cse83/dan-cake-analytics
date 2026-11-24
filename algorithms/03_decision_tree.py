# %% [markdown]
# # Algorithm 3 – Decision Tree
# QUESTION (problem B): will an order be LARGE (10+ units)?   (Step 7 also uses it for problem A)
# IDEA: ask yes/no questions one after another, like a flow chart:
#       "is channel_region_avg_qty <= 9.5?" -> yes -> "is Discount_% <= 3?" -> ... -> leaf = answer.
#       At each step the tree picks the question that best SEPARATES large from small orders
#       (measured by "Gini impurity": 0 = a group with only one class, 0.5 = a 50/50 mix).
# Good:  easy to read and explain, no scaling needed, finds non-linear rules.
# Weak:  a deep tree memorises the training data (OVERFITTING) -> limit max_depth.
# Run:   python algorithms/03_decision_tree.py

# %% Step 1. Load the data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

from common import (MONTH_FEATS, ORDER_FEATS, SEED, classification_scores, load_monthly, load_orders,
                    regression_scores, save_result, savefig, show, split_years)

s = split_years(load_orders())
train, test = s["train"], s["test"]
print("rows: train", len(train), "| test", len(test))

# %% Step 2. A tiny tree (depth 2) you can read completely
small = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=SEED)
small.fit(train[ORDER_FEATS], train["large_order"])
print("\nThe rules the tree learned (class 1 = large order):")
print(export_text(small, feature_names=ORDER_FEATS, show_weights=True))
# Read it top-down. "weights" = how many small (left) and large (right) orders end in that leaf (balanced weights).

# %% Step 3. How deep should the tree be? Tune on 2024, watch for overfitting
depths = range(1, 16)
fit_auc, val_auc = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, class_weight="balanced", random_state=SEED)
    t.fit(s["fit"][ORDER_FEATS], s["fit"]["large_order"])
    fit_auc.append(roc_auc_score(s["fit"]["large_order"], t.predict_proba(s["fit"][ORDER_FEATS])[:, 1]))
    val_auc.append(roc_auc_score(s["val"]["large_order"], t.predict_proba(s["val"][ORDER_FEATS])[:, 1]))
best_depth = list(depths)[int(np.argmax(val_auc))]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(depths, fit_auc, "o-", label="data it learned from (2020-23)")
ax.plot(depths, val_auc, "o-", label="unseen data (2024)")
ax.axvline(best_depth, ls="--", c="grey")
ax.set_xlabel("max_depth"); ax.set_ylabel("ROC-AUC"); ax.legend(); ax.set_title("Overfitting: training score keeps rising, unseen score falls")
savefig(fig, "03_depth_tuning.png")
print("\nbest max_depth on 2024:", best_depth)

# %% Step 4. Final tree: learn on 2020-2024 with the best depth, test on 2025-2026
tree = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=20, class_weight="balanced", random_state=SEED)
tree.fit(train[ORDER_FEATS], train["large_order"])
prob = tree.predict_proba(test[ORDER_FEATS])[:, 1]
results = {
    "Baseline (always 'small')": classification_scores(test.large_order, np.zeros(len(test), int), np.zeros(len(test))),
    f"Decision Tree (depth {best_depth})": classification_scores(test.large_order, tree.predict(test[ORDER_FEATS]), prob),
}
show(results, "Test results 2025-2026 (higher = better):")

# %% Step 5. Draw the top of the tree
fig, ax = plt.subplots(figsize=(18, 7))
plot_tree(tree, max_depth=2, feature_names=ORDER_FEATS, class_names=["small", "large"],
          filled=True, rounded=True, fontsize=8, ax=ax)
savefig(fig, "03_tree.png")
# Colour: orange = mostly small, blue = mostly large. Darker = purer group.

# %% Step 6. Which inputs did the tree use most?
imp = pd.Series(tree.feature_importances_, index=ORDER_FEATS).sort_values(ascending=False)
print("\nFeature importance (share of total impurity reduction, sums to 1):")
print(imp.head(8).round(3).to_string())

# %% Step 7. The same algorithm for problem A (a NUMBER: monthly units)
# For numbers the leaf gives the AVERAGE of its training rows, and splits reduce the squared error.
sm = split_years(load_monthly())
reg = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=SEED)
reg.fit(sm["train"][MONTH_FEATS], sm["train"]["units"])
reg_results = {
    "Baseline (= last month)": regression_scores(sm["test"].units, sm["test"].units_lag_1),
    "Decision Tree Regressor": regression_scores(sm["test"].units, reg.predict(sm["test"][MONTH_FEATS])),
}
show(reg_results, "Problem A - monthly units, test 2025-2026 (lower MAE = better):")

# %% Step 8. Save the final scores
save_result("Decision Tree", "B_classification_large_order", results[f"Decision Tree (depth {best_depth})"])
save_result("Decision Tree", "A_regression_monthly_units", reg_results["Decision Tree Regressor"])
