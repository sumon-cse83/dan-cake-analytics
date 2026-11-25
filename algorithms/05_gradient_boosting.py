# %% [markdown]
# # Algorithm 5 – Gradient Boosting
# QUESTION (problem A): how many units will each product sell next month?   (Step 6 also does problem B)
# IDEA: build small trees ONE AFTER ANOTHER. Each new tree learns the ERRORS (residuals) the previous trees
#       still make, and a small part of it (the learning_rate) is added to the prediction:
#           prediction = average + lr*tree1 + lr*tree2 + lr*tree3 + ...
#       Random Forest = many trees side by side (averaged). Boosting = trees in a chain (each fixes the last).
# Good:  often the most accurate on tables of data.
# Weak:  more settings to tune; too many trees with a high learning rate -> overfitting.
# Run:   python algorithms/05_gradient_boosting.py

# %% Step 1. Load the data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from common import (MONTH_FEATS, ORDER_FEATS, SEED, classification_scores, load_monthly, load_orders,
                    regression_scores, save_fact, save_result, savefig, show, split_years)

s = split_years(load_monthly())
train, test = s["train"], s["test"]
X, y = train[MONTH_FEATS], train["units"]
print("rows: train", len(train), "| test", len(test))

# %% Step 2. Boosting BY HAND - see the idea work in 6 lines
lr = 0.5
pred = np.full(len(y), y.mean())                        # start: predict the average for everyone
print(f"\nstart (average only)  training MAE = {mean_absolute_error(y, pred):.1f}")
for i in range(1, 6):
    residual = y - pred                                 # what is still wrong
    tree = DecisionTreeRegressor(max_depth=3).fit(X, residual)   # small tree learns the errors
    pred = pred + lr * tree.predict(X)                  # add a part of the correction
    print(f"after tree {i}            training MAE = {mean_absolute_error(y, pred):.1f}")
# The error falls with every tree. That is all gradient boosting is; sklearn does it faster and with more care.

# %% Step 3. Learning rate x number of trees (tuned on 2024)
# staged_predict gives the prediction after 1 tree, 2 trees, ... so one fit shows the whole curve.
fig, ax = plt.subplots(figsize=(4.7, 2.6))
best = {"mae": np.inf}
for rate in (0.3, 0.1, 0.03):
    gb = GradientBoostingRegressor(n_estimators=800, learning_rate=rate, max_depth=3, random_state=SEED)
    gb.fit(s["fit"][MONTH_FEATS], s["fit"]["units"])
    curve = [mean_absolute_error(s["val"].units, p) for p in gb.staged_predict(s["val"][MONTH_FEATS])]
    ax.plot(range(1, len(curve) + 1), curve, label=f"learning_rate {rate}")
    n = int(np.argmin(curve)) + 1
    print(f"learning_rate {rate:<4}: best MAE on 2024 = {min(curve):.1f} with {n} trees")
    if min(curve) < best["mae"]:
        best = {"mae": min(curve), "rate": rate, "n": n}
ax.set_ylim(top=min(250, ax.get_ylim()[1])); ax.set_xlabel("number of trees"); ax.set_ylabel("MAE on 2024")
ax.legend(); ax.set_title("High rate learns fast then overfits; low rate needs more trees")
savefig(fig, "05_learning_rate.png")
print("chosen:", best)

# %% Step 4. Final model on 2020-2024 with the chosen settings, test on 2025-2026
gb = GradientBoostingRegressor(n_estimators=best["n"], learning_rate=best["rate"], max_depth=3, random_state=SEED)
gb.fit(X, y)
results = {
    "Baseline (= last month)": regression_scores(test.units, test.units_lag_1),
    "Gradient Boosting": regression_scores(test.units, gb.predict(test[MONTH_FEATS])),
}
show(results, "Test results 2025-2026 (lower MAE = better):")
imp = pd.Series(gb.feature_importances_, index=MONTH_FEATS).sort_values(ascending=False)
print("\nFeature importance:\n" + imp.round(3).to_string())

# %% Step 5. The same algorithm for problem B (large order yes/no)
# For yes/no, the trees correct the LOG-ODDS (the score inside the sigmoid from lesson 2).
so = split_years(load_orders())
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED)
gbc.fit(so["train"][ORDER_FEATS], so["train"]["large_order"])
prob = gbc.predict_proba(so["test"][ORDER_FEATS])[:, 1]
# No class_weight option here, so with threshold 0.5 it rarely says "large". Use a lower threshold:
# the share of large orders in training (~0.18) is a simple, fair choice.
thr = so["train"]["large_order"].mean()
clf_results = {
    "GB Classifier, threshold 0.5": classification_scores(so["test"].large_order, (prob >= 0.5).astype(int), prob),
    f"GB Classifier, threshold {thr:.2f}": classification_scores(so["test"].large_order, (prob >= thr).astype(int), prob),
}
show(clf_results, "Problem B - large order, test 2025-2026 (higher = better):")

# %% Step 6. Save the final scores (and key numbers for the report)
manual = np.full(len(y), y.mean())
save_fact(gb_manual_start=mean_absolute_error(y, manual), gb_manual_after5=mean_absolute_error(y, pred),
          gb_rate=best["rate"], gb_trees=best["n"], gb_val_mae=best["mae"], gb_top_feature=imp.index[0])
save_result("Gradient Boosting", "A_regression_monthly_units", results["Gradient Boosting"])
save_result("Gradient Boosting", "B_classification_large_order", clf_results[f"GB Classifier, threshold {thr:.2f}"])
