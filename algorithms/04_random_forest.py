# %% [markdown]
# # Algorithm 4 – Random Forest
# QUESTION (problem A): how many units will each product sell next month?   (Step 7 also does problem B)
# IDEA: one decision tree is unstable (small data changes -> a different tree). So grow MANY trees, each:
#         - on a random "bootstrap" sample of the rows (drawn with replacement), and
#         - choosing each split from a random subset of the columns.
#       Then AVERAGE their answers (regression) or VOTE (classification). Many different, imperfect trees
#       together make fewer mistakes than one tree - "wisdom of the crowd".
# Good:  strong accuracy out of the box, little tuning, handles non-linear effects, no scaling needed.
# Weak:  slower and bigger than one tree, harder to explain (no single flow chart).
# Run:   python algorithms/04_random_forest.py

# %% Step 1. Load the data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from common import (MONTH_FEATS, ORDER_FEATS, SEED, classification_scores, load_monthly, load_orders,
                    regression_scores, save_result, savefig, show, split_years)

s = split_years(load_monthly())
train, test = s["train"], s["test"]
print("rows: train", len(train), "| test", len(test))

# %% Step 2. One tree vs. a forest - how many trees do we need? (tuned on 2024)
one_tree = DecisionTreeRegressor(random_state=SEED).fit(s["fit"][MONTH_FEATS], s["fit"]["units"])
print("\nOne full-depth tree, MAE on 2024:", round(mean_absolute_error(s["val"].units, one_tree.predict(s["val"][MONTH_FEATS])), 1))
sizes = [1, 5, 10, 25, 50, 100, 200, 400]
val_mae = []
for n in sizes:
    rf = RandomForestRegressor(n_estimators=n, min_samples_leaf=3, n_jobs=-1, random_state=SEED)
    rf.fit(s["fit"][MONTH_FEATS], s["fit"]["units"])
    val_mae.append(mean_absolute_error(s["val"].units, rf.predict(s["val"][MONTH_FEATS])))
print("Forest MAE on 2024 by number of trees:", dict(zip(sizes, np.round(val_mae, 1).tolist())))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(sizes, val_mae, "o-"); ax.set_xscale("log")
ax.set_xlabel("number of trees (n_estimators)"); ax.set_ylabel("MAE on 2024 (units)")
ax.set_title("More trees -> lower error, then it levels off")
savefig(fig, "04_n_trees.png")
# Adding trees never causes overfitting; it only costs time. Pick the point where the curve is flat.

# %% Step 3. Final forest on 2020-2024; the "out-of-bag" check
# Each tree never saw about 1/3 of the rows (not in its bootstrap sample). Scoring each row only with the trees
# that did NOT see it gives a free validation score: the OOB score (R2 here).
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, oob_score=True, n_jobs=-1, random_state=SEED)
rf.fit(train[MONTH_FEATS], train["units"])
print("\nOut-of-bag R2 (free internal check):", round(rf.oob_score_, 3))

# %% Step 4. Test on 2025-2026 vs. baseline and a single tree
results = {
    "Baseline (= last month)": regression_scores(test.units, test.units_lag_1),
    "One decision tree": regression_scores(test.units, one_tree.predict(test[MONTH_FEATS])),
    "Random Forest (300 trees)": regression_scores(test.units, rf.predict(test[MONTH_FEATS])),
}
show(results, "Test results 2025-2026 (lower MAE = better):")

# %% Step 5. Which inputs matter? Two ways
imp = pd.DataFrame({
    "impurity": rf.feature_importances_,      # how much each input reduced error inside the trees (fast, can favour many-valued columns)
    "permutation": permutation_importance(rf, test[MONTH_FEATS], test["units"], n_repeats=5,
                                          random_state=SEED, scoring="neg_mean_absolute_error").importances_mean,
}, index=MONTH_FEATS).sort_values("permutation", ascending=False)
# Permutation: shuffle ONE column and see how much MAE gets worse. Big increase = the model really needs it.
print("\nImportance (permutation = MAE increase in units when the column is shuffled):")
print(imp.round(3).to_string())

# %% Step 6. A bonus of forests: a range, not just one number
# Every tree gives its own prediction; their spread shows how sure the forest is.
row = test.iloc[[0]]
per_tree = np.array([t.predict(row[MONTH_FEATS].values)[0] for t in rf.estimators_])
print(f"\n{row.Product_ID.iloc[0]} {int(row.month.iloc[0])}/{int(row.year.iloc[0])}: forest says {per_tree.mean():.0f} units "
      f"(80% of trees between {np.percentile(per_tree, 10):.0f} and {np.percentile(per_tree, 90):.0f}); real {row.units.iloc[0]:.0f}")

# %% Step 7. The same algorithm for problem B (large order yes/no) - trees VOTE
so = split_years(load_orders())
rfc = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=SEED)
rfc.fit(so["train"][ORDER_FEATS], so["train"]["large_order"])
clf_results = {"Random Forest Classifier": classification_scores(
    so["test"].large_order, rfc.predict(so["test"][ORDER_FEATS]), rfc.predict_proba(so["test"][ORDER_FEATS])[:, 1])}
show(clf_results, "Problem B - large order, test 2025-2026 (higher = better):")

# %% Step 8. Save the final scores
save_result("Random Forest", "A_regression_monthly_units", results["Random Forest (300 trees)"])
save_result("Random Forest", "B_classification_large_order", clf_results["Random Forest Classifier"])
