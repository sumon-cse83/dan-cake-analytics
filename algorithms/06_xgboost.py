# %% [markdown]
# # Algorithm 6 – XGBoost (eXtreme Gradient Boosting)
# QUESTION (problem B): will an order be LARGE (10+ units)?   (Step 6 also does problem A)
# IDEA: the same chain-of-trees idea as lesson 5, in a faster library with extra protection against overfitting:
#         - regularisation (reg_lambda, reg_alpha): penalises trees with extreme leaf values
#         - subsample / colsample_bytree: each tree sees a random part of the rows / columns (like Random Forest)
#         - EARLY STOPPING: keep adding trees only while the validation score improves
#         - scale_pos_weight: built-in way to handle a rare class
# Good:  top accuracy on tabular data, fast, widely used in industry and competitions.
# Weak:  many settings; still a "black box" compared with one tree.
# Run:   python algorithms/06_xgboost.py

# %% Step 1. Load the data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from common import (MONTH_FEATS, ORDER_FEATS, SEED, classification_scores, load_monthly, load_orders,
                    regression_scores, save_fact, save_result, savefig, show, split_years)

s = split_years(load_orders())
train, test = s["train"], s["test"]
pos_weight = (train.large_order == 0).sum() / (train.large_order == 1).sum()
print("rows: train", len(train), "| test", len(test), f"| small orders per large order: {pos_weight:.2f}")
# scale_pos_weight = this ratio -> mistakes on large orders count ~4.5x, similar to class_weight="balanced".

# %% Step 2. Early stopping: let 2024 tell us how many trees to use
xgb = XGBClassifier(
    n_estimators=2000,            # an upper limit - early stopping will stop well before
    learning_rate=0.05,
    max_depth=4,                  # depth of each tree
    subsample=0.8,                # 80% of rows per tree
    colsample_bytree=0.8,         # 80% of columns per tree
    reg_lambda=1.0,               # L2 regularisation
    scale_pos_weight=pos_weight,
    eval_metric="auc",
    early_stopping_rounds=100,    # stop if AUC on the eval set has not improved for 100 trees
    random_state=SEED,
)
xgb.fit(s["fit"][ORDER_FEATS], s["fit"]["large_order"],
        eval_set=[(s["fit"][ORDER_FEATS], s["fit"]["large_order"]), (s["val"][ORDER_FEATS], s["val"]["large_order"])],
        verbose=False)
best_n = xgb.best_iteration + 1
print(f"\nbest number of trees on 2024: {best_n}  (AUC {xgb.best_score:.3f})")
hist = xgb.evals_result()
fig, ax = plt.subplots(figsize=(4.4, 2.5))
ax.plot(hist["validation_0"]["auc"], label="data it learned from (2020-23)")
ax.plot(hist["validation_1"]["auc"], label="unseen data (2024)")
ax.axvline(best_n - 1, ls="--", c="grey", label="early stopping point")
ax.set_xlabel("number of trees"); ax.set_ylabel("ROC-AUC"); ax.legend(); ax.set_title("XGBoost learning curve")
savefig(fig, "06_early_stopping.png")

# %% Step 3. Final model on 2020-2024 with that many trees, test on 2025-2026
final = XGBClassifier(n_estimators=best_n, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, scale_pos_weight=pos_weight, eval_metric="auc", random_state=SEED)
final.fit(train[ORDER_FEATS], train["large_order"])
prob = final.predict_proba(test[ORDER_FEATS])[:, 1]
results = {
    "Baseline (always 'small')": classification_scores(test.large_order, np.zeros(len(test), int), np.zeros(len(test))),
    "XGBoost": classification_scores(test.large_order, (prob >= 0.5).astype(int), prob),
}
show(results, "Test results 2025-2026 (higher = better):")

# %% Step 4. Feature importance by "gain" (how much each column improved the trees when used)
gain = pd.Series(final.get_booster().get_score(importance_type="gain")).reindex(ORDER_FEATS).fillna(0)
gain = (gain / gain.sum()).sort_values(ascending=False)
print("\nShare of total gain:\n" + gain.head(8).round(3).to_string())

# %% Step 5. The same library for problem A (monthly units)
sm = split_years(load_monthly())
xr = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                  eval_metric="mae", early_stopping_rounds=100, random_state=SEED)
xr.fit(sm["fit"][MONTH_FEATS], sm["fit"]["units"], eval_set=[(sm["val"][MONTH_FEATS], sm["val"]["units"])], verbose=False)
n_reg = xr.best_iteration + 1
xr = XGBRegressor(n_estimators=n_reg, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=SEED)
xr.fit(sm["train"][MONTH_FEATS], sm["train"]["units"])
reg_results = {
    "Baseline (= last month)": regression_scores(sm["test"].units, sm["test"].units_lag_1),
    f"XGBoost Regressor ({n_reg} trees)": regression_scores(sm["test"].units, xr.predict(sm["test"][MONTH_FEATS])),
}
show(reg_results, "Problem A - monthly units, test 2025-2026 (lower MAE = better):")

# %% Step 6. Save the final scores (and key numbers for the report)
save_fact(xgb_best_n=best_n, xgb_val_auc=xgb.best_score, xgb_pos_weight=pos_weight, xgb_reg_trees=n_reg,
          xgb_top_feature=gain.index[0], xgb_top_gain=gain.iloc[0])
save_result("XGBoost", "B_classification_large_order", results["XGBoost"])
save_result("XGBoost", "A_regression_monthly_units", reg_results[f"XGBoost Regressor ({n_reg} trees)"])
