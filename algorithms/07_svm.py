# %% [markdown]
# # Algorithm 7 – Support Vector Machine (SVM)
# QUESTION (problem B): will an order be LARGE (10+ units)?   (Step 6 also does problem A with SVR)
# IDEA: find the boundary that separates the two classes with the WIDEST possible gap (the "margin").
#       Only the points closest to the boundary - the "support vectors" - decide where it goes.
#       KERNEL TRICK: if no straight line separates the classes, the RBF kernel lets the boundary curve.
#       C = how strictly to avoid mistakes on the training data:
#           small C -> wide, smooth margin (some training mistakes allowed)
#           large C -> narrow, wiggly margin (risk of overfitting)
# Good:  strong on small/medium data with a clear margin; works in many dimensions.
# Weak:  SLOW on large data (time grows roughly with rows squared); needs scaling; no probabilities by default.
# Run:   python algorithms/07_svm.py      (the slowest lesson: about 1-2 minutes)

# %% Step 1. Load the data
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from common import (MONTH_FEATS, ORDER_FEATS, SEED, classification_scores, load_monthly, load_orders,
                    regression_scores, save_fact, save_result, savefig, show, split_years)

s = split_years(load_orders())
train, test = s["train"], s["test"]
# For TUNING use 4,000 rows from 2020-2023 (fast); the final model uses more.
fit_small = s["fit"].sample(4000, random_state=SEED)
val = s["val"]
print("rows: train", len(train), "| test", len(test), "| tuning sample", len(fit_small))

def val_auc(model):
    """Fit on the tuning sample, return ROC-AUC on 2024 and seconds taken."""
    t = time.time()
    model.fit(fit_small[ORDER_FEATS], fit_small["large_order"])
    # SVC has no predict_proba by default; decision_function = signed distance from the boundary (a score)
    return roc_auc_score(val["large_order"], model.decision_function(val[ORDER_FEATS])), time.time() - t

# %% Step 2. Why scaling is a MUST for SVM
# SVM measures distances. units_lag_1 is in the thousands, is_month_end is 0/1 -> without scaling,
# the big column decides everything.
raw = SVC(kernel="rbf", class_weight="balanced")
scaled = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
a_raw, t_raw = val_auc(raw)
a_sc, t_sc = val_auc(scaled)
print(f"\nwithout scaling: AUC {a_raw:.3f} ({t_raw:.1f}s)   |   with scaling: AUC {a_sc:.3f} ({t_sc:.1f}s)")

# %% Step 3. Kernel: straight boundary (linear) or curved (RBF)?
for kernel in ("linear", "rbf"):
    a, t = val_auc(make_pipeline(StandardScaler(), SVC(kernel=kernel, class_weight="balanced")))
    print(f"kernel {kernel:<6}: AUC on 2024 = {a:.3f}  ({t:.1f}s)")
    save_fact(**{f"svm_{kernel}_auc": a})

# %% Step 4. Tune C (and gamma = how far one point's influence reaches in the RBF kernel)
grid = []
for C in (0.1, 1, 10):
    for gamma in ("scale", 0.01):
        a, t = val_auc(make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")))
        grid.append({"C": C, "gamma": gamma, "AUC_2024": round(a, 3), "seconds": round(t, 1)})
grid = pd.DataFrame(grid).sort_values("AUC_2024", ascending=False)
print("\nTuning results (2024):\n" + grid.to_string(index=False))
best = grid.iloc[0]

fig, ax = plt.subplots(figsize=(4.4, 2.5))
for g, part in grid.groupby("gamma"):
    part = part.sort_values("C")
    ax.plot(part["C"], part["AUC_2024"], "o-", label=f"gamma={g}")
ax.set_xscale("log"); ax.set_xlabel("C"); ax.set_ylabel("ROC-AUC on 2024"); ax.legend(); ax.set_title("SVM tuning")
savefig(fig, "07_svm_tuning.png")

# %% Step 5. Final SVM: learn on 8,000 orders from 2020-2024, test on 2025-2026
final_train = train.sample(8000, random_state=SEED)
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=float(best["C"]),
                                          gamma=best["gamma"] if best["gamma"] == "scale" else float(best["gamma"]),
                                          class_weight="balanced"))
t = time.time()
svm.fit(final_train[ORDER_FEATS], final_train["large_order"])
score = svm.decision_function(test[ORDER_FEATS])
print(f"\nfinal fit + predict: {time.time() - t:.0f}s | support vectors used: {svm[-1].n_support_.sum()} of {len(final_train)} rows")
results = {
    "Baseline (always 'small')": classification_scores(test.large_order, np.zeros(len(test), int), np.zeros(len(test))),
    "SVM (RBF kernel)": classification_scores(test.large_order, svm.predict(test[ORDER_FEATS]), score),
}
show(results, "Test results 2025-2026 (higher = better):")

# %% Step 6. SVM for numbers: SVR (problem A)
# SVR fits a tube of width epsilon around the data and ignores errors inside the tube.
# The target (units, in hundreds) must also be scaled -> TransformedTargetRegressor scales it and undoes it.
sm = split_years(load_monthly())
svr = TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.05)),
                                 transformer=StandardScaler())
svr_lin = TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR(kernel="linear", C=1, epsilon=0.05)),
                                     transformer=StandardScaler())
svr.fit(sm["train"][MONTH_FEATS], sm["train"]["units"])
svr_lin.fit(sm["train"][MONTH_FEATS], sm["train"]["units"])
reg_results = {
    "Baseline (= last month)": regression_scores(sm["test"].units, sm["test"].units_lag_1),
    "SVR (RBF kernel)": regression_scores(sm["test"].units, svr.predict(sm["test"][MONTH_FEATS])),
    "SVR (linear kernel)": regression_scores(sm["test"].units, svr_lin.predict(sm["test"][MONTH_FEATS])),
}
show(reg_results, "Problem A - monthly units, test 2025-2026 (lower MAE = better):")
# If the linear kernel wins here, the relationship is mostly a straight line (see lesson 1).

# %% Step 7. Save the final scores (and key numbers for the report)
save_fact(svm_raw_auc=a_raw, svm_scaled_auc=a_sc, svm_best_C=float(best["C"]), svm_best_gamma=str(best["gamma"]),
          svm_best_val_auc=float(best["AUC_2024"]), svm_n_support=int(svm[-1].n_support_.sum()),
          svm_train_rows=len(final_train))
best_reg = min(["SVR (RBF kernel)", "SVR (linear kernel)"], key=lambda k: reg_results[k]["MAE"])
save_fact(svr_best=best_reg)
save_result("SVM", "B_classification_large_order", results["SVM (RBF kernel)"])
save_result("SVM", "A_regression_monthly_units", reg_results[best_reg])
