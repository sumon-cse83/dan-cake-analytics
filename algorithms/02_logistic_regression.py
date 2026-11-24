# %% [markdown]
# # Algorithm 2 – Logistic Regression
# QUESTION (problem B): will an order be LARGE (10 units or more)?  -> answer is yes (1) or no (0)
# IDEA: like linear regression, but the straight-line score z = b0 + b1*x1 + ... is squeezed by the
#       sigmoid curve  p = 1 / (1 + e^-z)  into a PROBABILITY between 0 and 1.
#       Then: p >= threshold (usually 0.5) -> predict "large".
# Despite the name it is a CLASSIFICATION algorithm.
# Good:  fast, gives probabilities, coefficients are explainable (odds ratios).
# Weak:  the boundary between classes is a straight line/plane.
# Run:   python algorithms/02_logistic_regression.py

# %% Step 1. Load the data and look at the target
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ORDER_FEATS, classification_scores, load_orders, save_result, savefig, show, split_years

orders = load_orders()
s = split_years(orders)
train, test = s["train"], s["test"]
print("rows: train", len(train), "| test", len(test))
print("share of large orders: train", round(train.large_order.mean(), 3), "| test", round(test.large_order.mean(), 3))
# Only about 1 in 5 orders is large -> the classes are IMBALANCED. Keep this in mind for Step 5.

# %% Step 2. The sigmoid curve: how a score becomes a probability
z = np.linspace(-6, 6, 100)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(z, 1 / (1 + np.exp(-z)))
ax.axhline(0.5, ls="--", c="grey"); ax.set_xlabel("z = b0 + b1*x1 + ..."); ax.set_ylabel("probability of 'large'")
ax.set_title("Sigmoid: any score -> a value between 0 and 1")
savefig(fig, "02_sigmoid.png")

# %% Step 3. One input: the average order size of the channel x region
one = LogisticRegression().fit(train[["channel_region_avg_qty"]], train["large_order"])
print(f"\nb0 = {one.intercept_[0]:.3f}, b1 = {one.coef_[0][0]:.3f}")
for q in (5, 10, 15):
    p = one.predict_proba(pd.DataFrame({"channel_region_avg_qty": [q]}))[0, 1]   # column 1 = P(large)
    print(f"  channel x region average {q:>2} units -> P(large order) = {p:.2f}")
# b1 > 0: the bigger the usual order size in that channel x region, the higher the chance of a large order.

# %% Step 4. All inputs (scaled) - and what the coefficients mean
# class_weight="balanced" makes a mistake on the RARE class (large) count more, so the model does not
# simply answer "small" every time.
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
logit.fit(train[ORDER_FEATS], train["large_order"])
coef = pd.Series(logit[-1].coef_[0], index=ORDER_FEATS)
odds = pd.DataFrame({"coefficient": coef, "odds_ratio": np.exp(coef)}).sort_values("coefficient", key=abs, ascending=False)
print("\nTop inputs. odds_ratio = how the odds of 'large' are multiplied for +1 standard deviation of the input:")
print(odds.head(8).round(3).to_string())
# Example: odds_ratio 2.0 -> doubles the odds; 0.5 -> halves them; 1.0 -> no effect.

# %% Step 5. Test: probabilities -> 0/1 with threshold 0.5, compare with a baseline
prob = logit.predict_proba(test[ORDER_FEATS])[:, 1]
pred = (prob >= 0.5).astype(int)
results = {
    "Baseline (always 'small')": classification_scores(test.large_order, np.zeros(len(test), int), np.zeros(len(test))),
    "Logistic, threshold 0.5": classification_scores(test.large_order, pred, prob),
}
# Baseline accuracy looks high (~77%) but it never finds a single large order (recall 0) -> accuracy alone misleads.

# %% Step 6. Choose a better threshold - on 2024 (validation), NOT on the test
tune = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
tune.fit(s["fit"][ORDER_FEATS], s["fit"]["large_order"])            # learn on 2020-2023 only
pv = tune.predict_proba(s["val"][ORDER_FEATS])[:, 1]
thresholds = np.arange(0.30, 0.81, 0.05)
f1s = [f1_score(s["val"].large_order, (pv >= t).astype(int)) for t in thresholds]
best_t = float(thresholds[int(np.argmax(f1s))])
print("\nF1 on 2024 for each threshold:")
print(pd.Series(f1s, index=thresholds.round(2), name="F1").round(3).to_string())
print("best threshold:", round(best_t, 2), "(if it is 0.50, the default was already the best choice)")
results["Logistic, tuned threshold"] = classification_scores(test.large_order, (prob >= best_t).astype(int), prob)
show(results, "Test results 2025-2026 (higher = better):")
# Lower threshold -> catch more large orders (recall up) but more false alarms (precision down). It is a business choice.

# %% Step 7. Pictures: confusion matrix and ROC curve
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_predictions(test.large_order, (prob >= best_t).astype(int),
                                        display_labels=["small", "large"], ax=ax[0], colorbar=False)
ax[0].set_title(f"Confusion matrix (threshold {best_t:.2f})")
RocCurveDisplay.from_predictions(test.large_order, prob, ax=ax[1], name="Logistic Regression")
ax[1].plot([0, 1], [0, 1], "k--", label="guessing"); ax[1].legend(); ax[1].set_title("ROC curve")
savefig(fig, "02_confusion_roc.png")
# Confusion matrix rows = truth, columns = prediction. Diagonal = correct answers.

# %% Step 8. Save the final score
save_result("Logistic Regression", "B_classification_large_order", results["Logistic, tuned threshold"])
