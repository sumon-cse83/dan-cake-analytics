# %% [markdown]
# # Algorithm 1 – Linear Regression
# QUESTION (problem A): how many units will each product sell next month?
# IDEA: draw the best straight line through the data.
#     one input   : units = b0 + b1 * x
#     many inputs : units = b0 + b1*x1 + b2*x2 + ... + bn*xn
# "Best" line = the one with the smallest total of squared errors (called Ordinary Least Squares, OLS).
# Good:  fast, easy to explain (each b is "how much units change when x goes up by 1").
# Weak:  only straight-line effects; one strange value can pull the line.
# Run:   python algorithms/01_linear_regression.py

# %% Step 1. Load the data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import MONTH_FEATS, load_monthly, regression_scores, save_result, savefig, show, split_years

pm = load_monthly()
s = split_years(pm)
train, test = s["train"], s["test"]
print("rows: train", len(train), "| test", len(test))
print(train[["Product_ID", "year", "month", "units", "units_lag_1", "units_lag_12"]].head())

# %% Step 2. The simplest model: ONE input (last month's units)
lr1 = LinearRegression()
lr1.fit(train[["units_lag_1"]], train["units"])          # .fit() = find the best b0 and b1
b0, b1 = lr1.intercept_, lr1.coef_[0]
print(f"\nLearned line:  units = {b0:.1f} + {b1:.3f} x last_month_units")
print(f"Meaning: a product that sold 1000 units last month is predicted to sell {b0 + b1 * 1000:.0f} this month.")
# b1 below 1 means demand is pulled back toward the average (a very high month is usually followed by a lower one).

# %% Step 3. Look at the line
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(train["units_lag_1"], train["units"], s=8, alpha=0.4, label="real product-months")
x = np.linspace(train["units_lag_1"].min(), train["units_lag_1"].max(), 50)
ax.plot(x, b0 + b1 * x, c="red", lw=2, label="learned line")
ax.set_xlabel("units last month"); ax.set_ylabel("units this month"); ax.legend()
ax.set_title("Linear regression with one input")
savefig(fig, "01_simple_line.png")

# %% Step 4. Use ALL inputs (multiple linear regression)
# Scale first, so every coefficient means "effect of +1 standard deviation" and they can be compared.
lr = make_pipeline(StandardScaler(), LinearRegression())
lr.fit(train[MONTH_FEATS], train["units"])
coef = pd.Series(lr[-1].coef_, index=MONTH_FEATS).sort_values(key=abs, ascending=False)
print("\nCoefficients (units change for +1 standard deviation of each input):")
print(coef.round(1).to_string())
# Positive = more of this input -> more units. Biggest absolute number = most influential input.
# Careful: lag_1, lag_12 and roll3 are related to each other, so their individual signs can be unstable.

# %% Step 5. Test on 2025-2026 and compare with a baseline
# Baseline = "next month will be the same as last month". A model must beat this to be worth using.
results = {
    "Baseline (= last month)": regression_scores(test["units"], test["units_lag_1"]),
    "Linear, 1 input": regression_scores(test["units"], lr1.predict(test[["units_lag_1"]])),
    "Linear, all inputs": regression_scores(test["units"], lr.predict(test[MONTH_FEATS])),
}
show(results, "Test results 2025-2026 (lower MAE/RMSE/MAPE = better, higher R2 = better):")

# %% Step 6. Check the errors (residuals = real - predicted)
pred = lr.predict(test[MONTH_FEATS])
resid = test["units"] - pred
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(pred, test["units"], s=10, alpha=0.5)
lim = [test["units"].min(), test["units"].max()]
ax[0].plot(lim, lim, "r--", label="perfect prediction")
ax[0].set_xlabel("predicted units"); ax[0].set_ylabel("real units"); ax[0].legend(); ax[0].set_title("Predicted vs real")
ax[1].hist(resid, bins=40)
ax[1].axvline(0, c="r"); ax[1].set_title(f"Errors: mean {resid.mean():.1f}, should be close to 0")
savefig(fig, "01_residuals.png")
# Good signs: points close to the red line, errors centred on 0 with no long tail on one side.

# %% Step 7. Use the model: predict one product's latest month
row = test.sort_values("month_index").iloc[[-1]]
print(f"\nProduct {row.Product_ID.iloc[0]}, month {int(row.month.iloc[0])}/{int(row.year.iloc[0])}: "
      f"predicted {lr.predict(row[MONTH_FEATS])[0]:.0f} units, real {row.units.iloc[0]:.0f} units")

# %% Step 8. Save the final score
save_result("Linear Regression", "A_regression_monthly_units", results["Linear, all inputs"])
