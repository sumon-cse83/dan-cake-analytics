# Step 4 – Data normalization / feature scaling (hands-on)
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                         # save charts as PNG files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("Dan_Cake_Bangladesh_500000_Transactions_2020_2026.csv")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# 1. Load and choose the columns to scale
# Scale only real measurements. Do NOT scale IDs, Year/Month, or columns calculated from others (Gross, Discount_BDT).
df = pd.read_csv(DATA)
cols = ["Quantity", "Market_Price_BDT", "Discount_%", "Unit_Selling_Price_BDT", "Net_Sales_BDT"]
print("rows, columns:", df.shape)

# 2. Why scale? The columns live on very different scales
print(df[cols].agg(["min", "max", "mean", "std", "skew"]).round(2))


# 3. Four scaling methods: fit() learns the numbers, transform() applies them, inverse() undoes them
def fit(x, method):
    if method == "minmax":
        return {"min": x.min(), "max": x.max()}
    if method == "zscore":
        return {"mean": x.mean(), "std": x.std()}
    if method == "robust":
        return {"median": x.median(), "iqr": x.quantile(.75) - x.quantile(.25)}
    if method == "logz":                                   # log first (fixes skew), then z-score
        lx = np.log1p(x)
        return {"mean": lx.mean(), "std": lx.std()}


def transform(x, method, p):
    if method == "minmax":
        return (x - p["min"]) / (p["max"] - p["min"])
    if method == "zscore":
        return (x - p["mean"]) / p["std"]
    if method == "robust":
        return (x - p["median"]) / p["iqr"]
    if method == "logz":
        return (np.log1p(x) - p["mean"]) / p["std"]


def inverse(z, method, p):
    if method == "minmax":
        return z * (p["max"] - p["min"]) + p["min"]
    if method == "zscore":
        return z * p["std"] + p["mean"]
    if method == "robust":
        return z * p["iqr"] + p["median"]
    if method == "logz":
        return np.expm1(z * p["std"] + p["mean"])


METHODS = ["minmax", "zscore", "robust", "logz"]
print("\nWhat each method does to Quantity and Net_Sales_BDT:")
rows = []
for c in ["Quantity", "Net_Sales_BDT"]:
    for m in METHODS:
        z = transform(df[c], m, fit(df[c], m))
        rows.append((c, m, round(z.min(), 2), round(z.max(), 2), round(z.mean(), 2), round(z.std(), 2), round(z.skew(), 2)))
print(pd.DataFrame(rows, columns=["column", "method", "min", "max", "mean", "std", "skew"]).to_string(index=False))
# Scaling changes the RANGE, not the SHAPE. Only the log step reduces the skew.

fig, ax = plt.subplots(2, 4, figsize=(15, 6))
for r, c in enumerate(["Quantity", "Net_Sales_BDT"]):
    ax[r, 0].hist(df[c], bins=60, color="#1F3864"); ax[r, 0].set_title(f"{c} original")
    for k, (m, t) in enumerate([("minmax", "min-max"), ("zscore", "z-score"), ("logz", "log + z-score")], start=1):
        ax[r, k].hist(transform(df[c], m, fit(df[c], m)), bins=60, color="#C65911"); ax[r, k].set_title(f"{c}: {t}")
plt.tight_layout(); plt.savefig("normalization_histograms.png", dpi=120); plt.close()
print("Chart saved: normalization_histograms.png")

# 4. The RIGHT way: learn the numbers from the TRAINING period only
train = df[df.Year <= 2024]
test = df[df.Year == 2026]
params = {m: {c: fit(train[c], m) for c in cols} for m in METHODS}          # learned from 2020-2024 only

print("\nMin-max with numbers learned from 2020-2024, applied to 2026 - share of values outside 0..1:")
out = {c: float(((transform(test[c], "minmax", params["minmax"][c]) < 0) | (transform(test[c], "minmax", params["minmax"][c]) > 1)).mean() * 100)
       for c in cols}
print(pd.Series(out).round(1).astype(str) + " %")
print("Z-score of 2026 under 2020-2024 numbers (mean per column; 0 would mean 'no change since training'):")
print(pd.Series({c: transform(test[c], "zscore", params["zscore"][c]).mean() for c in cols}).round(2))
# Prices have risen about 8% a year, so some 2026 prices sit ABOVE the training range. That is real information.
# If min/max were learned from ALL years, 2026 would look perfectly inside 0..1 and hide the drift
# (this mistake is called DATA LEAKAGE).
all_p = {c: fit(df[c], "minmax") for c in cols}
leak = {c: float(((transform(test[c], "minmax", all_p[c]) < 0) | (transform(test[c], "minmax", all_p[c]) > 1)).mean() * 100) for c in cols}
print("Wrong way (min/max learned from all years) - share outside 0..1:", {c: round(v, 1) for c, v in leak.items()})

# 5. Check that scaling is reversible and correct
for m in METHODS:
    back = inverse(transform(df["Quantity"], m, params[m]["Quantity"]), m, params[m]["Quantity"])
    assert np.allclose(back, df["Quantity"]), m
print("\nInverse check passed: scaled values convert back to the original numbers for all four methods.")
trz = transform(train["Quantity"], "zscore", params["zscore"]["Quantity"])
print(f"Training z-score of Quantity: mean {trz.mean():.3f}, std {trz.std():.3f}  (should be 0 and 1)")

# Optional: compare with scikit-learn (install with: pip install scikit-learn)
try:
    from sklearn.preprocessing import StandardScaler
    a = StandardScaler().fit(train[cols]).transform(test[cols])
    b = np.column_stack([transform(test[c], "zscore", params["zscore"][c]) for c in cols])
    print("scikit-learn StandardScaler gives the same numbers:", np.allclose(a, b, atol=1e-3))
except ImportError:
    print("scikit-learn not installed - optional check skipped")

# 6. Which method for which algorithm
guide = pd.DataFrame({
    "algorithm": ["Linear / Logistic Regression", "SVM", "KNN", "K-Means", "Decision Tree / Random Forest / Gradient Boosting / XGBoost"],
    "needs scaling?": ["yes (z-score)", "yes (z-score or min-max)", "yes (min-max or z-score)", "yes (z-score)", "no"]})
print("\n", guide.to_string(index=False))

# 7. Save: scaled columns + the numbers used, so new data can be scaled in exactly the same way later
out_df = df.copy()
for c in cols:
    out_df[f"{c}_minmax"] = transform(df[c], "minmax", params["minmax"][c])
    out_df[f"{c}_z"] = transform(df[c], "zscore", params["zscore"][c])
    out_df[f"{c}_robust"] = transform(df[c], "robust", params["robust"][c])
    if c in ("Quantity", "Net_Sales_BDT"):
        out_df[f"{c}_logz"] = transform(df[c], "logz", params["logz"][c])
out_df.to_csv("normalized_data.csv", index=False)
json.dump({m: {c: {k: float(v) for k, v in p.items()} for c, p in d.items()} for m, d in params.items()},
          open("scaling_parameters.json", "w"), indent=2)
print("Saved: normalized_data.csv and scaling_parameters.json (numbers learned from 2020-2024)")