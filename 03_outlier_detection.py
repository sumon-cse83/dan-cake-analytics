# Step 3 – Outlier detection (hands-on)
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                         # save charts as PNG files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("Dan_Cake_Bangladesh_500000_Transactions_2020_2026.csv")
SEED = 42
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# 1. Load
df = pd.read_csv(DATA)
df["Product_Year"] = df["Product_ID"] + "_" + df["Year"].astype(str)     # group key: same product, same year
print("rows, columns:", df.shape)

# 2. Look at the numbers first
num = ["Quantity", "Market_Price_BDT", "Discount_%", "Net_Sales_BDT"]
print(df[num].describe(percentiles=[.01, .5, .99]).round(2))
print("Skewness (0 = symmetric, >1 = long right tail):\n", df[num].skew().round(2))

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].boxplot(df["Quantity"]); ax[0].set_title("Quantity (raw)")
ax[1].boxplot(np.log1p(df["Quantity"])); ax[1].set_title("Quantity (log scale)")
ax[2].boxplot(df["Discount_%"]); ax[2].set_title("Discount %")
plt.tight_layout(); plt.savefig("outlier_boxplots.png", dpi=120); plt.close()
print("Chart saved: outlier_boxplots.png")


# 3. Four detection methods
def flag_iqr(s, k=1.5, log=False, groups=None):
    """Outside [Q1 - k*IQR, Q3 + k*IQR]. Optionally on a log scale and separately for each group."""
    x = np.log1p(s.clip(lower=0)) if log else s
    if groups is None:
        q1, q3 = x.quantile(.25), x.quantile(.75)
        q1, q3 = pd.Series(q1, index=x.index), pd.Series(q3, index=x.index)
    else:
        q = x.groupby(groups).quantile([.25, .75]).unstack()
        q1, q3 = groups.map(q[0.25]), groups.map(q[0.75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def flag_zscore(s, k=3):
    return ((s - s.mean()) / s.std()).abs() > k


def flag_mad(s, k=3.5):                     # "modified z-score": uses the median, so outliers cannot hide themselves
    med = s.median(); mad = (s - med).abs().median()
    return (0.6745 * (s - med) / mad).abs() > k


q = df["Quantity"]
methods = {
    "IQR on raw values (k=1.5)": flag_iqr(q),
    "Z-score |z| > 3": flag_zscore(q),
    "Modified z-score (MAD) > 3.5": flag_mad(q),
    "IQR on log scale, per product (k=1.5)": flag_iqr(q, 1.5, True, df["Product_ID"]),
}
table = pd.DataFrame({"flagged rows": {k: int(v.sum()) for k, v in methods.items()},
                      "% of rows": {k: round(v.mean() * 100, 2) for k, v in methods.items()}})
print("\nSame column, four methods, four answers:\n", table)
# Quantity is skewed and every product has its own normal size, so one global limit flags too many or too few.
# The log-scale, per-product IQR is the best fit for this data.

# 4. Domain rules: values that are IMPOSSIBLE (not just unusual)
RULES = {"Quantity": (1, 500), "Market_Price_BDT": (0.01, 1000), "Discount_%": (0, 50)}


def rule_violations(d):
    return {c: (d[c] < lo) | (d[c] > hi) for c, (lo, hi) in RULES.items()}


print("\nImpossible values in the real file:", {c: int(v.sum()) for c, v in rule_violations(df).items()})

# 5. Test copy: damage a copy on purpose, keep the original as the answer key
rng = np.random.default_rng(SEED)
truth = df.copy()
dirty = df.copy()
for c in ["Quantity", "Market_Price_BDT", "Discount_%"]:
    dirty[c] = dirty[c].astype(float)

order = rng.permutation(len(dirty))                       # non-overlapping random rows
g_qty_bad, g_price, g_disc, g_extreme = (order[i:i + n] for i, n in [(0, 100), (100, 180), (280, 160), (440, 400)])
dirty.loc[g_qty_bad, "Quantity"] = rng.choice([-3, 0, 9999], size=len(g_qty_bad))     # impossible
dirty.loc[g_price, "Market_Price_BDT"] *= 100                                          # price typed 100x too big
dirty.loc[g_disc, "Discount_%"] = 250                                                  # impossible discount
small = dirty.index[(truth.Quantity < 10)].intersection(pd.Index(g_extreme))
dirty.loc[small, "Quantity"] = truth.loc[small, "Quantity"] * 25                       # extreme but possible (bulk-like)
groups = {"impossible Quantity": g_qty_bad, "Market_Price x100": g_price, "Discount_% = 250": g_disc, "extreme Quantity x25": small.values}
print("\nTest copy created with", sum(len(v) for v in groups.values()), "damaged cells")

# 6. Detect: rules first, then statistics
viol = rule_violations(dirty)
stat = {
    "Quantity": flag_iqr(dirty["Quantity"].where(~viol["Quantity"]), 1.5, True, dirty["Product_ID"]),
    "Market_Price_BDT": flag_iqr(dirty["Market_Price_BDT"].where(~viol["Market_Price_BDT"]), 3.0, False, dirty["Product_Year"]),
    "Discount_%": flag_iqr(dirty["Discount_%"].where(~viol["Discount_%"]), 3.0, False, dirty["Product_ID"]),
}
col_of = {"impossible Quantity": "Quantity", "Market_Price x100": "Market_Price_BDT", "Discount_% = 250": "Discount_%", "extreme Quantity x25": "Quantity"}
rows = []
for name, idx in groups.items():
    c = col_of[name]
    by_rule, by_stat = viol[c].loc[idx], stat[c].loc[idx]
    rows.append((name, len(idx), round(by_rule.mean() * 100, 1), round(by_stat.mean() * 100, 1), round((by_rule | by_stat).mean() * 100, 1)))
print("\nDetection rate (% of damaged cells found):")
print(pd.DataFrame(rows, columns=["damage", "cells", "by rule %", "by statistics %", "either %"]).to_string(index=False))
untouched = ~dirty.index.isin(np.concatenate(list(groups.values())))
print(f"False alarms on untouched rows (Quantity statistics): {stat['Quantity'][untouched].mean() * 100:.2f}%  (real-file baseline 0.27%)")

# 7. Treat: different problems, different actions
clean = dirty.copy()
# 7a. Impossible values are errors -> blank them, then rebuild from the related columns (same idea as Step 1)
for c in RULES:
    clean.loc[viol[c], c] = np.nan
clean.loc[stat["Market_Price_BDT"], "Market_Price_BDT"] = np.nan          # a price 100x off is also an error
fix = clean.Gross_Sales_BDT / clean.Quantity
clean["Market_Price_BDT"] = clean["Market_Price_BDT"].fillna(fix.round(2))
clean["Quantity"] = clean["Quantity"].fillna((clean.Gross_Sales_BDT / clean.Market_Price_BDT).round())
clean["Discount_%"] = clean["Discount_%"].fillna((clean.Discount_BDT / clean.Gross_Sales_BDT * 100).round(2))
for c in ["Quantity", "Market_Price_BDT", "Discount_%"]:                 # anything still blank -> product median
    clean[c] = clean[c].fillna(clean.groupby("Product_ID")[c].transform("median"))

# 7b. Unusual-but-possible values (bulk orders) are NOT errors -> keep them, mark them
clean["flag_outlier_Quantity"] = flag_iqr(clean["Quantity"], 1.5, True, clean["Product_ID"])
# Optional version for models that dislike extremes: cap (winsorise) at the upper limit instead of deleting rows
lq = np.log1p(clean["Quantity"]); g = lq.groupby(clean["Product_ID"])
hi = np.expm1(g.transform(lambda v: v.quantile(.75)) + 1.5 * g.transform(lambda v: v.quantile(.75) - v.quantile(.25)))
clean["Quantity_capped"] = np.minimum(clean["Quantity"], hi)

# 8. Check
for name, idx in list(groups.items())[:3]:
    c = col_of[name]
    ok = ((clean.loc[idx, c] - truth.loc[idx, c]).abs() <= 0.011).mean() * 100
    print(f"{name:22s} repaired exactly: {ok:5.1f}%")
print("Extreme Quantity x25 rows flagged and kept:", f"{clean.loc[groups['extreme Quantity x25'], 'flag_outlier_Quantity'].mean() * 100:.1f}%")
print("Mean Quantity -> original:", round(truth.Quantity.mean(), 3), "| after treatment:", round(clean.Quantity.mean(), 3),
      "| capped:", round(clean.Quantity_capped.mean(), 3))
print("Rows kept:", len(clean), "of", len(dirty), "(outlier treatment deleted no rows)")

# 9. Save
clean.drop(columns="Product_Year").to_csv("cleaned_after_outlier_treatment.csv", index=False)
print("Saved: cleaned_after_outlier_treatment.csv")