# Step 6 – Feature engineering (hands-on): building new columns that help a model
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("Dan_Cake_Bangladesh_500000_Transactions_2020_2026.csv")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# 1. Load
df = pd.read_csv(DATA)
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%y")
print("rows, columns:", df.shape)
# RULE: a feature may only use information known WHEN THE ORDER IS PLACED. Anything built from the
# future (or from the order's own result, e.g. Net_Sales) is leakage and makes a model look falsely good.

# 2. Group 1 - Calendar features from the date
df["day_of_week"] = df["Date"].dt.dayofweek            # 0 = Monday
df["day_of_month"] = df["Date"].dt.day
df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
df["is_month_end"] = (df["Date"].dt.is_month_end).astype(int)
df["is_december"] = (df["Month"] == 12).astype(int)     # the EDA found a strong December peak
df["month_index"] = (df["Year"] - df["Year"].min()) * 12 + df["Month"]       # 1 = Jan 2020 ... counts months in order
df["product_age_years"] = df["Year"] - df["Product_Launch_Year"]

# 3. Group 2 - Price features (price is known at order time)
df["log_price"] = np.log(df["Unit_Selling_Price_BDT"])
df["price_vs_list"] = df["Unit_Selling_Price_BDT"] / df["List_Price_BDT"]           # below 1 = sold under list price
df["has_discount_over_3pct"] = (df["Discount_%"] > 3).astype(int)

# 4. Group 3 - Pack size text -> numbers ('30g x 12', '27-30g', '12 pcs', '280g')
ps = df["Pack_Size"]
g = ps.str.extract(r"(\d+)(?:-(\d+))?\s*g")                                         # grams (a range gets its average)
df["pack_grams"] = np.where(g[1].notna(), (g[0].astype(float) + g[1].astype(float)) / 2, g[0].astype(float))
cnt = ps.str.extract(r"x\s*(\d+)")[0].fillna(ps.str.extract(r"(\d+)\s*pcs")[0]).astype(float)
df["pack_count"] = cnt.fillna(1)
df["price_per_100g"] = df["Unit_Selling_Price_BDT"] / (df["pack_grams"] * df["pack_count"]) * 100      # NaN for 'pcs' packs
print("\nGroup 3 example:\n", df[["Pack_Size", "pack_grams", "pack_count", "price_per_100g"]].drop_duplicates("Pack_Size").head(6).round(2).to_string(index=False))

# 5. Group 4 - History features (the most valuable ones for forecasting)
# Build a monthly table per product, then look BACKWARDS: last month, same month last year, last 3 months.
units = df.pivot_table(index="month_index", columns="Product_ID", values="Quantity", aggfunc="sum")     # rows = months
lag_1 = units.shift(1)                                    # shift(1): month t only sees month t-1
lag_12 = units.shift(12)                                  # same month last year
roll3 = units.shift(1).rolling(3).mean()                  # average of the previous 3 months (shift FIRST, then roll)
hist = pd.concat({"units_lag_1": lag_1, "units_lag_12": lag_12, "units_roll3_prev": roll3}, names=["feature"])
hist = hist.stack().unstack(0).reset_index()               # long table: month_index, Product_ID, 3 feature columns
df = df.merge(hist, on=["month_index", "Product_ID"], how="left")

# The classic mistake: include the current month in the rolling average
leaky = units.rolling(3).mean()
mon = units.stack().rename("units").reset_index()
mon["good_feature"] = roll3.stack().reindex(pd.MultiIndex.from_frame(mon[["month_index", "Product_ID"]])).values
mon["leaky_feature"] = leaky.stack().reindex(pd.MultiIndex.from_frame(mon[["month_index", "Product_ID"]])).values
print("\nGroup 4 - correlation with the month's real units (item-month table):")
print(f"  correct (previous 3 months only): {mon['units'].corr(mon['good_feature']):.3f}")
print(f"  WRONG   (includes the month itself): {mon['units'].corr(mon['leaky_feature']):.3f}   <- looks better, but uses the answer")
print("  The first 12 months have no 'last year' history; the rows are kept but those cells are empty (NaN).")

# 6. Split by time, then build features that must LEARN from data (training years only)
train = df[df.Year <= 2024].copy()
new = df[df.Year >= 2025].copy()
for d in (train, new):
    d["channel_region"] = d["Channel"] + "|" + d["Region"]
m = train.groupby("channel_region")["Quantity"].agg(["mean", "count"]); gm = train["Quantity"].mean()
cr_map = (m["count"] * m["mean"] + 50 * gm) / (m["count"] + 50)                      # shrunk average per channel x region
train["channel_region_avg_qty"] = train["channel_region"].map(cr_map)
new["channel_region_avg_qty"] = new["channel_region"].map(cr_map).fillna(gm)
print("\nGroup 5 - Interaction (channel x region): 44 combinations, one number each, learned from 2020-2024")
print(cr_map.sort_values(ascending=False).head(3).round(2).to_string(), "...")

# 7. Targets for later models
for d in (train, new):
    d["log_quantity"] = np.log1p(d["Quantity"])
    d["large_order"] = (d["Quantity"] >= 10).astype(int)

# 8. Which new features are useful? (Spearman rank correlation with Quantity, on the NEW data)
new_feats = ["day_of_week", "day_of_month", "week_of_year", "is_month_end", "is_december", "month_index", "product_age_years",
             "log_price", "price_vs_list", "has_discount_over_3pct", "pack_grams", "pack_count", "price_per_100g",
             "units_lag_1", "units_lag_12", "units_roll3_prev", "channel_region_avg_qty", "Discount_%"]
cor = new[new_feats + ["Quantity"]].corr(method="spearman")["Quantity"].drop("Quantity").sort_values(key=abs, ascending=False)
print("\nFeature usefulness - Spearman correlation with Quantity (new data):\n", cor.round(3).to_string())
# Order-level correlations are small (the EDA found order size is mostly random). The channel x region feature
# is the strongest. History features describe a PRODUCT'S monthly total, so they help forecast monthly demand,
# not the size of a single order.

# 9. Remove features that repeat each other (correlation above 0.95) or never change
X = train[new_feats].dropna()
cm = X.corr().abs().where(np.triu(np.ones((len(new_feats), len(new_feats))), 1).astype(bool))
drop = [c for c in cm.columns if (cm[c] > 0.95).any()]
const = [c for c in new_feats if train[c].nunique() <= 1]
print("\nDrop (near-duplicate of another feature):", drop, "| constant:", const)
keep = [c for c in new_feats if c not in drop and c not in const]
print("Kept features:", len(keep), "->", keep)

# 10. Save
base = ["Transaction_ID", "Date", "Product_ID", "Quantity", "log_quantity", "large_order"]
train[base + keep].to_csv("features_train_2020_2024.csv", index=False)
new[base + keep].to_csv("features_new_2025_2026.csv", index=False)
json.dump({"kept_features": keep, "dropped": drop, "channel_region_means": cr_map.round(4).to_dict()}, open("feature_list.json", "w"), indent=2)
print("Saved: features_train_2020_2024.csv, features_new_2025_2026.csv, feature_list.json")