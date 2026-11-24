# Step 5 – Data encoding (hands-on): turning text categories into numbers
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("Dan_Cake_Bangladesh_500000_Transactions_2020_2026.csv")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# 1. Load and list the text (categorical) columns
df = pd.read_csv(DATA)
cat_cols = ["Region", "Channel", "Customer_Type", "Category", "Product_ID", "Product_Name", "Pack_Size", "Month_Name", "Quarter"]
print("rows, columns:", df.shape)
print("\nDistinct values per categorical column:\n", df[cat_cols].nunique())
# Models only understand numbers. A wrong choice invents an order that does not exist
# (e.g. Dhaka=1, Khulna=2 would mean Khulna is 'twice' Dhaka).

# 2. Split by time first. Every encoder below learns ONLY from the training years (2020-2024)
train = df[df.Year <= 2024].copy()
new = df[df.Year >= 2025].copy()                  # 2025-2026: treated like future data
print(f"\ntraining rows {len(train):,} | new rows {len(new):,}")

# 3. Method A - Ordinal (label) encoding: ONLY for categories that have a natural order
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
quarter_order = ["Q1", "Q2", "Q3", "Q4"]
for d in (train, new):
    d["Month_No"] = d["Month_Name"].map({m: i + 1 for i, m in enumerate(month_order)})
    d["Quarter_No"] = d["Quarter"].map({q: i + 1 for i, q in enumerate(quarter_order)})
print("\nA. Ordinal:", train[["Month_Name", "Month_No", "Quarter", "Quarter_No"]].drop_duplicates().head(4).to_string(index=False))

# 4. Method B - Cyclical encoding for months (December is next to January, but 12 and 1 are far apart as numbers)
for d in (train, new):
    d["Month_sin"] = np.sin(2 * np.pi * d["Month_No"] / 12)
    d["Month_cos"] = np.cos(2 * np.pi * d["Month_No"] / 12)
print("B. Cyclical: Dec and Jan are now neighbours ->",
      train.drop_duplicates("Month_No").set_index("Month_Name").loc[["Dec", "Jan"], ["Month_sin", "Month_cos"]].round(2).to_dict("index"))

# 5. Method C - One-hot encoding: one 0/1 column per value. For few, unordered categories.
low_card = ["Region", "Channel", "Customer_Type", "Category"]
oh_train = pd.get_dummies(train[low_card], dtype=np.uint8)
oh_new = pd.get_dummies(new[low_card], dtype=np.uint8).reindex(columns=oh_train.columns, fill_value=0)   # SAME columns as training
print(f"\nC. One-hot: {len(low_card)} columns became {oh_train.shape[1]} columns")
print(pd.concat([train[["Channel"]].head(3), oh_train.filter(like="Channel_").head(3)], axis=1).to_string(index=False))
# Linear / logistic regression: use pd.get_dummies(..., drop_first=True) to avoid the 'dummy variable trap'.
# A value never seen in training gets all zeros thanks to reindex() - the model does not crash.

# 6. Method D - Frequency encoding: replace each value by how common it is. Good for many categories (Product_ID).
freq = train["Product_ID"].value_counts(normalize=True)
train["Product_freq"] = train["Product_ID"].map(freq)
new["Product_freq"] = new["Product_ID"].map(freq).fillna(0)
print("\nD. Frequency: Product_ID (52 values) -> 1 column, e.g.\n", train[["Product_ID", "Product_freq"]].drop_duplicates().head(3).round(4).to_string(index=False))


# 7. Method E - Target (mean) encoding: replace each value by the average of the target (Quantity) for that value.
def target_map(d, col, target, m=50):
    """Average target per category, 'shrunk' towards the overall average when a category has few rows (m controls it)."""
    s = d.groupby(col)[target].agg(["mean", "count"])
    gm = d[target].mean()
    return (s["count"] * s["mean"] + m * gm) / (s["count"] + m), gm


def target_encode(train_df, new_df, col, target="Quantity", folds=5, seed=42):
    enc_new_map, gm = target_map(train_df, col, target)
    enc_new = new_df[col].map(enc_new_map).fillna(gm)
    # Training rows must NOT see their own answer: encode each row using the OTHER folds only (out-of-fold)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(train_df))
    enc_train = pd.Series(np.nan, index=train_df.index)
    for f in range(folds):
        mp, g = target_map(train_df[fold != f], col, target)
        enc_train[fold == f] = train_df.loc[fold == f, col].map(mp).fillna(g).values
    return enc_train, enc_new, enc_new_map


for col in ["Channel", "Region", "Product_ID"]:
    tr_enc, new_enc, mp = target_encode(train, new, col)
    train[f"{col}_te"], new[f"{col}_te"] = tr_enc, new_enc
    print(f"E. Target encoding {col:11s}: correlation with Quantity on NEW data = {new[f'{col}_te'].corr(new['Quantity']):.3f}")
print("   Channel and Region carry real signal; Product_ID carries almost none (matches the EDA finding).")

# 8. Check: no text left, no gaps, same columns for training and new data
feature_cols = ["Year", "Month_No", "Quarter_No", "Month_sin", "Month_cos", "Product_freq", "Channel_te", "Region_te", "Product_ID_te"]
X_train = pd.concat([train[feature_cols], oh_train], axis=1)
X_new = pd.concat([new[feature_cols], oh_new], axis=1)
assert list(X_train.columns) == list(X_new.columns)
print("\nEncoded features:", X_train.shape[1], "| missing values:", int(X_train.isna().sum().sum()) + int(X_new.isna().sum().sum()),
      "| non-numeric columns:", int((~X_train.dtypes.apply(pd.api.types.is_numeric_dtype)).sum()))

# 9. Which encoding for which algorithm
guide = pd.DataFrame({
    "algorithm": ["Linear / Logistic Regression, SVM, KNN, K-Means", "Decision Tree, Random Forest, Gradient Boosting, XGBoost"],
    "good choices": ["one-hot (drop_first for linear), frequency, target (few levels), cyclical month",
                     "ordinal, frequency, target, one-hot (trees do not need scaling or drop_first)"]})
print("\n", guide.to_string(index=False))

# 10. Save the encoded data and the encoders, so future data can be encoded the same way
out_train = pd.concat([train[["Transaction_ID", "Quantity"]], X_train], axis=1)
out_new = pd.concat([new[["Transaction_ID", "Quantity"]], X_new], axis=1)
out_train.to_csv("encoded_train_2020_2024.csv", index=False)
out_new.to_csv("encoded_new_2025_2026.csv", index=False)
json.dump({"month_order": month_order, "quarter_order": quarter_order, "one_hot_columns": list(oh_train.columns),
           "product_frequency": freq.round(6).to_dict(),
           "target_means": {c: {str(k): round(float(v), 4) for k, v in target_map(train, c, "Quantity")[0].items()} for c in ["Channel", "Region", "Product_ID"]}},
          open("encoding_maps.json", "w"), indent=2)
print("Saved: encoded_train_2020_2024.csv, encoded_new_2025_2026.csv, encoding_maps.json")