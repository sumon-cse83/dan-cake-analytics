# %% [markdown]
# # Step 1 – Missing value treatment (hands-on)
# Dataset: Dan Cake Bangladesh transactions (500,000 rows).
# Put the CSV in the SAME folder as this file.

# %% 0. Imports and settings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("Dan_Cake_Bangladesh_500000_Transactions_2020_2026.csv")
SEED = 42                                    # makes the "random" practice gaps repeatable
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print("pandas", pd.__version__, "| numpy", np.__version__)

# %% 1. Load the data
df = pd.read_csv(DATA)
print("rows, columns:", df.shape)
df.info()                                    # column names, types, non-null counts
df.head()


# %% 2. Look for missing values in the REAL file
def missing_report(d: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing cells per column (only columns that have any)."""
    r = pd.DataFrame({"missing": d.isna().sum(), "percent": (d.isna().mean() * 100).round(2)})
    return r[r.missing > 0].sort_values("missing", ascending=False)


rep = missing_report(df)
print("Total missing cells in the real file:", int(df.isna().sum().sum()))
print(rep if len(rep) else "-> No standard missing values (NaN) found.")

# "Hidden" missing values: text such as 'NA', 'null', '-', '' that Python does NOT see as missing.
PLACEHOLDERS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing", "unknown"}
for col in df.columns[df.dtypes.astype(str).isin(["object", "str", "string"])]:
    n = df[col].astype(str).str.strip().str.lower().isin(PLACEHOLDERS).sum()
    if n:
        print(f"Hidden placeholder values in {col}: {n}")
print("Hidden-placeholder check finished.")

# %% 3. Create PRACTICE data: copy the file and punch random holes in it
# The real file is complete, so we damage a COPY on purpose and keep the original as the answer key.
truth = df.copy()
dirty = df.copy()
num_cols = ["Quantity", "Market_Price_BDT", "Discount_%", "Unit_Selling_Price_BDT", "Gross_Sales_BDT",
            "Discount_BDT", "Net_Sales_BDT", "Observed_Price_BDT", "List_Price_BDT"]
dirty[num_cols] = dirty[num_cols].astype(float)          # integer columns cannot hold NaN, so use float

rng = np.random.default_rng(SEED)
holes = {  # column -> share of rows to blank out
    "Quantity": 0.03, "Market_Price_BDT": 0.03, "Net_Sales_BDT": 0.03, "Discount_%": 0.02,
    "Gross_Sales_BDT": 0.02, "Product_Name": 0.02, "Category": 0.02, "Region": 0.03, "Channel": 0.02,
    "Customer_Type": 0.02, "Year": 0.01, "Month_Name": 0.01, "Date": 0.002,
}
injected = {}
for col, share in holes.items():
    idx = rng.choice(dirty.index.values, size=int(len(dirty) * share), replace=False)
    dirty.loc[idx, col] = np.nan
    injected[col] = idx
print("Practice file created. Missing cells now:", int(dirty.isna().sum().sum()))
print(missing_report(dirty))

# %% 4. Understand the missing values BEFORE treating them
rep = missing_report(dirty)
rep["percent"].sort_values().plot.barh(figsize=(7, 4), color="#1F3864", title="Percent missing per column")
plt.xlabel("% of rows"); plt.tight_layout(); plt.show()

chk = dirty.assign(qty_missing=dirty["Quantity"].isna()).groupby("Channel")["qty_missing"].mean().mul(100).round(2)
print("% of Quantity missing, by channel (should all be close to 3%):\n", chk)
print("Rows with 0, 1, 2+ missing cells:\n", dirty.isna().sum(axis=1).clip(upper=2).value_counts().sort_index())

# %% 5. DECISION GUIDE
#   1. Can I calculate it from other columns?      -> YES: derive it (exact, best option)
#   2. Is it an identifier / essential field?      -> YES: drop the row (a guess would be fabrication)
#   3. Is it numeric?          -> fill with the MEDIAN of similar rows (same product)
#   4. Is it a category?       -> fill with the most frequent value ('mode') or a visible label 'Unknown'
#   5. Is most of the column empty (e.g. >60-80%)?  -> consider dropping the column

# %% 6. TREAT the missing values
clean = dirty.copy()
log = []                                       # (step, column, rows changed)


def note(step, col, n):
    if n:
        log.append((step, col, int(n)))


# --- 6a. Essential fields: drop the rows
essential = ["Transaction_ID", "Date", "Product_ID"]
before = len(clean)
clean = clean.dropna(subset=essential)
note("drop row: essential field missing", "+".join(essential), before - len(clean))

# --- 6b. Derive from other columns (exact)
clean["Date_parsed"] = pd.to_datetime(clean["Date"], format="%d-%m-%y")
for col, val in [("Year", clean.Date_parsed.dt.year), ("Month", clean.Date_parsed.dt.month),
                 ("Month_Name", clean.Date_parsed.dt.strftime("%b")), ("Quarter", "Q" + clean.Date_parsed.dt.quarter.astype(str))]:
    m = clean[col].isna()
    clean.loc[m, col] = val[m]; note("derived from Date", col, m.sum())

for col in ["Product_Name", "Category", "Pack_Size", "Observed_Price_BDT", "Product_Launch_Year"]:
    mode_by_product = clean.groupby("Product_ID")[col].agg(lambda s: s.mode().iat[0])
    m = clean[col].isna()
    clean.loc[m, col] = clean.loc[m, "Product_ID"].map(mode_by_product); note("derived from Product_ID", col, m.sum())


def fill(col, values, decimals=2):
    m = clean[col].isna() & values.notna() & np.isfinite(values)
    clean.loc[m, col] = values[m].round(decimals); note("derived by formula", col, m.sum())


for _ in range(2):                              # two passes, because one fill can unlock the next
    c = clean
    fill("Gross_Sales_BDT", c.Quantity * c.Market_Price_BDT)
    fill("Quantity", c.Gross_Sales_BDT / c.Market_Price_BDT, 0)
    fill("Market_Price_BDT", c.Gross_Sales_BDT / c.Quantity)
    fill("Gross_Sales_BDT", c.Net_Sales_BDT + c.Discount_BDT)
    fill("Discount_BDT", c.Gross_Sales_BDT - c.Net_Sales_BDT)
    fill("Net_Sales_BDT", c.Gross_Sales_BDT - c.Discount_BDT)
    fill("Unit_Selling_Price_BDT", c.Net_Sales_BDT / c.Quantity)
    fill("Discount_%", c.Discount_BDT / c.Gross_Sales_BDT * 100)

# --- 6c. Statistical fill for numeric columns that are still empty
for col, groups in [("Quantity", ["Product_ID"]), ("Discount_%", ["Product_ID"]),
                    ("Market_Price_BDT", ["Product_ID", "Year"]), ("Unit_Selling_Price_BDT", ["Product_ID", "Year"]),
                    ("List_Price_BDT", ["Product_ID", "Year"])]:
    m = clean[col].isna()
    if m.any():
        grp_median = clean.groupby(groups)[col].transform("median")
        clean[col] = clean[col].fillna(grp_median).fillna(clean[col].median())
        note(f"median by {'+'.join(groups)}", col, m.sum())

# --- 6d. Categorical columns that cannot be derived
for col in ["Region", "Channel", "Customer_Type"]:
    m = clean[col].isna()
    clean.loc[m, col] = "Unknown"; note("label 'Unknown'", col, m.sum())

for _ in range(2):
    c = clean
    fill("Gross_Sales_BDT", c.Quantity * c.Market_Price_BDT)
    fill("Net_Sales_BDT", c.Gross_Sales_BDT - c.Discount_BDT)

# --- 6e. Tidy types
for col in ["Quantity", "Year", "Month", "Observed_Price_BDT", "List_Price_BDT", "Product_Launch_Year"]:
    clean[col] = clean[col].round().astype("int64")
clean = clean.drop(columns="Date_parsed")

print(pd.DataFrame(log, columns=["step", "column", "rows changed"]).to_string(index=False))

# %% 7. CHECK the result
print("Missing cells left:", int(clean.isna().sum().sum()))
print("Rows: before", len(dirty), "-> after", len(clean), "(dropped", len(dirty) - len(clean), ")")

rows = []
for col, idx in injected.items():
    idx = pd.Index(idx).intersection(clean.index)            # rows that survived
    a, b = clean.loc[idx, col], truth.loc[idx, col]
    if col == "Date":
        continue
    if pd.api.types.is_numeric_dtype(b):
        ok = ((a.astype(float) - b.astype(float)).abs() <= 0.011).mean()
    else:
        ok = (a == b).mean()
    rows.append((col, len(idx), round(ok * 100, 1)))
result = pd.DataFrame(rows, columns=["column", "cells blanked", "% restored exactly"])
print(result.to_string(index=False))
# Formulas and lookups restore ~100%. 'Unknown' columns restore 0% BY DESIGN: the true value is gone.
# Discount_% is ~90%: recomputed from rounded BDT amounts, so it can differ in the 2nd decimal on small orders.

cmp = pd.DataFrame({"original mean": truth[["Quantity", "Net_Sales_BDT", "Discount_%"]].mean(),
                    "after treatment": clean[["Quantity", "Net_Sales_BDT", "Discount_%"]].mean()}).round(3)
print(cmp)

# %% 8. Save
OUT = Path("practice_after_missing_value_treatment.csv")
clean.sort_index().to_csv(OUT, index=False)
pd.DataFrame(log, columns=["step", "column", "rows changed"]).to_csv("missing_value_treatment_log.csv", index=False)
print("Saved:", OUT.resolve())