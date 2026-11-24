# Step 2 – Duplicate removal (hands-on)
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("Dan_Cake_Bangladesh_500000_Transactions_2020_2026.csv")
SEED = 42
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# 1. Load
df = pd.read_csv(DATA)
print("rows, columns:", df.shape)

# 2. Look for duplicates in the REAL file
# exact = every column identical          -> safe to delete
# key   = same Transaction_ID, values differ -> need a rule (which version is right?)
# near  = different ID, same date/product/region... -> may be a genuine repeat sale, do NOT delete blindly
print("Exact duplicate rows      :", int(df.duplicated().sum()))
print("Duplicate Transaction_IDs :", int(df["Transaction_ID"].duplicated().sum()))
hidden = df["Transaction_ID"].str.strip().str.upper().duplicated().sum()
print("Duplicate IDs after trimming spaces / upper-casing:", int(hidden))
near_cols = ["Date", "Product_ID", "Region", "Channel", "Customer_Type", "Quantity"]
print("Near-duplicates (same", near_cols, "):", int(df.duplicated(subset=near_cols).sum()),
      "-> many legitimate repeat orders, so these are only FLAGGED, never auto-deleted")

# 3. Create PRACTICE data: copy the file and add duplicates on purpose
rng = np.random.default_rng(SEED)
truth = df.set_index("Transaction_ID")            # answer key
pick = lambda n: df.iloc[rng.choice(len(df), n, replace=False)].copy()

A = pick(500)                                      # A: exact copies
B = pick(100)                                      # B: hidden copies - ID typed with lower case and spaces
B["Transaction_ID"] = " " + B["Transaction_ID"].str.lower() + " "
C = pick(150)                                      # C: older, LESS complete version (Region lost, Quantity wrong)
C["Region"] = np.nan
C["Quantity"] = C["Quantity"] + 5
D = pick(50)                                       # D: two COMPLETE versions that disagree
D["Quantity"] = D["Quantity"] + 5

dirty = pd.concat([df, A, B, C, D], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
print("Practice file rows:", len(dirty), "(original", len(df), "+", len(dirty) - len(df), "extra)")
print("Exact duplicates now:", int(dirty.duplicated().sum()), "| duplicate IDs now:", int(dirty.Transaction_ID.duplicated().sum()))

# 4. Look at an example before treating
some_id = C["Transaction_ID"].iloc[0]
print(dirty[dirty.Transaction_ID == some_id][["Transaction_ID", "Date", "Product_ID", "Region", "Quantity", "Net_Sales_BDT"]])

# 5. TREAT the duplicates
clean = dirty.copy()
removed = []                                       # keep everything we remove, with the reason

# 5a. Standardise first: otherwise ' dc001 ' and 'DC001' look like different rows
for col in clean.select_dtypes(include=["object", "string"]).columns:
    clean[col] = clean[col].str.strip()
for col in ["Transaction_ID", "Product_ID"]:
    clean[col] = clean[col].str.upper()

# 5b. Exact duplicates: keep the first, remove the rest
m = clean.duplicated(keep="first")
removed.append(clean[m].assign(reason="exact duplicate"))
clean = clean[~m]
print("5b exact duplicates removed:", int(m.sum()))

# 5c. Same ID, different content: keep the MOST COMPLETE version (fewest empty cells)
clean = clean.assign(_na=clean.isna().sum(axis=1), _pos=np.arange(len(clean)))
srt = clean.sort_values(["Transaction_ID", "_na", "_pos"])           # best version first within each ID
is_extra = srt.duplicated("Transaction_ID", keep="first")
best_count = srt._na.eq(srt.groupby("Transaction_ID")._na.transform("min"))
ties = srt[best_count].groupby("Transaction_ID").size()
ambiguous_ids = ties[ties > 1].index                                  # several equally complete, different versions
removed.append(srt[is_extra].assign(reason="same ID, less complete or later version"))
clean = srt[~is_extra].sort_values("_pos")
print("5c same-ID conflicts resolved:", int(is_extra.sum()), "| of which ambiguous (need a human):", len(ambiguous_ids))

# 5d. Near-duplicates: FLAG only
clean["flag_near_duplicate"] = clean.duplicated(subset=near_cols, keep="first")
print("5d near-duplicates flagged (kept):", int(clean.flag_near_duplicate.sum()))

clean = clean.drop(columns=["_na", "_pos"])
removed = pd.concat(removed).drop(columns=["_na", "_pos"], errors="ignore")

# 6. CHECK the result
print("Rows: practice", len(dirty), "-> clean", len(clean), "| removed", len(removed))
print("Duplicate IDs left:", int(clean.Transaction_ID.duplicated().sum()), "| exact duplicates left:",
      int(clean.drop(columns="flag_near_duplicate").duplicated().sum()))
print("All original transactions still present:", set(clean.Transaction_ID) == set(df.Transaction_ID))

back = clean.set_index("Transaction_ID")
def correct(ids):   # did we keep the original version? (compare quantity and region with the answer key)
    ids = pd.Index(ids.str.strip().str.upper()).intersection(back.index)
    return ((back.loc[ids, "Quantity"] == truth.loc[ids, "Quantity"]) & (back.loc[ids, "Region"] == truth.loc[ids, "Region"])).mean() * 100
print(f"Group A (exact copies)         correct: {correct(A.Transaction_ID):.0f}%")
print(f"Group B (hidden copies)        correct: {correct(B.Transaction_ID):.0f}%")
print(f"Group C (less complete copy)   correct: {correct(C.Transaction_ID):.0f}%   <- 'most complete' rule works")
print(f"Group D (two complete, differ) correct: {correct(D.Transaction_ID):.0f}%   <- a coin flip: needs a person")

# 7. Save
clean.to_csv("practice_after_duplicate_removal.csv", index=False)
removed.to_csv("removed_duplicates.csv", index=False)
pd.Series(ambiguous_ids, name="Transaction_ID").to_csv("duplicates_to_review.csv", index=False)
print("Saved: practice_after_duplicate_removal.csv, removed_duplicates.csv, duplicates_to_review.csv")