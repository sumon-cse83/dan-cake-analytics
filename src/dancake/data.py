"""Layer 1-3 of the framework: ingestion, validation, cleaning and storage.

The functions are written so that they also work on a *dirty* file: the real
Dan Cake file turned out to be complete, but the same code handles missing
values, duplicate keys, impossible values and outliers if they appear.
"""
from __future__ import annotations

import json
import sqlite3
import time

import numpy as np
import pandas as pd

from . import config as C

PLACEHOLDERS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing", "unknown"}


# ----------------------------------------------------------------------------- ingestion
def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    """Read the CSV with explicit types (data type conversion step)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format=C.DATE_FORMAT, errors="coerce")
    for c in C.NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in C.CATEGORICAL + ["Transaction_ID"]:
        df[c] = df[c].astype("string").str.strip()
    return df


# ----------------------------------------------------------------------------- validation
def validate(df: pd.DataFrame) -> dict:
    """Run every data-quality check and return a report (nothing is changed here)."""
    rep: dict = {"rows": int(len(df)), "columns": int(df.shape[1])}
    rep["missing_cells"] = int(df.isna().sum().sum())
    rep["missing_by_column"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    hidden = {}
    for c in C.CATEGORICAL:
        n = int(df[c].str.lower().isin(PLACEHOLDERS).sum())
        if n:
            hidden[c] = n
    rep["hidden_placeholders"] = hidden
    rep["exact_duplicates"] = int(df.duplicated().sum())
    rep["duplicate_ids"] = int(df["Transaction_ID"].str.upper().duplicated().sum())
    rep["rule_violations"] = {c: int(((df[c] < lo) | (df[c] > hi)).sum()) for c, (lo, hi) in C.DOMAIN_RULES.items()}
    # Accounting identities: does each row add up?
    rep["identity_gross_max_err"] = float((df.Gross_Sales_BDT - df.Quantity * df.Market_Price_BDT).abs().max())
    rep["identity_net_max_err"] = float((df.Net_Sales_BDT - (df.Gross_Sales_BDT - df.Discount_BDT)).abs().max())
    rep["identity_price_max_err"] = float(
        (df.Unit_Selling_Price_BDT - df.Market_Price_BDT * (1 - df["Discount_%"] / 100)).abs().max())
    # Consistency of calendar columns with the date
    rep["calendar_mismatch"] = int(((df.Date.dt.year != df.Year) | (df.Date.dt.month != df.Month)).sum())
    # Each Product_ID must map to exactly one name / category / pack
    rep["product_attribute_conflicts"] = int(
        (df.groupby("Product_ID")[["Product_Name", "Category", "Pack_Size"]].nunique() > 1).any(axis=1).sum())
    rep["date_min"], rep["date_max"] = str(df.Date.min().date()), str(df.Date.max().date())
    rep["n_products"] = int(df.Product_ID.nunique())
    rep["n_regions"] = int(df.Region.nunique())
    rep["n_channels"] = int(df.Channel.nunique())
    rep["n_customer_types"] = int(df.Customer_Type.nunique())
    return rep


# ----------------------------------------------------------------------------- cleaning
def flag_outliers_log_iqr(values: pd.Series, groups: pd.Series, k: float = 1.5) -> pd.Series:
    """Per-group IQR fence on log1p(values): robust for skewed counts such as Quantity."""
    x = np.log1p(values.clip(lower=0))
    g = x.groupby(groups)
    q1, q3 = g.transform(lambda v: v.quantile(.25)), g.transform(lambda v: v.quantile(.75))
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Apply the cleaning rules in a fixed order and return (clean data, audit log)."""
    log: list[tuple[str, int]] = []
    d = df.copy()

    # 1. hidden placeholders -> real missing values
    for c in C.CATEGORICAL:
        m = d[c].str.lower().isin(PLACEHOLDERS)
        d.loc[m, c] = pd.NA
        log.append((f"placeholder to NaN: {c}", int(m.sum())))

    # 2. exact duplicates, then duplicate keys (keep the most complete version)
    n = len(d)
    d = d.drop_duplicates()
    log.append(("exact duplicate rows removed", n - len(d)))
    d["_complete"] = d.notna().sum(axis=1)
    d["_key"] = d["Transaction_ID"].str.upper()
    n = len(d)
    d = d.sort_values("_complete", ascending=False).drop_duplicates("_key").drop(columns=["_complete", "_key"])
    log.append(("duplicate Transaction_ID removed (kept most complete)", n - len(d)))

    # 3. rows without an essential field cannot be repaired
    n = len(d)
    d = d.dropna(subset=["Transaction_ID", "Date", "Product_ID"])
    log.append(("rows dropped: essential field missing", n - len(d)))

    # 4. impossible values -> NaN, then derive from the accounting identities
    for c, (lo, hi) in C.DOMAIN_RULES.items():
        m = (d[c] < lo) | (d[c] > hi)
        d.loc[m, c] = np.nan
        log.append((f"impossible value blanked: {c}", int(m.sum())))
    derive = {
        "Quantity": (d.Gross_Sales_BDT / d.Market_Price_BDT).round(),
        "Market_Price_BDT": (d.Gross_Sales_BDT / d.Quantity).round(2),
        "Gross_Sales_BDT": (d.Quantity * d.Market_Price_BDT).round(2),
        "Net_Sales_BDT": (d.Gross_Sales_BDT - d.Discount_BDT).round(2),
        "Discount_%": (d.Discount_BDT / d.Gross_Sales_BDT * 100).round(2),
    }
    for c, v in derive.items():
        m = d[c].isna() & v.notna()
        d.loc[m, c] = v[m]
        log.append((f"derived from identity: {c}", int(m.sum())))

    # 5. remaining numeric gaps -> product median; categorical gaps -> product mode or 'Unknown'
    for c in C.NUMERIC:
        m = d[c].isna()
        if m.any():
            d[c] = d[c].fillna(d.groupby("Product_ID")[c].transform("median"))
        log.append((f"median imputed: {c}", int(m.sum())))
    for c in ["Product_Name", "Category", "Pack_Size"]:
        m = d[c].isna()
        if m.any():
            mode = d.groupby("Product_ID")[c].agg(lambda s: s.mode().iat[0] if s.notna().any() else "Unknown")
            d.loc[m, c] = d.loc[m, "Product_ID"].map(mode)
        log.append((f"product mode imputed: {c}", int(m.sum())))
    for c in ["Region", "Channel", "Customer_Type"]:
        m = d[c].isna()
        d[c] = d[c].fillna("Unknown")
        log.append((f"labelled 'Unknown': {c}", int(m.sum())))

    # 6. calendar columns are always rebuilt from the date (single source of truth)
    d["Year"], d["Month"] = d.Date.dt.year, d.Date.dt.month
    d["Month_Name"] = d.Date.dt.strftime("%b")
    d["Quarter"] = "Q" + d.Date.dt.quarter.astype(str)

    # 7. unusual-but-possible values are kept and flagged (bulk orders are real business)
    d["flag_outlier_qty"] = flag_outliers_log_iqr(d["Quantity"], d["Product_ID"]).astype(int)
    log.append(("Quantity outliers flagged (kept)", int(d.flag_outlier_qty.sum())))

    d["Quantity"] = d["Quantity"].astype(int)
    d = d.sort_values(["Date", "Transaction_ID"]).reset_index(drop=True)
    for c in C.CATEGORICAL:
        d[c] = d[c].astype("category")
    return d, log


# ----------------------------------------------------------------------------- storage (star schema)
def build_warehouse(d: pd.DataFrame, db_path=C.WAREHOUSE_DB) -> dict:
    """Store the clean data as a small star schema in SQLite (portable SQL)."""
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    dim_product = (d.groupby("Product_ID", observed=True)
                   .agg(product_name=("Product_Name", "first"), category=("Category", "first"),
                        pack_size=("Pack_Size", "first"), launch_year=("Product_Launch_Year", "first"),
                        mrp_bdt=("Observed_Price_BDT", "first"))
                   .reset_index().rename(columns={"Product_ID": "product_id"}))
    seg = d[["Region", "Channel", "Customer_Type"]].drop_duplicates().astype(str).sort_values(
        ["Region", "Channel", "Customer_Type"]).reset_index(drop=True)
    seg.insert(0, "segment_id", np.arange(1, len(seg) + 1))
    seg.columns = ["segment_id", "region", "channel", "customer_type"]
    dates = pd.DataFrame({"date": pd.date_range(d.Date.min(), d.Date.max(), freq="D")})
    dates["date_id"] = dates.date.dt.strftime("%Y%m%d").astype(int)
    dates = dates.assign(year=dates.date.dt.year, quarter=dates.date.dt.quarter, month=dates.date.dt.month,
                         month_name=dates.date.dt.strftime("%b"), day_of_week=dates.date.dt.dayofweek,
                         is_weekend=dates.date.dt.dayofweek.isin([4, 5]).astype(int))  # Fri-Sat weekend in Bangladesh
    dates["date"] = dates.date.dt.strftime("%Y-%m-%d")
    fact = d.merge(seg, left_on=[d.Region.astype(str), d.Channel.astype(str), d.Customer_Type.astype(str)],
                   right_on=["region", "channel", "customer_type"])
    fact = pd.DataFrame({
        "transaction_id": fact.Transaction_ID.astype(str),
        "date_id": fact.Date.dt.strftime("%Y%m%d").astype(int),
        "product_id": fact.Product_ID.astype(str), "segment_id": fact.segment_id,
        "list_price_bdt": fact.List_Price_BDT, "market_price_bdt": fact.Market_Price_BDT,
        "discount_pct": fact["Discount_%"], "unit_price_bdt": fact.Unit_Selling_Price_BDT,
        "quantity": fact.Quantity, "gross_sales_bdt": fact.Gross_Sales_BDT,
        "discount_bdt": fact.Discount_BDT, "net_sales_bdt": fact.Net_Sales_BDT,
        "flag_outlier_qty": fact.flag_outlier_qty})
    t = time.time()
    dim_product.to_sql("dim_product", con, index=False)
    seg.to_sql("dim_segment", con, index=False)
    dates.to_sql("dim_date", con, index=False)
    fact.to_sql("fact_sales", con, index=False, chunksize=50_000)
    con.executescript("""
        CREATE INDEX ix_fact_date ON fact_sales(date_id);
        CREATE INDEX ix_fact_product ON fact_sales(product_id);
        CREATE INDEX ix_fact_segment ON fact_sales(segment_id);
    """)
    counts = {t_: con.execute(f"SELECT COUNT(*) FROM {t_}").fetchone()[0]
              for t_ in ["fact_sales", "dim_product", "dim_segment", "dim_date"]}
    # Reconciliation: the warehouse must give the same total as pandas
    wh_total = con.execute("SELECT SUM(net_sales_bdt) FROM fact_sales").fetchone()[0]
    con.close()
    counts["load_seconds"] = round(time.time() - t, 1)
    counts["reconciliation_diff_bdt"] = round(float(wh_total - d.Net_Sales_BDT.sum()), 4)
    return counts


def run() -> pd.DataFrame:
    raw = load_raw()
    report = validate(raw)
    d, log = clean(raw)
    report["after_cleaning_rows"] = int(len(d))
    report["after_cleaning_missing"] = int(d.isna().sum().sum())
    report["cleaning_log"] = log
    report["warehouse"] = build_warehouse(d)
    d.to_parquet(C.PROCESSED / "transactions_clean.parquet", index=False)
    json.dump(report, open(C.METRICS / "data_quality.json", "w"), indent=2)
    print("[data] rows:", len(d), "| quality report -> outputs/metrics/data_quality.json")
    return d


def load_clean() -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / "transactions_clean.parquet")
