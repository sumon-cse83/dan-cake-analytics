"""Layer 5: feature engineering for the monthly product-demand forecasting problem.

Design rule (prevents *data leakage*): a feature for month t may only use information
available at the end of month t-1, plus things that are planned in advance (the calendar,
the product's list price, its launch date). The same function is used for training,
testing and for recursive forecasting of future months, so the rule is enforced once.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 6, 12]


def pack_grams(pack: str) -> float:
    """Total grams in a pack: '30g x 12' -> 360, '27-30g' -> 28.5, '5 pcs' -> NaN."""
    m = re.search(r"(\d+)(?:-(\d+))?\s*g", pack)
    if not m:
        return np.nan
    g = (float(m.group(1)) + float(m.group(2))) / 2 if m.group(2) else float(m.group(1))
    k = re.search(r"x\s*(\d+)", pack)
    return g * (int(k.group(1)) if k else 1)


def monthly_tables(d: pd.DataFrame):
    """Return (units wide table months x products, list-price table, product metadata)."""
    ym = d.Date.dt.to_period("M").dt.to_timestamp()
    units = d.pivot_table(index=ym, columns="Product_ID", values="Quantity", aggfunc="sum", observed=True)
    price = d.pivot_table(index=ym, columns="Product_ID", values="List_Price_BDT", aggfunc="mean", observed=True)
    units.columns, price.columns = units.columns.astype(str), price.columns.astype(str)
    first = units.notna().idxmax()
    for p in units.columns:                                  # after launch, a month with no sale is a zero
        units.loc[units.index >= first[p], p] = units.loc[units.index >= first[p], p].fillna(0)
    meta = (d.groupby("Product_ID", observed=True)
            .agg(Product_Name=("Product_Name", "first"), Category=("Category", "first"),
                 Pack_Size=("Pack_Size", "first"), launch=("Product_Launch_Year", "first")).reset_index())
    meta["Product_ID"] = meta.Product_ID.astype(str)
    meta[["Product_Name", "Category", "Pack_Size"]] = meta[["Product_Name", "Category", "Pack_Size"]].astype(str)
    meta["first_month"] = meta.Product_ID.map(first)
    meta["pack_grams"] = meta.Pack_Size.map(pack_grams)
    meta["is_gift_pack"] = meta.Pack_Size.str.contains(r"x\s*\d+|12 pcs").astype(int)
    meta = meta.set_index("Product_ID")
    return units, price, meta


def build_features(units: pd.DataFrame, price: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Long table (month, product) with target `units` and leakage-free predictors."""
    U = units
    feats = {f"lag_{k}": U.shift(k) for k in LAGS}
    prev = U.shift(1)
    feats["roll3_mean"] = prev.rolling(3).mean()
    feats["roll6_mean"] = prev.rolling(6).mean()
    feats["roll12_mean"] = prev.rolling(12).mean()
    feats["roll3_std"] = prev.rolling(3).std()
    feats["yoy_growth_lag"] = (U.shift(1) / U.shift(13)).replace([np.inf, -np.inf], np.nan) - 1   # last month's YoY growth
    feats["trend_12"] = (prev.rolling(12).mean() / prev.shift(12).rolling(12).mean()).replace([np.inf, -np.inf], np.nan) - 1
    # price is planned in advance (published list price); fill forward so future months can use the last known price
    P = price.reindex(U.index).ffill()
    feats["list_price"] = P
    feats["price_change_yoy"] = P / P.shift(12) - 1
    long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    long["units"] = U.stack(future_stack=True)
    long.index.names = ["month", "Product_ID"]
    long = long.reset_index()
    long = long.merge(meta[["Category", "pack_grams", "is_gift_pack", "first_month"]], left_on="Product_ID", right_index=True)
    long = long[long.month >= long.first_month]                     # nothing before launch
    m = long.month
    long["month_num"] = m.dt.month
    long["month_sin"] = np.sin(2 * np.pi * m.dt.month / 12)
    long["month_cos"] = np.cos(2 * np.pi * m.dt.month / 12)
    long["is_december"] = (m.dt.month == 12).astype(int)
    long["is_february"] = (m.dt.month == 2).astype(int)
    long["days_in_month"] = m.dt.days_in_month
    long["product_age_m"] = (m.dt.year - long.first_month.dt.year) * 12 + (m.dt.month - long.first_month.dt.month)
    long["t_index"] = (m.dt.year - 2020) * 12 + m.dt.month
    return long.drop(columns="first_month").sort_values(["month", "Product_ID"]).reset_index(drop=True)


NUMERIC_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "roll3_mean", "roll6_mean", "roll12_mean",
                    "roll3_std", "yoy_growth_lag", "trend_12", "list_price", "price_change_yoy", "pack_grams",
                    "is_gift_pack", "month_num", "month_sin", "month_cos", "is_december", "is_february",
                    "days_in_month", "product_age_m", "t_index"]
CATEGORICAL_FEATURES = ["Category"]
