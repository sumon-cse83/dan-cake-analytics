"""Automated tests: the preprocessing must repair dirty data and the features must not leak.

Run:  python -m pytest tests -q
They use a small synthetic sample, so they finish in seconds and do not need the raw CSV.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dancake import data  # noqa: E402
from dancake.features import build_features, pack_grams  # noqa: E402


def make_sample(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    prods = rng.choice(["MUF001", "POU002", "SWR001"], n)
    info = {"MUF001": ("Chocolate Muffin", "Muffin", "30g", 20), "POU002": ("Vanilla Pound Cake", "Pound Cake", "220g", 100),
            "SWR001": ("Chocolate Swiss Roll", "Swiss Roll", "140g", 105)}
    qty = rng.integers(1, 20, n)
    lp = np.array([info[p][3] for p in prods], float)
    mp = (lp * rng.uniform(.97, 1.0, n)).round(2)
    disc = rng.uniform(0, 5, n).round(2)
    gross = (qty * mp).round(2)
    dbdt = (gross * disc / 100).round(2)
    df = pd.DataFrame({
        "Transaction_ID": [f"DC{i:010d}" for i in range(n)], "Date": dates, "Year": dates.year, "Month": dates.month,
        "Month_Name": dates.strftime("%b"), "Quarter": "Q" + dates.quarter.astype(str),
        "Product_ID": prods, "Product_Name": [info[p][0] for p in prods], "Category": [info[p][1] for p in prods],
        "Pack_Size": [info[p][2] for p in prods], "Observed_Price_BDT": lp.astype(int) + 5, "List_Price_BDT": lp.astype(int),
        "Market_Price_BDT": mp, "Discount_%": disc, "Unit_Selling_Price_BDT": (mp * (1 - disc / 100)).round(2),
        "Quantity": qty, "Gross_Sales_BDT": gross, "Discount_BDT": dbdt, "Net_Sales_BDT": (gross - dbdt).round(2),
        "Region": rng.choice(["Dhaka", "Sylhet"], n), "Channel": rng.choice(["General Trade", "Distributor"], n),
        "Customer_Type": rng.choice(["Retailer", "Corporate"], n), "Product_Launch_Year": 2020})
    for c in data.C.CATEGORICAL + ["Transaction_ID"]:
        df[c] = df[c].astype("string")
    return df


def test_clean_data_passes_validation():
    rep = data.validate(make_sample())
    assert rep["missing_cells"] == 0 and rep["duplicate_ids"] == 0
    assert rep["identity_gross_max_err"] < 0.01


def test_missing_values_are_derived_exactly():
    truth = make_sample()
    dirty = truth.copy()
    dirty["Quantity"] = dirty["Quantity"].astype(float)
    holes = dirty.sample(60, random_state=1).index
    dirty.loc[holes, "Quantity"] = np.nan
    clean, _ = data.clean(dirty)
    merged = clean.set_index("Transaction_ID").loc[truth.Transaction_ID.astype(str), "Quantity"].values
    assert (merged == truth.Quantity.values).all()


def test_duplicates_and_impossible_values_are_fixed():
    df = make_sample()
    dup = df.iloc[:10].copy()
    bad = df.copy()
    bad.loc[0:4, "Discount_%"] = 250.0                       # impossible discount
    bad = pd.concat([bad, dup], ignore_index=True)           # exact duplicates
    clean, log = data.clean(bad)
    assert len(clean) == len(df)
    assert clean["Discount_%"].max() <= 50
    # rows 0-4 were altered before copying, so 5 copies are exact and 5 share only the ID
    assert dict(log)["exact duplicate rows removed"] == 5
    assert dict(log)["duplicate Transaction_ID removed (kept most complete)"] == 5


def test_outliers_are_flagged_not_deleted():
    df = make_sample()
    df.loc[0, "Quantity"] = 400
    df.loc[0, "Gross_Sales_BDT"] = 400 * df.loc[0, "Market_Price_BDT"]
    clean, _ = data.clean(df)
    assert len(clean) == len(df)
    assert clean.loc[clean.Transaction_ID == df.loc[0, "Transaction_ID"], "flag_outlier_qty"].iat[0] == 1


@pytest.mark.parametrize("pack,grams", [("30g", 30), ("27-30g", 28.5), ("30g x 12", 360), ("33g x 6", 198)])
def test_pack_grams(pack, grams):
    assert pack_grams(pack) == pytest.approx(grams)


def test_features_do_not_use_the_current_month():
    months = pd.date_range("2021-01-01", periods=30, freq="MS")
    units = pd.DataFrame({"A": np.arange(30, dtype=float) * 10 + 100}, index=months)
    price = pd.DataFrame({"A": 10.0}, index=months)
    meta = pd.DataFrame({"Category": ["Muffin"], "pack_grams": [30.0], "is_gift_pack": [0], "first_month": [months[0]]},
                        index=pd.Index(["A"], name="Product_ID"))
    f = build_features(units, price, meta).set_index("month")
    m = months[20]
    assert f.loc[m, "lag_1"] == units.loc[months[19], "A"]
    assert f.loc[m, "roll3_mean"] == pytest.approx(units.loc[months[17]:months[19], "A"].mean())
    # changing the current month's value must not change any of its features
    units2 = units.copy(); units2.loc[m, "A"] = 1e6
    f2 = build_features(units2, price, meta).set_index("month")
    cols = [c for c in f.columns if c not in ("units", "Product_ID", "Category")]
    pd.testing.assert_series_equal(f.loc[m, cols], f2.loc[m, cols], check_names=False)
