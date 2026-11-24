"""Central configuration: every path, constant and split date lives here.

Changing the project to a new dataset usually means editing ONLY this file
(paths, column names and the train/test cut-off).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- input / output paths
RAW_CSV = ROOT / "Dan_Cake_Bangladesh_500000_Transactions_2020_2026.csv"
PROCESSED = ROOT / "data" / "processed"
WAREHOUSE_DB = ROOT / "data" / "warehouse" / "dancake.db"
DASHBOARD_DATA = ROOT / "data" / "dashboard"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
METRICS = OUTPUTS / "metrics"
MODELS = OUTPUTS / "models"
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
TEX_TABLES = REPORT / "tables"
TEX_GENERATED = REPORT / "generated"
SITE = ROOT / "docs"          # GitHub Pages serves this folder

for _p in (PROCESSED, WAREHOUSE_DB.parent, DASHBOARD_DATA, TABLES, METRICS, MODELS,
           FIGURES, TEX_TABLES, TEX_GENERATED, SITE):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data contract
DATE_FORMAT = "%d-%m-%y"
CATEGORICAL = ["Product_ID", "Product_Name", "Category", "Pack_Size", "Region", "Channel",
               "Customer_Type", "Month_Name", "Quarter"]
NUMERIC = ["Observed_Price_BDT", "List_Price_BDT", "Market_Price_BDT", "Discount_%",
           "Unit_Selling_Price_BDT", "Quantity", "Gross_Sales_BDT", "Discount_BDT", "Net_Sales_BDT"]
# Domain rules: values outside these ranges are impossible (errors), not merely unusual.
DOMAIN_RULES = {"Quantity": (1, 500), "Market_Price_BDT": (0.01, 1000), "Discount_%": (0, 50),
                "Net_Sales_BDT": (0, 1_000_000)}

# ---------------------------------------------------------------- modelling
SEED = 42
TRAIN_END = "2024-12-31"          # models learn from 2020-2024 ...
TEST_START = "2025-01-01"         # ... and are judged on 2025-01 to 2026-09 (unseen "future")
LARGE_ORDER_UNITS = 10            # classification target: an order of >= 10 units
FORECAST_HORIZON = 12             # months forecast beyond the last observed month

# ---------------------------------------------------------------- visual identity (report + dashboard)
PALETTE = ["#1F4E79", "#C55A11", "#2E8B57", "#7F3C8D", "#B8860B", "#3A7CA5",
           "#A23B72", "#5B8E7D", "#8C564B", "#4D4D4D", "#D1495B", "#00798C", "#6A4C93"]
PRIMARY = PALETTE[0]
ACCENT = PALETTE[1]
