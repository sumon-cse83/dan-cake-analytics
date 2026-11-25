"""Layer 8 support: compact data for the dashboard, a LaTeX macro file with every number quoted in the
report, and the JSON used by the static website.

Why aggregate? 500,000 rows are too large to ship with a free web app, but a monthly cube by
region x channel x customer type x category keeps every filter in the dashboard exact.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from . import config as C
from .utils import load_json


def build_cubes(d: pd.DataFrame):
    d = d.assign(month=d.Date.dt.to_period("M").dt.to_timestamp())
    dims = ["month", "Region", "Channel", "Customer_Type", "Category"]
    cube = d.groupby(dims, observed=True).agg(orders=("Quantity", "size"), units=("Quantity", "sum"),
                                              gross=("Gross_Sales_BDT", "sum"), discount=("Discount_BDT", "sum"),
                                              net=("Net_Sales_BDT", "sum"), large=("Quantity", lambda q: (q >= C.LARGE_ORDER_UNITS).sum())).reset_index()
    for c in dims[1:]:
        cube[c] = cube[c].astype(str)
    cube.to_parquet(C.DASHBOARD_DATA / "cube_monthly.parquet", index=False)
    pm = d.groupby(["month", "Product_ID", "Product_Name", "Category", "Region"], observed=True).agg(
        units=("Quantity", "sum"), net=("Net_Sales_BDT", "sum"), orders=("Quantity", "size")).reset_index()
    for c in ["Product_ID", "Product_Name", "Category", "Region"]:
        pm[c] = pm[c].astype(str)
    pm.to_parquet(C.DASHBOARD_DATA / "product_region_monthly.parquet", index=False)
    daily = d.groupby("Date").agg(net=("Net_Sales_BDT", "sum"), orders=("Quantity", "size")).reset_index()
    daily.to_parquet(C.DASHBOARD_DATA / "daily_sales.parquet", index=False)
    for name in ["data_quality", "eda", "forecasting", "classification", "clustering", "anomaly", "insights"]:
        shutil.copy(C.METRICS / f"{name}.json", C.DASHBOARD_DATA / f"{name}.json")
    print(f"[export] cube {len(cube):,} rows, product table {len(pm):,} rows")


def latex_numbers():
    """Every number quoted in the report text comes from here: \\newcommand{\\TotalSales}{...}."""
    e, f, c, k, a, q = (load_json(n) for n in ["eda", "forecasting", "classification", "clustering", "anomaly", "data_quality"])
    one = {r["Model"]: r for r in f["one_step"]}
    rec = {r["Model"]: r for r in f["recursive"]}
    comp = {r["Model"]: r for r in f["company_level"]}
    cls = {r["Model"]: r for r in c["results"]}
    hw = [v for kk, v in comp.items() if kk.startswith("Holt")][0]
    sn = one["Seasonal naive (same month last year)"]
    best = one[f["best_model"]]
    fu = f["future"]
    td = {r["Method"]: r for r in a["transaction_detection"]}
    n = {
        "NRows": f"{q['rows']:,}", "NCols": q["columns"], "NProducts": q["n_products"], "NRegions": q["n_regions"],
        "NChannels": q["n_channels"], "NCustTypes": q["n_customer_types"], "DateMin": "January 2020", "DateMax": "September 2026",
        "NOutlierFlags": f"{q['cleaning_log'][-1][1]:,}", "OutlierPct": f"{q['cleaning_log'][-1][1] / q['rows'] * 100:.2f}",
        "WarehouseSeconds": q["warehouse"]["load_seconds"],
        "TotalSales": f"{e['total_net_sales'] / 1e6:,.1f}", "TotalUnits": f"{e['total_units'] / 1e6:.2f}",
        "AOV": f"{e['avg_order_value']:.0f}", "AvgDiscount": f"{e['avg_discount_pct']:.2f}",
        "DiscountCost": f"{e['discount_cost'] / 1e6:.2f}",
        "CAGRSales": f"{e['cagr_sales'] * 100:.1f}", "CAGRUnits": f"{e['cagr_units'] * 100:.1f}", "CAGRPrice": f"{e['cagr_price'] * 100:.1f}",
        "AvgVolGrowth": f"{e['avg_volume_growth']:.1f}", "AvgPriceGrowth": f"{e['avg_price_growth']:.1f}",
        "YTDGrowth": f"{e['ytd_growth_latest'] * 100:.1f}",
        "PeakMonth": {"Dec": "December"}.get(e["peak_month"], e["peak_month"]), "PeakIndex": f"{e['peak_index']:.1f}",
        "TroughMonth": {"Feb": "February"}.get(e["trough_month"], e["trough_month"]), "TroughIndex": f"{e['trough_index']:.1f}",
        "DecPerDay": f"{e['seasonal_index_per_day']['Dec']:.1f}", "FebPerDay": f"{e['seasonal_index_per_day']['Feb']:.1f}",
        "NovIndex": f"{e['seasonal_index_total']['Nov']:.1f}", "OctIndex": f"{e['seasonal_index_total']['Oct']:.1f}",
        "KWMonthH": f"{e['kw_month_daily_H']:.1f}", "DowCV": f"{e['dow_cv'] * 100:.2f}",
        "TopCategory": e["top_category"], "TopCategoryShare": f"{e['top_category_share']:.1f}",
        "TopTenShare": f"{e['share_top10_products']:.1f}", "NClassA": e["abc_counts"].get("A", 0),
        "NClassB": e["abc_counts"].get("B", 0), "NClassC": e["abc_counts"].get("C", 0),
        "DhakaShare": f"{e['top_region_share']:.1f}", "DhakaCtgShare": f"{e['dhaka_ctg_share']:.1f}",
        "GTShare": f"{e['channel_share']['General Trade']:.1f}", "MTShare": f"{e['channel_share']['Modern Trade']:.1f}",
        "DistShare": f"{e['channel_share']['Distributor']:.1f}", "EcomShare": f"{e['channel_share']['E-commerce']:.1f}",
        "DistQty": f"{e['channel_qty']['Distributor']:.1f}", "EcomQty": f"{e['channel_qty']['E-commerce']:.1f}",
        "GTQty": f"{e['channel_qty']['General Trade']:.1f}", "MTQty": f"{e['channel_qty']['Modern Trade']:.1f}",
        "LargeRate": f"{e['large_order_rate'] * 100:.1f}", "QtySkew": f"{e['qty_skew']:.2f}", "ValueSkew": f"{e['value_skew']:.2f}",
        "QtyMedian": f"{e['qty_median']:.0f}",
        "EpsChannel": f"{e['tests']['Units/order by channel']['effect']:.3f}",
        "EpsRegion": f"{e['tests']['Units/order by region']['effect']:.3f}",
        "PCustType": f"{e['tests']['Units/order by customer type']['p']:.2f}",
        "RhoDisc": f"{e['tests']['Discount % vs units/order']['stat']:.4f}",
        "PDisc": f"{e['tests']['Discount % vs units/order']['p']:.2f}",
        "CramerV": f"{e['tests']['Large order vs channel']['effect']:.3f}",
        "CorrSalesQty": f"{e['corr_sales_qty']:.2f}", "CorrSalesPrice": f"{e['corr_sales_price']:.2f}",
        "CorrQtyPrice": f"{e['corr_qty_price']:.3f}",
        "Elasticity": f"{e['elasticity']:.2f}", "ElasticityLo": f"{e['elasticity_ci'][0]:.2f}",
        "ElasticityHi": f"{e['elasticity_ci'][1]:.2f}", "ElasticityP": f"{e['elasticity_p']:.2f}",
        "ElasticityNaive": f"{e['elasticity_naive']:.2f}", "ElasticityRtwo": f"{e['elasticity_r2']:.2f}",
        "FastCategory": e["fastest_category"], "FastCategoryCAGR": f"{e['fastest_category_cagr']:.1f}",
        # forecasting
        "PanelSeries": f["panel"]["series"], "PanelTrain": f"{f['panel']['train_rows']:,}", "PanelTest": f"{f['panel']['test_rows']:,}",
        "PanelFeatures": f["panel"]["n_features"], "TestMonths": f["panel"]["test_months"],
        "BestReg": f["best_model"], "BestRegRMSE": f"{best['RMSE']:.1f}", "BestRegMAE": f"{best['MAE']:.1f}",
        "BestRegWAPE": f"{best['WAPE']:.2f}", "BestRegRtwo": f"{best['R2']:.3f}", "BestRegMAPE": f"{best['MAPE']:.2f}",
        "SNaiveRMSE": f"{sn['RMSE']:.1f}", "SNaiveWAPE": f"{sn['WAPE']:.2f}", "SNaiveRtwo": f"{sn['R2']:.3f}",
        "NaiveRMSE": f"{one['Naive (last month)']['RMSE']:.1f}",
        "GainVsSNaive": f"{f['best_vs_seasonal_naive_pct']:.1f}", "GainVsNaive": f"{f['best_vs_naive_pct']:.1f}",
        "LinRMSE": f"{one['Linear Regression']['RMSE']:.1f}", "RFRMSE": f"{one['Random Forest']['RMSE']:.1f}",
        "XGBRMSE": f"{one['XGBoost']['RMSE']:.1f}", "DTRMSE": f"{one['Decision Tree']['RMSE']:.1f}",
        "GBRMSE": f"{one['Gradient Boosting']['RMSE']:.1f}", "KNNRMSE": f"{one['K-Nearest Neighbors']['RMSE']:.1f}",
        "SVRRMSE": f"{one['Support Vector Regression']['RMSE']:.1f}",
        "WorstRegRMSE": f"{max(r['RMSE'] for r in f['one_step'] if 'naive' not in r['Model'].lower()):.1f}",
        "BestRec": f["best_recursive_model"], "BestRecWAPE": f"{rec[f['best_recursive_model']]['WAPE']:.2f}",
        "BestRecRtwo": f"{rec[f['best_recursive_model']]['R2']:.3f}", "BestRecRMSE": f"{rec[f['best_recursive_model']]['RMSE']:.1f}",
        "HWMAPE": f"{hw['MAPE']:.2f}",
        "FutSales": f"{fu['net_sales'] / 1e6:.1f}", "FutSalesLo": f"{fu['net_sales_lo'] / 1e6:.1f}",
        "FutSalesHi": f"{fu['net_sales_hi'] / 1e6:.1f}", "FutGrowth": f"{fu['growth_sales_pct']:.1f}",
        "FutUnitsGrowth": f"{fu['growth_units_pct']:.1f}", "FutUnitsMLGrowth": f"{fu['growth_units_ml_raw_pct']:.1f}",
        "FutUnits": f"{fu['units']:,.0f}", "PrevSales": f"{fu['prev12_sales'] / 1e6:.1f}",
        "FutTopCategory": fu["top_category"],
        # classification
        "ClsBest": c["best_model"], "ClsAUC": f"{c['results'][0]['ROC-AUC']:.3f}", "ClsF": f"{c['results'][0]['F1']:.3f}",
        "ClsRecall": f"{c['results'][0]['Recall']:.3f}", "ClsPrecision": f"{c['results'][0]['Precision']:.3f}",
        "ClsLift": f"{c['results'][0]['Lift@10%']:.2f}", "ClsAUCmin": f"{min(r['ROC-AUC'] for r in c['results']):.3f}",
        "ClsXGBAUC": f"{cls['XGBoost']['ROC-AUC']:.3f}", "ClsKNNAUC": f"{cls['K-Nearest Neighbors']['ROC-AUC']:.3f}",
        "ClsTrain": f"{c['n_train']:,}", "ClsTest": f"{c['n_test']:,}", "ClsPosTest": f"{c['positive_rate_test'] * 100:.1f}",
        "ClsPosTrain": f"{c['positive_rate_train'] * 100:.1f}", "ClsMajority": f"{c['majority_baseline_accuracy'] * 100:.1f}",
        "ClsChannelImp": f"{c['xgb_importance']['Channel'] * 100:.0f}", "ClsRegionImp": f"{c['xgb_importance']['Region'] * 100:.0f}",
        "LargeDist": f"{c['large_rate_by_channel']['Distributor'] * 100:.0f}", "LargeEcom": f"{c['large_rate_by_channel']['E-commerce'] * 100:.0f}",
        # clustering
        "ProdK": k["product_k"], "ProdSil": f"{k['product_scores']['silhouette']:.3f}", "ProdDB": f"{k['product_scores']['davies_bouldin']:.3f}",
        "MktK": k["segment_k"], "MktSil": f"{k['segment_scores']['silhouette']:.3f}", "MktDB": f"{k['segment_scores']['davies_bouldin']:.3f}",
        # anomaly
        "AnomDays": a["daily"]["flag_any"], "AnomDaysZ": a["daily"]["flag_z"], "AnomDaysIF": a["daily"]["flag_iso"],
        "AnomIFRecall": f"{td['Isolation Forest (4 features)']['Recall'] * 100:.1f}",
        "AnomIQRRecall": f"{td['IQR on log quantity (per product)']['Recall'] * 100:.1f}",
        "AnomRuleRecall": f"{td['Business rules (price, discount)']['Recall'] * 100:.1f}",
        "AnomComboRecall": f"{td['Rules OR Isolation Forest']['Recall'] * 100:.1f}",
        "AnomIFPrecision": f"{td['Isolation Forest (4 features)']['Precision'] * 100:.1f}",
    }
    lines = ["% AUTO-GENERATED by src/dancake/export.py - do not edit by hand; re-run the pipeline instead."]
    for key, v in n.items():
        val = str(v).replace("&", r"\&").replace("%", r"\%")
        lines.append(rf"\newcommand{{\{key}}}{{{val}\xspace}}")
    (C.TEX_GENERATED / "numbers.tex").write_text("\n".join(lines), encoding="utf-8")
    json.dump(n, open(C.METRICS / "report_numbers.json", "w"), indent=1, default=str)
    print(f"[export] {len(n)} LaTeX macros -> report/generated/numbers.tex")


def hyperparameter_table():
    from .utils import to_latex_table
    tp = load_json("forecasting")["tuned_params"]
    rows = [{"Model": m, "Selected hyper-parameters": ", ".join(f"{k}={v}" for k, v in sorted(p.items())) or "none (closed form)"}
            for m, p in tp.items()]
    to_latex_table(pd.DataFrame(rows), "hyperparameters", "Hyper-parameters selected by randomised search with "
                   "expanding-window validation (demand forecasting)", "tab:hyper", col_align=r"l>{\raggedright\arraybackslash}p{10cm}",
                   font=r"\footnotesize")


def run(d: pd.DataFrame):
    build_cubes(d)
    latex_numbers()
    hyperparameter_table()
    (C.REPORT / "sql").mkdir(exist_ok=True)                    # keep the report folder self-contained (Overleaf)
    shutil.copy(C.ROOT / "sql" / "kpi_queries.sql", C.REPORT / "sql" / "kpi_queries.sql")
    from . import report_bn                                    # Bangla captions/notes/insights for main_bn.tex
    report_bn.run()
