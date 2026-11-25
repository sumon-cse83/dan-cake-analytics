"""Dan Cake Bangladesh - Intelligent Sales Analytics Dashboard (Streamlit).

Run locally:   streamlit run dashboard/app.py
Data:          data/dashboard/*.parquet and *.json, produced by `python run_pipeline.py`
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "dashboard"
PALETTE = ["#1F4E79", "#C55A11", "#2E8B57", "#7F3C8D", "#B8860B", "#3A7CA5", "#A23B72", "#5B8E7D", "#8C564B",
           "#4D4D4D", "#D1495B", "#00798C", "#6A4C93"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Dan Cake Analytics", page_icon="🍰", layout="wide")
px.defaults.color_discrete_sequence = PALETTE
px.defaults.template = "plotly_white"
st.markdown("""
<style>
 .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
 div[data-testid="stMetric"] {background: rgba(31,78,121,.06); border: 1px solid rgba(31,78,121,.15);
     border-radius: 10px; padding: 10px 14px;}
 div[data-testid="stMetricLabel"] p {font-size: .82rem;}
 .insight {border-left: 4px solid #C55A11; background: rgba(197,90,17,.06); padding: .6rem .9rem; border-radius: 6px;
     margin-bottom: .7rem;}
 .insight b.tag {color:#C55A11; font-size:.78rem; text-transform: uppercase; letter-spacing:.04em;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data loading (cached)
@st.cache_data
def load():
    pq = {p.stem: pd.read_parquet(p) for p in DATA.glob("*.parquet")}
    js = {p.stem: json.loads(p.read_text()) for p in DATA.glob("*.json")}
    return pq, js


pq, js = load()
cube = pq["cube_monthly"]
cube["Year"] = cube.month.dt.year


def fig_style(fig, h=360, legend=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      colorway=PALETTE, showlegend=legend, legend=dict(orientation="h", y=-0.18),
                      font=dict(size=12))
    return fig


def bdt(x, unit="M"):
    if unit == "M":
        return f"৳{x / 1e6:,.0f}M" if x >= 1e8 else f"৳{x / 1e6:,.1f}M"
    return f"৳{x:,.0f}"


# ------------------------------------------------------------------ sidebar: navigation + global filters
st.sidebar.title("🍰 Dan Cake Analytics")
st.sidebar.caption("Intelligent data-analytics & ML framework · 500,000 transactions · 2020–2026")
PAGES = ["Executive overview", "Trends & seasonality", "Products & portfolio", "Markets & customers", "Demand forecast",
         "Machine-learning models", "Anomaly monitor", "Insights & recommendations", "Data quality & method"]
SLUGS = ["overview", "trends", "products", "markets", "forecast", "models", "anomalies", "insights", "quality"]
start = SLUGS.index(st.query_params.get("page")) if st.query_params.get("page") in SLUGS else 0   # deep links: ?page=forecast
page = st.sidebar.radio("Navigate", PAGES, index=start)
st.query_params["page"] = SLUGS[PAGES.index(page)]
st.sidebar.divider()
st.sidebar.subheader("Filters")
years = sorted(cube.Year.unique())
yr = st.sidebar.select_slider("Years", options=years, value=(years[0], years[-1]))
def ms(label, col):
    opts = sorted(cube[col].unique())
    return st.sidebar.multiselect(label, opts, default=[], placeholder="All")
f_region = ms("Region", "Region")
f_channel = ms("Channel", "Channel")
f_cust = ms("Customer type", "Customer_Type")
f_cat = ms("Category", "Category")


def apply(df, years=True):
    m = pd.Series(True, index=df.index)
    if years:
        m &= df.month.dt.year.between(*yr)
    for col, sel in [("Region", f_region), ("Channel", f_channel), ("Customer_Type", f_cust), ("Category", f_cat)]:
        if sel and col in df:
            m &= df[col].isin(sel)
    return df[m]


cf = apply(cube)
active = [s for s in [f_region, f_channel, f_cust, f_cat] if s]
st.sidebar.caption(f"{'Filtered view' if active else 'All data'} · {cf.orders.sum():,} orders")


# ================================================================== pages
def kpi_row(cf):
    last_year = cf.Year.max()
    cur = cf[cf.Year == last_year]
    last_m = cur.month.max().month if len(cur) else 12
    prev = cube_prev = apply(cube, years=False)
    prev = prev[(prev.Year == last_year - 1) & (prev.month.dt.month <= last_m)]
    g = (cur.net.sum() / prev.net.sum() - 1) * 100 if prev.net.sum() else np.nan
    c = st.columns(6)
    c[0].metric("Net sales", bdt(cf.net.sum()))
    c[1].metric("Units sold", f"{cf.units.sum() / 1e6:,.2f}M")
    c[2].metric("Orders", f"{cf.orders.sum() / 1e3:,.0f}K")
    c[3].metric("Avg order value", bdt(cf.net.sum() / max(cf.orders.sum(), 1), "x"))
    c[4].metric("Discount rate", f"{cf.discount.sum() / max(cf.gross.sum(), 1) * 100:.2f}%")
    c[5].metric("YTD growth", f"{g:+.0f}%" if pd.notna(g) else "–",
                help=f"Jan–{MONTHS[last_m - 1]} {last_year} vs the same months of {last_year - 1}")


if page == "Executive overview":
    st.title("Executive overview")
    st.caption("What happened? Headline KPIs for the selected period and filters.")
    kpi_row(cf)
    m = cf.groupby("month")[["net", "units"]].sum().reset_index()
    m["12-month average"] = m.net.rolling(12).mean()
    fig = go.Figure()
    fig.add_bar(x=m.month, y=m.net, name="Net sales", marker_color="#9DB4CC")
    fig.add_scatter(x=m.month, y=m["12-month average"], name="12-month average", line=dict(color=PALETTE[1], width=3))
    st.plotly_chart(fig_style(fig.update_layout(title="Monthly net sales (BDT)")), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    cat = cf.groupby("Category").net.sum().sort_values(ascending=False).reset_index()
    pie = px.pie(cat, names="Category", values="net", hole=.55, title="Sales by category")
    pie.update_traces(textposition="inside", textinfo="percent", insidetextorientation="horizontal")
    pie.update_layout(uniformtext_minsize=9, uniformtext_mode="hide", legend=dict(orientation="v", x=1, y=.5, font=dict(size=10)))
    c1.plotly_chart(fig_style(pie, 380).update_layout(legend=dict(orientation="v", x=1.02, y=.5)), use_container_width=True)
    reg = cf.groupby("Region").net.sum().sort_values().reset_index()
    c2.plotly_chart(fig_style(px.bar(reg, x="net", y="Region", orientation="h", title="Sales by region",
                                     color_discrete_sequence=[PALETTE[0]]), 380, False), use_container_width=True)
    ch = cf.groupby(["Year", "Channel"]).net.sum().reset_index()
    c3.plotly_chart(fig_style(px.bar(ch, x="Year", y="net", color="Channel", title="Channel mix by year"), 380), use_container_width=True)
    yr_tab = cf.groupby("Year").agg(orders=("orders", "sum"), units=("units", "sum"), net=("net", "sum"),
                                    discount=("discount", "sum"), gross=("gross", "sum")).reset_index()
    yr_tab["AOV"] = yr_tab.net / yr_tab.orders
    yr_tab["Growth %"] = yr_tab.net.pct_change() * 100
    months_in_year = cf.groupby("Year").month.nunique().reindex(yr_tab.Year).values
    yr_tab.loc[months_in_year < 12, "Growth %"] = np.nan          # a partial year is not comparable
    yr_tab["Discount %"] = yr_tab.discount / yr_tab.gross * 100
    st.subheader("Annual summary")
    st.dataframe(yr_tab[["Year", "orders", "units", "net", "AOV", "Discount %", "Growth %"]].style.format(
        {"orders": "{:,}", "units": "{:,}", "net": "৳ {:,.0f}", "AOV": "{:.1f}", "Discount %": "{:.2f}", "Growth %": "{:+.1f}"}, na_rep="–"),
        use_container_width=True, hide_index=True)
    st.caption("Note: the final year may be partial (data end in September 2026).")

elif page == "Trends & seasonality":
    st.title("Trends & seasonality")
    dim = st.radio("Break down by", ["Category", "Region", "Channel", "Customer_Type"], horizontal=True)
    metric = st.radio("Measure", ["net", "units", "orders"], horizontal=True,
                      format_func={"net": "Net sales", "units": "Units", "orders": "Orders"}.get)
    t = cf.groupby(["month", dim])[metric].sum().reset_index()
    st.plotly_chart(fig_style(px.area(t, x="month", y=metric, color=dim, title=f"Monthly {metric} by {dim}"), 420), use_container_width=True)
    c1, c2 = st.columns(2)
    hm = cf.groupby(["Year", cf.month.dt.month])[metric].sum().unstack()
    hm.columns = [MONTHS[i - 1] for i in hm.columns]
    c1.plotly_chart(fig_style(px.imshow(hm, text_auto=".2s", aspect="auto", color_continuous_scale="Blues",
                                        title="Year × month heat map"), 380, False), use_container_width=True)
    full = hm.dropna()
    idx = (full.div(full.mean(axis=1), axis=0) * 100).mean()
    c2.plotly_chart(fig_style(px.bar(x=idx.index, y=idx.values, title="Seasonal index (average month = 100)",
                                     labels={"x": "", "y": "index"}, color_discrete_sequence=[PALETTE[0]]).add_hline(
        y=100, line_dash="dash"), 380, False), use_container_width=True)
    yoy = cf.groupby(["Year", cf.month.dt.month])[metric].sum().reset_index()
    yoy.columns = ["Year", "Month", metric]
    yoy["Month"] = yoy.Month.map(lambda i: MONTHS[i - 1])
    st.plotly_chart(fig_style(px.line(yoy, x="Month", y=metric, color=yoy.Year.astype(str), markers=True,
                                      title="Year-over-year comparison"), 380), use_container_width=True)
    daily = pq["daily_sales"]
    daily = daily[daily.Date.dt.year.between(*yr)]
    dow = daily.groupby(daily.Date.dt.day_name()).net.mean().reindex(
        ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    st.caption(f"Day-of-week effect is negligible: average daily sales vary by only {dow.std() / dow.mean() * 100:.1f}% "
               "across weekdays (all filters except years ignored for daily data).")

elif page == "Products & portfolio":
    st.title("Products & portfolio")
    pr = pq["product_region_monthly"]
    pr = pr[pr.month.dt.year.between(*yr)]
    if f_region:
        pr = pr[pr.Region.isin(f_region)]
    if f_cat:
        pr = pr[pr.Category.isin(f_cat)]
    p = pr.groupby(["Product_ID", "Product_Name", "Category"]).agg(net=("net", "sum"), units=("units", "sum")).reset_index()
    p = p.sort_values("net", ascending=False)
    p["cum %"] = p.net.cumsum() / p.net.sum() * 100
    p["ABC"] = np.where(p["cum %"] <= 80, "A", np.where(p["cum %"] <= 95, "B", "C"))
    c1, c2 = st.columns([1.4, 1])
    n = c1.slider("Top N products", 5, len(p), min(15, len(p)))
    top = p.head(n)
    c1.plotly_chart(fig_style(px.bar(top[::-1], x="net", y=top.Product_ID[::-1] + " " + top.Product_Name[::-1], orientation="h",
                                     color="Category", title=f"Top {n} products by net sales"), 30 * n + 120),
                    use_container_width=True)
    fig = go.Figure()
    fig.add_bar(x=list(range(1, len(p) + 1)), y=p.net, marker_color=p.ABC.map({"A": PALETTE[0], "B": PALETTE[1], "C": "#aaa"}), name="sales")
    fig.add_scatter(x=list(range(1, len(p) + 1)), y=p["cum %"], yaxis="y2", name="cumulative %", line=dict(color="black"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 101]), title="Pareto / ABC analysis")
    c2.plotly_chart(fig_style(fig, 420), use_container_width=True)
    c2.write(f"**A** products: {(p.ABC == 'A').sum()} · **B**: {(p.ABC == 'B').sum()} · **C**: {(p.ABC == 'C').sum()}")
    seg = pq["product_segments"]
    st.subheader("Portfolio segments (K-Means clustering)")
    fig = px.scatter(seg, x="avg_price", y="net_sales", size="units_per_order", color="segment", hover_name="Product_Name",
                     log_x=True, log_y=True, title="Price vs revenue, coloured by machine-learned segment",
                     labels={"avg_price": "average selling price (BDT, log)", "net_sales": "net sales 2020–26 (BDT, log)"})
    st.plotly_chart(fig_style(fig, 450), use_container_width=True)
    st.dataframe(pd.DataFrame(js["clustering"]["product_segments"])[["segment", "Products", "Share", "Price", "Growth", "Dec", "Examples"]]
                 .rename(columns={"Share": "Sales share %", "Price": "Avg price", "Growth": "Unit growth %/yr", "Dec": "December index"})
                 .style.format({"Sales share %": "{:.1f}", "Avg price": "{:.0f}", "Unit growth %/yr": "{:+.1f}", "December index": "{:.0f}"}),
                 hide_index=True, use_container_width=True)
    st.subheader("Product drill-down")
    choice = st.selectbox("Product", p.Product_ID + " – " + p.Product_Name)
    pid = choice.split(" – ")[0]
    one = pr[pr.Product_ID == pid].groupby("month")[["units", "net"]].sum().reset_index()
    st.plotly_chart(fig_style(px.line(one, x="month", y="units", markers=True, title=f"{choice}: monthly units"), 320, False),
                    use_container_width=True)

elif page == "Markets & customers":
    st.title("Markets & customers")
    c1, c2 = st.columns(2)
    rc = cf.pivot_table(index="Region", columns="Channel", values="net", aggfunc="sum")
    rc = rc.loc[rc.sum(axis=1).sort_values(ascending=False).index]
    c1.plotly_chart(fig_style(px.imshow(rc / 1e6, text_auto=".1f", aspect="auto", color_continuous_scale="YlOrBr",
                                        title="Net sales by region × channel (BDT M)"), 460, False), use_container_width=True)
    q = cf.groupby(["Region", "Channel"]).agg(units=("units", "sum"), orders=("orders", "sum")).reset_index()
    q["units/order"] = q.units / q.orders
    qp = q.pivot(index="Region", columns="Channel", values="units/order").loc[rc.index]
    c2.plotly_chart(fig_style(px.imshow(qp, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                                        title="Average units per order"), 460, False), use_container_width=True)
    ct = cf.groupby("Customer_Type").agg(net=("net", "sum"), orders=("orders", "sum"), units=("units", "sum")).reset_index()
    ct["AOV"] = ct.net / ct.orders
    c3, c4 = st.columns(2)
    c3.plotly_chart(fig_style(px.bar(ct.sort_values("net"), x="net", y="Customer_Type", orientation="h",
                                     title="Net sales by customer type", color_discrete_sequence=[PALETTE[2]]), 330, False),
                    use_container_width=True)
    gr = cf.groupby(["Year", "Region"]).net.sum().unstack()
    full_years = [y for y in gr.index if y < cube.Year.max()]
    if len(full_years) >= 2:
        a, b = full_years[0], full_years[-1]
        cagr = ((gr.loc[b] / gr.loc[a]) ** (1 / (b - a)) - 1) * 100
        c4.plotly_chart(fig_style(px.bar(cagr.sort_values(), orientation="h", title=f"Regional sales CAGR {a}–{b} (%)",
                                         color_discrete_sequence=[PALETTE[1]]), 330, False), use_container_width=True)
    st.subheader("Machine-learned market segments")
    ms_ = pq["market_segments"]
    fig = px.scatter(ms_, x="aov", y="growth_23_25", size="net_sales", color="segment",
                     hover_data=["Region", "Channel", "Customer_Type"],
                     labels={"aov": "average order value (BDT)", "growth_23_25": "sales growth 2023–25 (%/yr)"},
                     title="220 region × channel × customer cells")
    st.plotly_chart(fig_style(fig, 450), use_container_width=True)
    st.dataframe(pd.DataFrame(js["clustering"]["market_segments"])[["segment", "Cells", "Share", "AOV", "UPO", "Growth", "Channels", "Regions"]]
                 .rename(columns={"Share": "Sales share %", "UPO": "Units/order", "Growth": "Growth %/yr"})
                 .style.format({"Sales share %": "{:.1f}", "AOV": "{:.0f}", "Units/order": "{:.2f}", "Growth %/yr": "{:+.1f}"}),
                 hide_index=True, use_container_width=True)

elif page == "Demand forecast":
    st.title("Demand forecast")
    fc = js["forecasting"]; fu = fc["future"]
    st.caption("What is likely to happen next? Product-level ML forecasts reconciled to a Holt-Winters company total.")
    c = st.columns(4)
    c[0].metric("Forecast net sales (12 m)", bdt(fu["net_sales"]), f"{fu['growth_sales_pct']:+.1f}% vs last 12 m")
    c[1].metric("80% range", f"{fu['net_sales_lo'] / 1e6:,.1f}–{fu['net_sales_hi'] / 1e6:,.1f}M")
    c[2].metric("Forecast units (12 m)", f"{fu['units'] / 1e6:,.3f}M", f"{fu['growth_units_pct']:+.1f}%")
    best_row = [r for r in fc["one_step"] if r["Model"] == fc["best_model"]][0]
    c[3].metric("Forecast error (WAPE)", f"{best_row['WAPE']:.1f}%", f"{fc['best_vs_seasonal_naive_pct']:.0f}% better than naive",
                help=f"Best one-step model: {fc['best_model']} (test period Jan 2025 – Sep 2026)")
    ft = pq["forecast_future_total"]
    hist = pq["daily_sales"].groupby(pd.Grouper(key="Date", freq="MS")).net.sum().reset_index()
    fig = go.Figure()
    fig.add_scatter(x=hist.Date, y=hist.net, mode="lines", name="actual", line=dict(color="black"))
    fig.add_scatter(x=ft.month, y=ft.net_sales_hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=ft.month, y=ft.net_sales_lo, mode="lines", fill="tonexty", line=dict(width=0), name="80% interval",
                    fillcolor="rgba(197,90,17,.2)")
    fig.add_scatter(x=ft.month, y=ft.net_sales, mode="lines+markers", name="forecast", line=dict(color=PALETTE[1], width=3))
    st.plotly_chart(fig_style(fig.update_layout(title="Monthly net sales: history and 12-month forecast"), 420), use_container_width=True)
    fp = pq["forecast_future_products"]
    c1, c2 = st.columns([1, 1.3])
    catf = fp.groupby("Category").net_sales.sum().sort_values(ascending=False).reset_index()
    c1.plotly_chart(fig_style(px.bar(catf, x="Category", y="net_sales", title="Forecast net sales by category (12 m)",
                                     color_discrete_sequence=[PALETTE[0]]), 380, False), use_container_width=True)
    prods = fp.groupby(["Product_ID", "Product_Name"], as_index=False).net_sales.sum().sort_values("net_sales", ascending=False)
    pick = c2.selectbox("Product forecast", prods.Product_ID + " – " + prods.Product_Name)
    pid = pick.split(" – ")[0]
    pr = pq["product_region_monthly"]; h = pr[pr.Product_ID == pid].groupby("month").units.sum().reset_index()
    tp = pq["forecast_test_predictions"]; tp = tp[tp.Product_ID == pid]
    fig = go.Figure()
    fig.add_scatter(x=h.month, y=h.units, name="actual", line=dict(color="black"))
    fig.add_scatter(x=tp.month, y=tp[f"pred_{fc['best_model']}"], name=f"{fc['best_model']} (test, one-step)", line=dict(color=PALETTE[2], dash="dot"))
    f1 = fp[fp.Product_ID == pid]
    fig.add_scatter(x=f1.month, y=f1.units, name="12-month forecast", line=dict(color=PALETTE[1], width=3))
    c2.plotly_chart(fig_style(fig.update_layout(title=f"{pick}: units"), 380), use_container_width=True)
    st.subheader("Model leaderboard (test period Jan 2025 – Sep 2026)")
    sc = pq["forecast_model_scores"].sort_values("RMSE")
    st.dataframe(sc[["Model", "kind", "MAE", "RMSE", "R2", "MAPE", "WAPE", "CV RMSE (log)"]].style.format(
        {"MAE": "{:.1f}", "RMSE": "{:.1f}", "R2": "{:.3f}", "MAPE": "{:.2f}", "WAPE": "{:.2f}", "CV RMSE (log)": "{:.4f}"}),
        hide_index=True, use_container_width=True)
    with st.expander("Download the product forecast"):
        st.download_button("forecast_products.csv", fp.to_csv(index=False), "dan_cake_forecast_products.csv", "text/csv")

elif page == "Machine-learning models":
    st.title("Machine-learning models")
    t1, t2, t3 = st.tabs(["Regression (demand)", "Classification (large orders)", "Clustering"])
    with t1:
        fc = js["forecasting"]
        sc = pd.DataFrame(fc["one_step"]).sort_values("RMSE")
        sc["kind"] = np.where(sc.Model.str.contains("naive", case=False), "baseline", "machine learning")
        st.plotly_chart(fig_style(px.bar(sc, x="RMSE", y="Model", orientation="h", color="kind",
                                         title="Test RMSE – lower is better"), 380), use_container_width=True)
        rc = pd.DataFrame(fc["recursive"])
        st.write("**Recursive 21-month test** (each model uses only its own predictions after Dec 2024)")
        st.dataframe(rc.style.format({c: "{:.2f}" for c in ["MAE", "MSE", "RMSE", "R2", "MAPE", "WAPE"]}), hide_index=True,
                     use_container_width=True)
        imp = pd.Series(fc["permutation_importance"]).sort_values()
        st.plotly_chart(fig_style(px.bar(imp, orientation="h", title=f"Permutation importance – {fc['best_model']}",
                                         color_discrete_sequence=[PALETTE[0]]), 380, False), use_container_width=True)
        st.write("**Tuned hyper-parameters**"); st.json(fc["tuned_params"], expanded=False)
    with t2:
        cl = js["classification"]
        r = pd.DataFrame(cl["results"])
        st.dataframe(r.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]} |
                                    {"Threshold": "{:.2f}", "Lift@10%": "{:.2f}", "Time (s)": "{:.0f}"}),
                     hide_index=True, use_container_width=True)
        st.caption(f"Large orders in test years: {cl['positive_rate_test']:.1%}. Majority-class accuracy would be "
                   f"{cl['majority_baseline_accuracy']:.1%}, so ROC-AUC and F1 are the meaningful measures.")
        imp = pd.Series(cl["xgb_importance"]).sort_values()
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_style(px.bar(imp, orientation="h", title="Drivers of a large order (XGBoost gain)",
                                         color_discrete_sequence=[PALETTE[1]]), 420, False), use_container_width=True)
        lr = pd.Series(cl["large_rate_by_channel"]).mul(100).sort_values()
        c2.plotly_chart(fig_style(px.bar(lr, orientation="h", title="Share of large orders by channel (%)",
                                         color_discrete_sequence=[PALETTE[0]]), 420, False), use_container_width=True)
    with t3:
        k = js["clustering"]
        st.dataframe(pd.DataFrame(k["algorithm_comparison"]).style.format(
            {"Silhouette": "{:.3f}", "Davies-Bouldin": "{:.3f}", "Calinski-Harabasz": "{:.1f}"}), hide_index=True, use_container_width=True)
        st.caption("Silhouette (higher = better separated), Davies–Bouldin (lower = better), Calinski–Harabasz (higher = better).")

elif page == "Anomaly monitor":
    st.title("Anomaly monitor")
    an = pq["daily_anomalies"]; an = an[an.Date.dt.year.between(*yr)]
    fig = go.Figure()
    fig.add_scatter(x=an.Date, y=an.net, name="daily net sales", line=dict(color="#9DB4CC", width=1))
    fig.add_scatter(x=an.Date, y=an.expected, name="expected", line=dict(color=PALETTE[0], width=2))
    fl = an[an.flag == 1]
    fig.add_scatter(x=fl.Date, y=fl.net, mode="markers", name="anomaly", marker=dict(color=PALETTE[1], size=9))
    st.plotly_chart(fig_style(fig.update_layout(title="Daily sales vs expected level"), 420), use_container_width=True)
    tbl = fl.sort_values("robust_z", key=np.abs, ascending=False)[["Date", "net", "expected", "robust_z", "direction", "flag_z", "flag_iso"]].copy()
    tbl["Date"] = tbl.Date.dt.strftime("%d %b %Y (%a)")
    tbl.columns = ["Date", "Net sales", "Expected", "Robust z", "Direction", "Robust-z flag", "Isolation-Forest flag"]
    st.dataframe(tbl.style.format({"Net sales": "{:,.0f}", "Expected": "{:,.0f}", "Robust z": "{:+.2f}"}), hide_index=True, use_container_width=True)
    st.subheader("Most unusual orders (Isolation Forest score)")
    so = pq["suspicious_orders"].copy()
    so["Date"] = so.Date.dt.strftime("%d %b %Y")
    st.dataframe(so.head(100).style.format({"anomaly_score": "{:.3f}", "Net_Sales_BDT": "{:,.2f}"}), hide_index=True, use_container_width=True)
    ad = pd.DataFrame(js["anomaly"]["transaction_detection"])
    st.write("**Detector validation on 1,000 planted errors**")
    st.dataframe(ad.style.format({c: "{:.2f}" for c in ad.columns if c not in ("Method", "Flagged")}), hide_index=True, use_container_width=True)

elif page == "Insights & recommendations":
    st.title("Insights & recommendations")
    st.caption("Generated automatically from the analysis results – they update whenever the pipeline is re-run.")
    for i in js["insights"]:
        st.markdown(f"<div class='insight'><b class='tag'>{i['area']} · {i['type']}</b><br>{i['insight']}<br>"
                    f"<b>➜ {i['recommendation']}</b></div>", unsafe_allow_html=True)

else:
    st.title("Data quality & method")
    q = js["data_quality"]
    c = st.columns(4)
    c[0].metric("Rows", f"{q['rows']:,}"); c[1].metric("Missing cells", q["missing_cells"])
    c[2].metric("Duplicate IDs", q["duplicate_ids"]); c[3].metric("Rule violations", sum(q["rule_violations"].values()))
    st.write(f"Period **{q['date_min']} → {q['date_max']}** · {q['n_products']} products · {q['n_regions']} regions · "
             f"{q['n_channels']} channels · {q['n_customer_types']} customer types.")
    st.dataframe(pd.DataFrame(q["cleaning_log"], columns=["Cleaning step", "Rows affected"]), hide_index=True, use_container_width=True)
    st.subheader("Framework architecture")
    st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EAF1F8", color="#1F4E79", fontname="Helvetica"];
      A[label="Raw CSV\\n500k rows"]; B[label="Validation &\\ncleaning"]; C[label="SQLite\\nstar schema"]; D[label="EDA &\\nstatistics"];
      E[label="Feature\\nengineering"]; F[label="ML models\\nregression · classification\\nclustering · anomaly"]; G[label="Evaluation\\n& selection"];
      H[label="Dashboard"]; I[label="Insights &\\nrecommendations"]; A->B->C->D->E->F->G->H->I; }""")
    st.markdown("Source code, report and documentation: see the project repository README.")
