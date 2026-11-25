"""Layer 4: Exploratory Data Analysis - trends, seasonality, distributions, comparisons and tests.

Every chart is saved to report/figures and every key number to outputs/metrics/eda.json,
so the report and the dashboard always show the same results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from . import config as C
from .utils import plt, save_json, savefig, to_latex_table

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def run(d: pd.DataFrame) -> dict:
    out: dict = {}
    d = d.copy()
    d["ym"] = d.Date.dt.to_period("M").dt.to_timestamp()
    full_years = [y for y in sorted(d.Year.unique()) if d.loc[d.Year == y, "Month"].nunique() == 12]

    # ------------------------------------------------------------ headline KPIs
    out["total_net_sales"] = float(d.Net_Sales_BDT.sum())
    out["total_units"] = int(d.Quantity.sum())
    out["total_orders"] = int(len(d))
    out["avg_order_value"] = float(d.Net_Sales_BDT.mean())
    out["avg_discount_pct"] = float(d["Discount_%"].mean())
    out["discount_cost"] = float(d.Discount_BDT.sum())

    # ------------------------------------------------------------ annual table and growth
    yr = d.groupby("Year").agg(Orders=("Quantity", "size"), Units=("Quantity", "sum"),
                               Net_Sales=("Net_Sales_BDT", "sum"), AOV=("Net_Sales_BDT", "mean"),
                               Price=("Unit_Selling_Price_BDT", "mean"), Discount=("Discount_%", "mean"))
    yr["Months"] = d.groupby("Year").Month.nunique()
    yr["Growth"] = yr.Net_Sales.pct_change() * 100
    yr.loc[yr.Months < 12, "Growth"] = np.nan
    first, last = full_years[0], full_years[-1]
    out["cagr_sales"] = float((yr.Net_Sales[last] / yr.Net_Sales[first]) ** (1 / (last - first)) - 1)
    out["cagr_units"] = float((yr.Units[last] / yr.Units[first]) ** (1 / (last - first)) - 1)
    out["cagr_price"] = float((yr.Price[last] / yr.Price[first]) ** (1 / (last - first)) - 1)
    out["first_full_year"], out["last_full_year"] = int(first), int(last)
    ytd = d[d.Month <= d[d.Year == d.Year.max()].Month.max()]
    ytd_tab = ytd.groupby("Year").Net_Sales_BDT.sum()
    out["ytd_growth_latest"] = float(ytd_tab.iloc[-1] / ytd_tab.iloc[-2] - 1)
    out["latest_year"], out["latest_month"] = int(d.Year.max()), int(d[d.Year == d.Year.max()].Month.max())
    t = yr.reset_index()
    t["Year"] = t.apply(lambda r: f"{int(r.Year)}" + ("*" if r.Months < 12 else ""), axis=1)
    t = t[["Year", "Orders", "Units", "Net_Sales", "AOV", "Price", "Discount", "Growth"]]
    t.columns = ["Year", "Orders", "Units", "Net sales (BDT M)", "AOV (BDT)", "Avg price", "Disc. %", "YoY %"]
    t["Net sales (BDT M)"] /= 1e6
    to_latex_table(t, "annual_summary", "Annual performance summary, 2020--2026", "tab:annual",
                   fmt={"Orders": "{:,.0f}", "Units": "{:,.0f}", "Net sales (BDT M)": "{:,.2f}", "AOV (BDT)": "{:,.1f}",
                        "Avg price": "{:,.1f}", "Disc. %": "{:.2f}", "YoY %": "{:+.1f}"},
                   note="* 2026 covers January--September only, so growth is not computed for it. "
                        "AOV = average order value; price = mean unit selling price.")

    # ------------------------------------------------------------ Fig: monthly trend
    m = d.groupby("ym").agg(net=("Net_Sales_BDT", "sum"), units=("Quantity", "sum"))
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(m.index, m.net / 1e6, color=C.PRIMARY, lw=1.4, label="Monthly net sales")
    ax[0].plot(m.index, (m.net / 1e6).rolling(12).mean(), color=C.ACCENT, lw=2, label="12-month moving average")
    ax[0].set_ylabel("BDT million"); ax[0].set_title("Monthly net sales, Jan 2020 - Sep 2026"); ax[0].legend()
    ax[1].bar(m.index, m.units / 1e3, width=20, color=C.PALETTE[2])
    ax[1].set_ylabel("Units ('000)"); ax[1].set_title("Monthly units sold")
    savefig(fig, "eda_monthly_trend")

    # ------------------------------------------------------------ seasonality (per-day index removes month-length effect)
    dm = d.groupby(["Year", "Month"]).agg(net=("Net_Sales_BDT", "sum"), days=("Date", "nunique")).reset_index()
    dm = dm[dm.Year.isin(full_years)]
    dm["per_day"] = dm.net / dm.days
    dm["index"] = dm.per_day / dm.groupby("Year").per_day.transform("mean") * 100
    seas = dm.groupby("Month")["index"].mean()
    seas_total = dm.assign(i=dm.net / dm.groupby("Year").net.transform("mean") * 100).groupby("Month").i.mean()
    out["seasonal_index_per_day"] = {MONTHS[k - 1]: round(float(v), 1) for k, v in seas.items()}
    out["seasonal_index_total"] = {MONTHS[k - 1]: round(float(v), 1) for k, v in seas_total.items()}
    out["peak_month"] = MONTHS[int(seas_total.idxmax()) - 1]
    out["trough_month"] = MONTHS[int(seas_total.idxmin()) - 1]
    out["peak_index"], out["trough_index"] = float(seas_total.max()), float(seas_total.min())
    # Kruskal-Wallis test: do daily sales differ by calendar month?
    daily = d.groupby("Date").Net_Sales_BDT.sum().rename("net").reset_index()
    daily["month"] = daily.Date.dt.month
    daily = daily[daily.Date.dt.year.isin(full_years)]
    daily["detr"] = daily.net / daily.groupby(daily.Date.dt.year).net.transform("mean")
    kw = stats.kruskal(*[g.detr.values for _, g in daily.groupby("month")])
    out["kw_month_daily_H"], out["kw_month_daily_p"] = float(kw.statistic), float(kw.pvalue)
    heat = dm.pivot(index="Year", columns="Month", values="net") / 1e6
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.2), gridspec_kw={"height_ratios": [1, 1.1]})
    ax[0].bar(MONTHS, seas_total.values, color=[C.ACCENT if v == seas_total.max() else C.PRIMARY for v in seas_total.values])
    ax[0].plot(MONTHS, seas.values, "o-", color=C.PALETTE[2], lw=1.5, label="per-day index")
    ax[0].axhline(100, color="grey", lw=.8, ls="--")
    ax[0].set_ylim(80, max(130, seas_total.max() + 3)); ax[0].set_title("Seasonal index by month (year average = 100)"); ax[0].legend(["per-day index"], loc="upper left")
    im = ax[1].imshow(heat.values, aspect="auto", cmap="Blues")
    ax[1].set_xticks(range(12), MONTHS); ax[1].set_yticks(range(len(heat)), heat.index)
    ax[1].grid(False); ax[1].set_title("Net sales by year and month (BDT million)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax[1].text(j, i, f"{heat.values[i, j]:.1f}", ha="center", va="center", fontsize=8,
                       color="white" if heat.values[i, j] > heat.values.mean() else "black")
    fig.colorbar(im, ax=ax[1], shrink=.9, pad=.01)
    savefig(fig, "eda_seasonality")

    # ------------------------------------------------------------ day of week
    dow = d.groupby(d.Date.dt.dayofweek).agg(net=("Net_Sales_BDT", "sum"))
    ndays = d.groupby(d.Date.dt.dayofweek).Date.nunique()
    dow_avg = dow.net / ndays
    out["dow_cv"] = float(dow_avg.std() / dow_avg.mean())

    # ------------------------------------------------------------ categories & products
    cat = d.groupby(["Category", "Year"], observed=True).Net_Sales_BDT.sum().unstack()
    cat_share = (cat.sum(axis=1) / cat.sum().sum() * 100).sort_values(ascending=False)
    cat_cagr = ((cat[last] / cat[first]) ** (1 / (last - first)) - 1) * 100
    ct = pd.DataFrame({"Category": cat_share.index, "Net sales (BDT M)": cat.sum(axis=1)[cat_share.index].values / 1e6,
                       "Share %": cat_share.values,
                       "Products": d.groupby("Category", observed=True).Product_ID.nunique()[cat_share.index].values,
                       f"CAGR {first}-{str(last)[2:]} %": cat_cagr[cat_share.index].values})
    to_latex_table(ct, "category_summary", "Revenue and growth by product category", "tab:category",
                   fmt={"Net sales (BDT M)": "{:,.2f}", "Share %": "{:.1f}", "Products": "{:.0f}",
                        f"CAGR {first}-{str(last)[2:]} %": "{:+.1f}"},
                   note="CAGR is computed between the first and last complete years. Categories launched after "
                        f"{first} show growth from their launch year base.")
    out["top_category"], out["top_category_share"] = cat_share.index[0], float(cat_share.iloc[0])
    out["fastest_category"] = cat_cagr.replace([np.inf], np.nan).dropna().idxmax()
    out["fastest_category_cagr"] = float(cat_cagr.replace([np.inf], np.nan).dropna().max())

    prod = d.groupby(["Product_ID", "Product_Name"], observed=True).Net_Sales_BDT.sum().sort_values(ascending=False).reset_index()
    prod["cum_share"] = prod.Net_Sales_BDT.cumsum() / prod.Net_Sales_BDT.sum() * 100
    prod["ABC"] = np.where(prod.cum_share <= 80, "A", np.where(prod.cum_share <= 95, "B", "C"))
    abc = prod.ABC.value_counts().to_dict()
    out["abc_counts"] = {k: int(v) for k, v in abc.items()}
    out["share_top10_products"] = float(prod.Net_Sales_BDT.head(10).sum() / prod.Net_Sales_BDT.sum() * 100)
    prod.to_csv(C.TABLES / "product_abc.csv", index=False)
    top = prod.head(10).copy()
    top["Product"] = top.Product_ID.astype(str) + " " + top.Product_Name.astype(str)
    tt = top[["Product", "Net_Sales_BDT", "cum_share", "ABC"]].copy()
    tt.columns = ["Product", "Net sales (BDT M)", "Cumulative share %", "Class"]
    tt["Net sales (BDT M)"] /= 1e6
    to_latex_table(tt, "top_products", "Top ten products by net sales (Pareto/ABC analysis)", "tab:topproducts",
                   fmt={"Net sales (BDT M)": "{:,.2f}", "Cumulative share %": "{:.1f}"})

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    cs = cat_share.sort_values()
    ax[0].barh(cs.index, cs.values, color=C.PRIMARY)
    for i, v in enumerate(cs.values):
        ax[0].text(v + .3, i, f"{v:.1f}%", va="center", fontsize=7.5)
    ax[0].set_title("Share of net sales by category"); ax[0].set_xlim(0, cs.max() * 1.22); ax[0].set_xlabel("% of total")
    ax[1].bar(range(len(prod)), prod.Net_Sales_BDT / 1e6, color=prod.ABC.map({"A": C.PRIMARY, "B": C.ACCENT, "C": "#999999"}))
    ax2 = ax[1].twinx(); ax2.plot(range(len(prod)), prod.cum_share, color="black", lw=1.3); ax2.set_ylim(0, 102)
    ax2.axhline(80, ls="--", color="grey", lw=.8); ax2.grid(False); ax2.set_ylabel("Cumulative %")
    ax[1].set_title(f"Pareto curve (A={abc.get('A', 0)}, B={abc.get('B', 0)}, C={abc.get('C', 0)} products)")
    ax[1].set_xlabel("Products ranked by net sales"); ax[1].set_ylabel("BDT million")
    savefig(fig, "eda_category_pareto")

    # ------------------------------------------------------------ regions, channels, customers
    reg = d.groupby("Region", observed=True).agg(net=("Net_Sales_BDT", "sum"), orders=("Quantity", "size"),
                                                  aov=("Net_Sales_BDT", "mean"), qty=("Quantity", "mean")).sort_values("net", ascending=False)
    reg["share"] = reg.net / reg.net.sum() * 100
    out["top_region"], out["top_region_share"] = reg.index[0], float(reg.share.iloc[0])
    out["dhaka_ctg_share"] = float(reg.share.get("Dhaka", 0) + reg.share.get("Chattogram", 0))
    ch = d.groupby("Channel", observed=True).agg(net=("Net_Sales_BDT", "sum"), orders=("Quantity", "size"),
                                                  aov=("Net_Sales_BDT", "mean"), qty=("Quantity", "mean")).sort_values("net", ascending=False)
    ch["share"] = ch.net / ch.net.sum() * 100
    cu = d.groupby("Customer_Type", observed=True).agg(net=("Net_Sales_BDT", "sum"), orders=("Quantity", "size"),
                                                        aov=("Net_Sales_BDT", "mean"), qty=("Quantity", "mean")).sort_values("net", ascending=False)
    cu["share"] = cu.net / cu.net.sum() * 100
    seg_rows = []
    for name, tab in [("Region", reg), ("Channel", ch), ("Customer type", cu)]:
        for k, r in tab.iterrows():
            seg_rows.append([name, k, r.net / 1e6, r.share, r.orders, r.aov, r.qty])
    st = pd.DataFrame(seg_rows, columns=["Dimension", "Level", "Net sales (BDT M)", "Share %", "Orders", "AOV (BDT)", "Units/order"])
    to_latex_table(st, "segment_summary", "Performance by region, channel and customer type", "tab:segments",
                   fmt={"Net sales (BDT M)": "{:,.2f}", "Share %": "{:.1f}", "Orders": "{:,.0f}", "AOV (BDT)": "{:,.1f}",
                        "Units/order": "{:.2f}"}, col_align="llrrrrr", font=r"\footnotesize")
    out["channel_qty"] = {k: float(v) for k, v in ch.qty.items()}
    out["channel_share"] = {k: float(v) for k, v in ch.share.items()}
    out["customer_aov"] = {k: float(v) for k, v in cu.aov.items()}
    rc = d.pivot_table(index="Region", columns="Channel", values="Net_Sales_BDT", aggfunc="sum", observed=True) / 1e6
    rc = rc.loc[reg.index]
    rq = d.pivot_table(index="Region", columns="Channel", values="Quantity", aggfunc="mean", observed=True).loc[reg.index]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
    short = {"Distributor": "Distrib.", "E-commerce": "E-com.", "General Trade": "General\nTrade",
             "Modern Trade": "Modern\nTrade"}
    for a, data, title, f in [(ax[0], rc, "Net sales (BDT million)", "{:.1f}"),
                              (ax[1], rq, "Mean units per order", "{:.1f}")]:
        im = a.imshow(data.values, cmap="YlOrBr", aspect="auto")
        a.set_xticks(range(data.shape[1]), [short.get(c, c) for c in data.columns])
        a.set_yticks(range(data.shape[0]), data.index); a.grid(False); a.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data.values[i, j]
                a.text(j, i, f.format(v), ha="center", va="center", fontsize=8,
                       color="white" if v > data.values.min() + .65 * (data.values.max() - data.values.min()) else "black")
    savefig(fig, "eda_region_channel")

    # Statistical tests on order size
    kw_ch = stats.kruskal(*[g.values for _, g in d.groupby("Channel", observed=True).Quantity])
    kw_rg = stats.kruskal(*[g.values for _, g in d.groupby("Region", observed=True).Quantity])
    kw_cu = stats.kruskal(*[g.values for _, g in d.groupby("Customer_Type", observed=True).Quantity])
    n = len(d)
    eps = lambda H, k: (H - k + 1) / (n - k)  # epsilon-squared effect size for Kruskal-Wallis
    tests = [("Units/order by channel", "Kruskal-Wallis", kw_ch.statistic, kw_ch.pvalue, eps(kw_ch.statistic, 4)),
             ("Units/order by region", "Kruskal-Wallis", kw_rg.statistic, kw_rg.pvalue, eps(kw_rg.statistic, 11)),
             ("Units/order by customer type", "Kruskal-Wallis", kw_cu.statistic, kw_cu.pvalue, eps(kw_cu.statistic, 5)),
             ("Daily sales by month (detrended)", "Kruskal-Wallis", kw.statistic, kw.pvalue,
              (kw.statistic - 11) / (len(daily) - 12))]
    rho, p = stats.spearmanr(d["Discount_%"], d.Quantity)
    tests.append(("Discount % vs units/order", "Spearman rho", rho, p, rho ** 2))
    large = (d.Quantity >= C.LARGE_ORDER_UNITS).astype(int)
    tab = pd.crosstab(d.Channel, large)
    chi = stats.chi2_contingency(tab)
    cv = np.sqrt(chi.statistic / (n * (min(tab.shape) - 1)))
    tests.append(("Large order vs channel", "Chi-square", chi.statistic, chi.pvalue, cv))
    tt = pd.DataFrame(tests, columns=["Hypothesis (H0: no difference)", "Test", "Statistic", "p-value", "Effect size"])
    tt["p-value"] = tt["p-value"].map(lambda v: "<0.001" if v < .001 else f"{v:.3f}")
    to_latex_table(tt, "stat_tests", "Statistical tests of the main exploratory hypotheses", "tab:tests",
                   fmt={"Statistic": "{:,.2f}", "Effect size": "{:.4f}"}, col_align="llrrr",
                   note=r"Effect sizes: $\varepsilon^2$ for Kruskal--Wallis, $\rho^2$ for Spearman and Cram\'er's $V$ for "
                        r"chi-square. With $n=500{,}000$ almost any difference is `significant', so the effect size "
                        r"decides practical importance.")
    out["tests"] = {r[0]: {"stat": float(r[2]), "p": float(r[3]), "effect": float(r[4])} for r in tests}
    out["large_order_rate"] = float(large.mean())

    # ------------------------------------------------------------ distributions & correlation
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.8))
    ax[0].hist(d.Quantity, bins=np.arange(0.5, 60.5, 1), color=C.PRIMARY)
    ax[0].set_title("Units per order"); ax[0].set_xlabel("units (truncated at 60)")
    ax[1].hist(np.log10(d.Net_Sales_BDT), bins=60, color=C.ACCENT)
    ax[1].set_title("Order value (log10 BDT)"); ax[1].set_xlabel("log10(net sales)")
    ax[2].hist(d["Discount_%"], bins=60, color=C.PALETTE[2]); ax[2].set_title("Discount %")
    savefig(fig, "eda_distributions")
    out["qty_skew"] = float(d.Quantity.skew())
    out["qty_median"] = float(d.Quantity.median())
    out["value_skew"] = float(d.Net_Sales_BDT.skew())
    out["n_outlier_flags"] = int(d.flag_outlier_qty.sum())

    num = ["Quantity", "Unit_Selling_Price_BDT", "List_Price_BDT", "Discount_%", "Net_Sales_BDT", "Product_Launch_Year", "Month"]
    corr = d[num].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(4.4, 3.7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["Units", "Unit price", "List price", "Discount %", "Net sales", "Launch yr", "Month"]
    ax.set_xticks(range(len(num)), labels, rotation=40, ha="right"); ax.set_yticks(range(len(num)), labels); ax.grid(False)
    for i in range(len(num)):
        for j in range(len(num)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    fig.colorbar(im, ax=ax, shrink=.8); ax.set_title("Spearman correlation matrix")
    savefig(fig, "eda_correlation")
    out["corr_qty_price"] = float(corr.loc["Quantity", "Unit_Selling_Price_BDT"])
    out["corr_sales_price"] = float(corr.loc["Net_Sales_BDT", "Unit_Selling_Price_BDT"])
    out["corr_sales_qty"] = float(corr.loc["Net_Sales_BDT", "Quantity"])

    # ------------------------------------------------------------ "why": decompose growth into price and volume
    pv = []
    for y in full_years[1:]:
        a, b = d[d.Year == y - 1], d[d.Year == y]
        s0, s1 = a.Net_Sales_BDT.sum(), b.Net_Sales_BDT.sum()
        q0, q1 = a.Quantity.sum(), b.Quantity.sum()
        p0, p1 = s0 / q0, s1 / q1
        pv.append([y, (s1 / s0 - 1) * 100, (q1 / q0 - 1) * 100, (p1 / p0 - 1) * 100])
    pv = pd.DataFrame(pv, columns=["Year", "Sales growth %", "Volume growth %", "Avg price growth %"])
    to_latex_table(pv, "price_volume", "Decomposition of annual sales growth into volume and price effects",
                   "tab:pricevolume", fmt={"Year": "{:.0f}", "Sales growth %": "{:+.1f}", "Volume growth %": "{:+.1f}",
                                            "Avg price growth %": "{:+.1f}"},
                   note=r"$(1+g_{sales}) = (1+g_{volume})(1+g_{price})$; price is net revenue per unit and therefore "
                        r"also reflects changes in product mix.")
    out["avg_volume_growth"] = float(pv["Volume growth %"].mean())
    out["avg_price_growth"] = float(pv["Avg price growth %"].mean())

    # Price elasticity of demand: log(units) on log(price) with product and calendar-month fixed effects
    pm = d.groupby(["Product_ID", "ym"], observed=True).agg(units=("Quantity", "sum"), price=("Unit_Selling_Price_BDT", "mean")).reset_index()
    pm["year"] = pm.ym.dt.year.astype(str)
    pm["month"] = pm.ym.dt.month.astype(str)
    pm["Product_ID"] = pm.Product_ID.astype(str)
    pm["log_units"], pm["log_price"] = np.log(pm.units), np.log(pm.price)
    # string columns are categorical fixed effects: product (level), year (common growth), month (season).
    # Year effects matter: without them, the parallel rise of prices and volumes gives a spurious positive slope.
    mod = smf.ols("log_units ~ log_price + Product_ID + year + month", data=pm).fit(cov_type="HC1")
    naive = smf.ols("log_units ~ log_price + Product_ID + month", data=pm).fit(cov_type="HC1")
    out["elasticity_naive"] = float(naive.params["log_price"])
    out["elasticity"] = float(mod.params["log_price"])
    out["elasticity_ci"] = [float(x) for x in mod.conf_int().loc["log_price"]]
    out["elasticity_p"] = float(mod.pvalues["log_price"])
    out["elasticity_r2"] = float(mod.rsquared)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax[0].bar(pv.Year - .2, pv["Volume growth %"], .4, label="volume", color=C.PRIMARY)
    ax[0].bar(pv.Year + .2, pv["Avg price growth %"], .4, label="price", color=C.ACCENT)
    ax[0].plot(pv.Year, pv["Sales growth %"], "ko-", label="sales")
    ax[0].set_title("Where does growth come from?"); ax[0].set_ylabel("% change vs previous year"); ax[0].legend()
    lp = d.groupby("ym").agg(price=("List_Price_BDT", "mean"))
    base_units = d.groupby("ym").Quantity.sum()
    ax[1].plot(lp.index, lp.price / lp.price.iloc[:12].mean() * 100, label="list price index", color=C.ACCENT)
    ax[1].plot(base_units.index, base_units.rolling(12).mean() / base_units.iloc[:12].mean() * 100,
               label="units (12-m avg) index", color=C.PRIMARY)
    ax[1].set_title("Price and volume index (2020 = 100)"); ax[1].legend()
    savefig(fig, "eda_price_volume")

    # ------------------------------------------------------------ product launches / lifecycle
    launch = d.groupby("Product_Launch_Year").agg(products=("Product_ID", "nunique"), net=("Net_Sales_BDT", "sum"))
    out["launch_cohorts"] = {int(k): int(v) for k, v in launch.products.items()}

    save_json(out, "eda")
    print(f"[eda] CAGR sales {out['cagr_sales']:.1%}, peak {out['peak_month']}, elasticity {out['elasticity']:.2f}")
    return out
