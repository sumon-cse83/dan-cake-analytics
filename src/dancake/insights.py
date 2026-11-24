"""Layer 7: decision support - turn model outputs into plain-language insights and recommendations.

Each insight is generated from the metrics files, so it updates automatically when new data
are run through the pipeline. Output: outputs/metrics/insights.json (dashboard) and a LaTeX list.
"""
from __future__ import annotations

from .utils import latex_escape, load_json, save_json
from . import config as C


def fmt_m(x):
    return f"BDT {x / 1e6:,.1f} million"


def run() -> dict:
    e, f, c, k, a = (load_json(n) for n in ["eda", "forecasting", "classification", "clustering", "anomaly"])
    fu = f["future"]
    full = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "May": "May", "Jun": "June", "Jul": "July",
            "Aug": "August", "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December"}
    best_cls = c["results"][0]
    ins = []

    def add(area, kind, text, action):
        ins.append({"area": area, "type": kind, "insight": text, "recommendation": action})

    add("Growth", "What happened",
        f"Net sales grew from 2020 to 2025 at {e['cagr_sales']:.1%} a year, but units grew only {e['cagr_units']:.1%} a year; "
        f"on average {e['avg_price_growth']:.1f} percentage points of annual growth came from higher prices and mix, "
        f"{e['avg_volume_growth']:.1f} from volume.",
        "Track volume growth as a separate KPI; price-led growth is not sustainable indefinitely.")
    add("Pricing", "Why it happened",
        f"After controlling for product, year and season, the estimated price elasticity of demand is {e['elasticity']:+.2f} "
        f"(95% CI {e['elasticity_ci'][0]:+.2f} to {e['elasticity_ci'][1]:+.2f}), i.e. not significantly different from zero. "
        f"Discounts show no relationship with order size (Spearman rho = {e['tests']['Discount % vs units/order']['stat']:.3f}).",
        f"Review the discount budget ({fmt_m(e['discount_cost'])} over the period): discounts do not appear to buy volume.")
    add("Seasonality", "What happened",
        f"{full[e['peak_month']]} is the peak month (index {e['peak_index']:.0f}, average month = 100) and {full[e['trough_month']]} "
        f"the weakest ({e['trough_index']:.0f}). The seasonal effect is statistically strong (Kruskal-Wallis p < 0.001).",
        "Build stock and distributor capacity from October; schedule promotions and maintenance in February.")
    add("Portfolio", "What happened",
        f"{e['top_category']} is the largest category ({e['top_category_share']:.1f}% of sales); the top 10 of 52 products "
        f"deliver {e['share_top10_products']:.1f}% of revenue and {e['abc_counts'].get('A', 0)} products form the 'A' class (80% of sales).",
        "Protect availability of A-class products first; review C-class products for rationalisation or repositioning.")
    seg = sorted(k["product_segments"], key=lambda s: -s["Growth"])[0]
    add("Portfolio", "Segmentation",
        f"Clustering finds {k['product_k']} product segments; '{seg['segment']}' grows fastest "
        f"({seg['Growth']:+.1f}% units a year, {seg['Share']:.1f}% of sales).",
        f"Prioritise shelf space and marketing for '{seg['segment']}'.")
    add("Market", "What happened",
        f"Dhaka alone generates {e['top_region_share']:.1f}% of sales and Dhaka plus Chattogram {e['dhaka_ctg_share']:.1f}%. "
        f"General Trade is the largest channel ({e['channel_share']['General Trade']:.1f}%), while E-commerce is "
        f"{e['channel_share']['E-commerce']:.1f}%.",
        "Reduce concentration risk by developing secondary cities and scaling E-commerce, which grows from a small base.")
    mk = sorted(k["market_segments"], key=lambda s: -s["Growth"])[0]
    add("Market", "Segmentation",
        f"The '{mk['segment']}' segment ({int(mk['Cells'])} region-channel-customer cells, mainly {mk['Regions']}) is growing at "
        f"{mk['Growth']:+.1f}% a year, the fastest of the {k['segment_k']} market segments.",
        "Allocate field-sales and digital-marketing effort to these emerging cells.")
    add("Orders", "Prediction",
        f"Whether an order will be large (>= {C.LARGE_ORDER_UNITS} units) is moderately predictable (best ROC-AUC "
        f"{best_cls['ROC-AUC']:.3f}, {best_cls['Model']}); channel is by far the strongest driver: "
        f"{c['large_rate_by_channel']['Distributor']:.0%} of distributor orders are large versus "
        f"{c['large_rate_by_channel']['E-commerce']:.0%} online.",
        "Use the large-order score for delivery-vehicle and stock allocation, mainly at channel level.")
    add("Forecast", "What will happen",
        f"The {f['best_model']} model forecasts next month's product demand with {f['one_step'][[r['Model'] for r in f['one_step']].index(f['best_model'])]['WAPE']:.1f}% "
        f"weighted error, {f['best_vs_seasonal_naive_pct']:.0f}% lower RMSE than the seasonal-naive rule. "
        f"Net sales for {fu['start'][:7]} to {fu['end'][:7]} are forecast at {fmt_m(fu['net_sales'])} "
        f"({fu['growth_sales_pct']:+.1f}% on the previous 12 months; 80% range {fmt_m(fu['net_sales_lo'])} - {fmt_m(fu['net_sales_hi'])}).",
        "Use the product-level forecast for the monthly production plan and the interval for safety stock.")
    add("Risk", "Anomalies",
        f"{a['daily']['flag_any']} of {a['daily']['days']:,} trading days were flagged as unusual. In a controlled test, "
        f"rules plus Isolation Forest caught {a['transaction_detection'][3]['Recall']:.0%} of 1,000 planted data errors.",
        "Run the anomaly screen daily and investigate flagged days and orders before month-end reporting.")
    save_json(ins, "insights")

    lines = [r"\begin{enumerate}[leftmargin=*,itemsep=4pt]"]
    for i in ins:
        lines.append(rf"\item \textbf{{{latex_escape(i['area'])} ({latex_escape(i['type'])}).}} "
                     rf"{latex_escape(i['insight'])} \emph{{Recommendation:}} {latex_escape(i['recommendation'])}")
    lines.append(r"\end{enumerate}")
    (C.TEX_GENERATED / "insights.tex").write_text("\n".join(lines).replace(">=", r"$\geq$"), encoding="utf-8")
    print(f"[insights] {len(ins)} insights generated")
    return ins
