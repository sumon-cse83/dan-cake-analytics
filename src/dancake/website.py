"""Build the public project website (docs/index.html) for free hosting on GitHub Pages.

The page is static: data are embedded as JSON and drawn with Plotly.js in the browser,
so it needs no server. It links to the live Streamlit dashboard and the PDF report.

    python -m dancake.website --dashboard-url https://<your-app>.streamlit.app --repo-url https://github.com/<you>/<repo>
"""
from __future__ import annotations

import argparse
import html
import json
import shutil

import pandas as pd

from . import config as C
from .utils import load_json

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dan Cake Analytics</title>
<meta name="description" content="An intelligent data analytics and machine learning framework for predictive business decision-making - Dan Cake Bangladesh case study (500,000 transactions).">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#1b2430;--muted:#5d6b7a;--line:#e3e8ee;--brand:#1F4E79;--accent:#C55A11;--ok:#2E8B57}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1419;--card:#171e26;--ink:#e6edf3;--muted:#9aa7b4;--line:#2a3440;--brand:#6FA8DC;--accent:#F0A060;--ok:#5cc28a}}
:root[data-theme="dark"]{--bg:#0f1419;--card:#171e26;--ink:#e6edf3;--muted:#9aa7b4;--line:#2a3440;--brand:#6FA8DC;--accent:#F0A060;--ok:#5cc28a}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--ink);line-height:1.6}
a{color:var(--brand)}
header{background:linear-gradient(135deg,#1F4E79,#173a5a);color:#fff;padding:56px 16px 44px}
.wrap{max-width:1120px;margin:0 auto;padding:0 16px}
header h1{font-size:clamp(1.6rem,4vw,2.5rem);margin:.2em 0;line-height:1.2}
header p{max-width:760px;color:#d8e4f0;margin:.4em 0 1.4em}
.tag{display:inline-block;font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;background:rgba(255,255,255,.14);padding:4px 10px;border-radius:99px}
.btns a{display:inline-block;margin:6px 8px 0 0;padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600}
.btn1{background:#C55A11;color:#fff}.btn2{background:rgba(255,255,255,.14);color:#fff;border:1px solid rgba(255,255,255,.35)}
nav{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;gap:18px;overflow-x:auto;padding-top:10px;padding-bottom:10px}
nav a{text-decoration:none;color:var(--muted);font-weight:500;white-space:nowrap;font-size:.92rem}
section{padding:40px 0 8px}
h2{font-size:1.45rem;margin:0 0 .3em}.lead{color:var(--muted);margin-top:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:-28px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:0 2px 8px rgba(0,0,0,.04)}
.kpi b{display:block;font-size:1.45rem;color:var(--brand)}.kpi span{font-size:.82rem;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
@media (max-width:520px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}
.chart{width:100%;height:340px}
table{width:100%;border-collapse:collapse;font-size:.88rem}th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}tr.best td{font-weight:700;color:var(--ok)}
.tablewrap{overflow-x:auto}
.ins{border-left:4px solid var(--accent);padding:10px 14px;margin:0 0 12px;background:var(--card);border-radius:8px}
.ins small{color:var(--accent);font-weight:700;letter-spacing:.05em;text-transform:uppercase}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.flow div{background:var(--card);border:1px solid var(--brand);border-radius:8px;padding:8px 12px;font-size:.88rem}
.flow i{color:var(--muted);font-style:normal}
footer{margin-top:48px;padding:28px 0;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}
</style>
</head>
<body>
<header><div class="wrap">
<span class="tag">MSc Project · Data Science · Business Intelligence</span>
<h1>An Intelligent Data Analytics and Machine Learning Framework for Predictive Business Decision-Making</h1>
<p>Case study: __NROWS__ sales transactions of Dan Cake products in Bangladesh, January 2020 – September 2026. The framework answers three management questions: <b>what happened, why it happened, and what is likely to happen next.</b></p>
<div class="btns"><a class="btn1" href="__DASH__">Open live dashboard</a><a class="btn2" href="report.pdf">Read the report (PDF)</a><a class="btn2" href="__REPO__">Source code</a></div>
</div></header>
<div class="wrap"><div class="kpis" id="kpis"></div></div>
<nav><div class="wrap"><a href="#what">What happened</a><a href="#why">Why</a><a href="#next">What next</a><a href="#models">Models</a><a href="#insights">Recommendations</a><a href="#method">Method</a></div></nav>
<main class="wrap">
<section id="what"><h2>What happened</h2><p class="lead">Sales growth, seasonality and the structure of the business.</p>
<div class="grid"><div class="card"><div id="c_trend" class="chart"></div></div><div class="card"><div id="c_season" class="chart"></div></div>
<div class="card"><div id="c_cat" class="chart"></div></div><div class="card"><div id="c_channel" class="chart"></div></div></div></section>
<section id="why"><h2>Why it happened</h2><p class="lead">Growth decomposed into volume and price; drivers of order size.</p>
<div class="grid"><div class="card"><div id="c_pv" class="chart"></div></div><div class="card"><div id="c_drivers" class="chart"></div></div></div></section>
<section id="next"><h2>What is likely to happen next</h2><p class="lead">Twelve-month net-sales outlook with an 80% prediction interval.</p>
<div class="card"><div id="c_fc" class="chart" style="height:380px"></div></div></section>
<section id="models"><h2>Machine-learning model comparison</h2><p class="lead">All models trained on 2020–2024 and tested on unseen data from January 2025 to September 2026.</p>
<div class="grid"><div class="card"><h3>Demand forecasting (regression)</h3><div class="tablewrap" id="t_reg"></div></div>
<div class="card"><h3>Large-order prediction (classification)</h3><div class="tablewrap" id="t_cls"></div></div></div></section>
<section id="insights"><h2>Insights & recommendations</h2><div id="ins"></div></section>
<section id="method"><h2>Framework & method</h2>
<div class="flow"><div>Raw data (CSV)</div><i>→</i><div>Validation & cleaning</div><i>→</i><div>SQLite star schema</div><i>→</i><div>EDA & statistical tests</div><i>→</i><div>Feature engineering</div><i>→</i><div>ML: regression · classification · clustering · anomaly detection</div><i>→</i><div>Evaluation</div><i>→</i><div>Dashboard</div><i>→</i><div>Decision support</div></div>
<p>Python (pandas, scikit-learn, XGBoost, statsmodels), Streamlit and Plotly. The full pipeline runs with one command (<code>python run_pipeline.py</code>) and every number in the report and on this page is generated from it.</p></section>
</main>
<footer><div class="wrap">© __YEAR__ · MSc project report and prototype. Data: Dan Cake Bangladesh transaction dataset (500,000 records) used for academic purposes. Built with Python and Plotly.</div></footer>
<script>
const D = __DATA__;
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const ink = dark ? '#e6edf3' : '#1b2430', grid = dark ? '#2a3440' : '#e3e8ee';
const L = (t, extra={}) => Object.assign({title:{text:t,font:{size:14}},margin:{l:50,r:14,t:42,b:40},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{family:'Inter',color:ink,size:11},xaxis:{gridcolor:grid},yaxis:{gridcolor:grid},legend:{orientation:'h',y:-0.2},colorway:['#1F4E79','#C55A11','#2E8B57','#7F3C8D','#B8860B']}, extra);
const cfg = {displayModeBar:false, responsive:true};
document.getElementById('kpis').innerHTML = D.kpis.map(k => `<div class="kpi"><b>${k[1]}</b><span>${k[0]}</span></div>`).join('');
Plotly.newPlot('c_trend',[{x:D.trend.x,y:D.trend.y,type:'bar',name:'monthly',marker:{color:'#9DB4CC'}},{x:D.trend.x,y:D.trend.ma,name:'12-month average',line:{color:'#C55A11',width:3}}],L('Monthly net sales (BDT million)'),cfg);
Plotly.newPlot('c_season',[{x:D.season.x,y:D.season.y,type:'bar',marker:{color:D.season.y.map(v=>v>110?'#C55A11':'#1F4E79')}}],L('Seasonal index (average month = 100)',{yaxis:{range:[80,130],gridcolor:grid}}),cfg);
Plotly.newPlot('c_cat',[{labels:D.cat.x,values:D.cat.y,type:'pie',hole:.55,textinfo:'percent',textposition:'inside',sort:false}],L('Share of sales by category',{uniformtext:{minsize:10,mode:'hide'},legend:{orientation:'v',x:1,y:.5}}),cfg);
Plotly.newPlot('c_channel',D.channel.series.map(s=>({x:D.channel.x,y:s.y,name:s.name,type:'bar'})),L('Net sales by channel and year (BDT million)',{barmode:'stack'}),cfg);
Plotly.newPlot('c_pv',[{x:D.pv.x,y:D.pv.vol,name:'volume',type:'bar'},{x:D.pv.x,y:D.pv.price,name:'price & mix',type:'bar'},{x:D.pv.x,y:D.pv.sales,name:'sales',mode:'lines+markers',line:{color:ink}}],L('Annual growth: volume vs price (%)'),cfg);
Plotly.newPlot('c_drivers',[{y:D.drivers.x,x:D.drivers.y,type:'bar',orientation:'h',marker:{color:'#C55A11'}}],L('What drives a large order? (share of model gain)',{margin:{l:120,r:14,t:42,b:40}}),cfg);
Plotly.newPlot('c_fc',[{x:D.hist.x,y:D.hist.y,mode:'lines',name:'actual',line:{color:ink}},{x:D.fc.x,y:D.fc.hi,mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'},{x:D.fc.x,y:D.fc.lo,mode:'lines',fill:'tonexty',fillcolor:'rgba(197,90,17,.2)',line:{width:0},name:'80% interval'},{x:D.fc.x,y:D.fc.y,name:'forecast',line:{color:'#C55A11',width:3}}],L('Net sales (BDT million): history and forecast'),cfg);
const tbl=(rows,cols,best)=>'<table><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr>'+rows.map(r=>`<tr class="${r[0]===best?'best':''}">`+r.map(v=>`<td>${v}</td>`).join('')+'</tr>').join('')+'</table>';
document.getElementById('t_reg').innerHTML = tbl(D.reg,['Model','RMSE','WAPE %','R²'],D.reg_best);
document.getElementById('t_cls').innerHTML = tbl(D.cls,['Model','ROC-AUC','F1','Recall'],D.cls_best);
document.getElementById('ins').innerHTML = D.ins.map(i=>`<div class="ins"><small>${i.area} · ${i.type}</small><div>${i.insight}</div><b>→ ${i.recommendation}</b></div>`).join('');
</script>
</body></html>"""


def build(dashboard_url="https://YOUR-APP.streamlit.app", repo_url="https://github.com/sumon-cse83/dan-cake-analytics"):
    e, f, c, ins, q = (load_json(n) for n in ["eda", "forecasting", "classification", "insights", "data_quality"])
    cube = pd.read_parquet(C.DASHBOARD_DATA / "cube_monthly.parquet")
    m = cube.groupby("month").net.sum() / 1e6
    ch = cube.assign(Year=cube.month.dt.year).groupby(["Year", "Channel"]).net.sum().unstack() / 1e6
    cat = cube.groupby("Category").net.sum().sort_values(ascending=False)
    pv = pd.read_csv(C.TABLES / "price_volume.csv")
    ft = pd.read_parquet(C.DASHBOARD_DATA / "forecast_future_total.parquet")
    hist = m[-36:]
    one = sorted([r for r in f["one_step"]], key=lambda r: r["RMSE"])
    nice = {"chan_region_rate": "Channel x region history", "product_rate": "Product history", "is_december": "December",
            "Customer_Type": "Customer type", "product_age": "Product age", "month_cos": "Season (month)",
            "month_sin": "Season (month) ", "price_vs_list": "Price vs list", "log_price": "Price", "Discount_%": "Discount"}
    drivers = pd.Series(c["xgb_importance"]).rename(index=lambda k: nice.get(k, k)).sort_values().tail(8)
    fu = f["future"]
    data = {
        "kpis": [["Net sales 2020–26", f"৳{e['total_net_sales'] / 1e6:,.0f}M"], ["Orders analysed", f"{e['total_orders']:,}"],
                 ["Sales CAGR 2020–25", f"{e['cagr_sales']:.1%}"], ["December peak", f"+{e['peak_index'] - 100:.0f}%"],
                 ["Forecast error (WAPE)", f"{[r for r in one if r['Model'] == f['best_model']][0]['WAPE']:.1f}%"],
                 ["Next-12-month outlook", f"৳{fu['net_sales'] / 1e6:,.1f}M"]],
        "trend": {"x": [str(i.date()) for i in m.index], "y": m.round(3).tolist(), "ma": m.rolling(12).mean().round(3).where(m.rolling(12).mean().notna(), None).tolist()},
        "season": {"x": list(e["seasonal_index_total"].keys()), "y": list(e["seasonal_index_total"].values())},
        "cat": {"x": cat.index.tolist(), "y": (cat / 1e6).round(2).tolist()},
        "channel": {"x": ch.index.tolist(), "series": [{"name": k, "y": ch[k].round(2).tolist()} for k in ch.columns]},
        "pv": {"x": pv.Year.tolist(), "vol": pv["Volume growth %"].round(1).tolist(), "price": pv["Avg price growth %"].round(1).tolist(),
               "sales": pv["Sales growth %"].round(1).tolist()},
        "drivers": {"x": drivers.index.tolist(), "y": drivers.round(3).tolist()},
        "hist": {"x": [str(i.date()) for i in hist.index], "y": hist.round(3).tolist()},
        "fc": {"x": [str(i.date()) for i in ft.month], "y": (ft.net_sales / 1e6).round(3).tolist(),
               "lo": (ft.net_sales_lo / 1e6).round(3).tolist(), "hi": (ft.net_sales_hi / 1e6).round(3).tolist()},
        "reg": [[r["Model"], f"{r['RMSE']:.1f}", f"{r['WAPE']:.2f}", f"{r['R2']:.3f}"] for r in one],
        "reg_best": f["best_model"],
        "cls": [[r["Model"], f"{r['ROC-AUC']:.3f}", f"{r['F1']:.3f}", f"{r['Recall']:.3f}"] for r in c["results"]],
        "cls_best": c["best_model"],
        "ins": [{k: html.escape(v) for k, v in i.items()} for i in ins],
    }
    page = (TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__DASH__", dashboard_url)
            .replace("__REPO__", repo_url).replace("__NROWS__", f"{q['rows']:,}").replace("__YEAR__", "2026"))
    (C.SITE / "index.html").write_text(page, encoding="utf-8")
    pdf = C.REPORT / "main.pdf"
    if pdf.exists():
        shutil.copy(pdf, C.SITE / "report.pdf")
    (C.SITE / ".nojekyll").write_text("")
    print(f"[website] docs/index.html written ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard-url", default="https://YOUR-APP.streamlit.app")
    ap.add_argument("--repo-url", default="https://github.com/sumon-cse83/dan-cake-analytics")
    a = ap.parse_args()
    build(a.dashboard_url, a.repo_url)
