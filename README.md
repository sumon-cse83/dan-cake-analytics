# Dan Cake Analytics — An Intelligent Data Analytics & Machine Learning Framework

MSc project: **"An Intelligent Data Analytics and Machine Learning Framework for Predictive Business
Decision-Making"**, demonstrated on 500,000 Dan Cake Bangladesh sales transactions (Jan 2020 – Sep 2026).

The framework answers three management questions in one reproducible system:

| Question | How it is answered |
|---|---|
| **What happened?** | validated KPIs, trends, seasonality, Pareto/ABC, region × channel analysis |
| **Why did it happen?** | price–volume decomposition, fixed-effects price elasticity, hypothesis tests, feature importance / SHAP, clustering |
| **What will happen next?** | product-level demand forecasts (7 ML models + baselines + Holt-Winters/SARIMA), large-order prediction, anomaly detection |

## What is in the box

```
run_pipeline.py            ONE command: raw CSV -> cleaned data -> models -> dashboard data -> report tables
src/dancake/
  config.py                all paths, constants, train/test dates  (edit this for a new project)
  data.py                  ingest, validate, clean, SQLite star schema
  eda.py                   exploratory analysis, statistical tests, elasticity, figures
  features.py              leakage-safe feature engineering
  forecasting.py           regression: demand forecasting, tuning, recursive test, 12-month outlook
  classification.py        classification: large-order prediction (7 algorithms)
  clustering.py            K-Means / Ward / GMM segmentation of products and markets
  anomaly.py               daily + transaction anomaly detection (robust z, Isolation Forest)
  insights.py              automatic insights & recommendations
  export.py                dashboard data + LaTeX numbers/tables
  website.py               static project website (docs/index.html)
dashboard/app.py           Streamlit dashboard (9 pages)
report/                    LaTeX MSc report (main.tex, chapters/, references.bib, generated figures & tables)
docs/                      project website for GitHub Pages
sql/kpi_queries.sql        example SQL on the warehouse (usable from Power BI)
tests/                     automated tests (pytest)
guides/                    step-by-step guides: deployment, LaTeX, reusing the framework
01_…06_*.py                original hands-on preprocessing practice scripts
```

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py              # about 20 minutes; prints progress per step
python -m pytest tests -q           # 9 tests
streamlit run dashboard/app.py      # opens http://localhost:8501
```

Run only some steps: `python run_pipeline.py --steps eda forecast insights export`

## Key results (generated — see `outputs/metrics/*.json`)

* Net sales CAGR 2020–2025 **16.6 %**, but volume only **5.0 %** → growth is mainly price and mix.
* Price elasticity ≈ 0 (not significant) and discounts do not change order size.
* December peak (+25 % vs average month), February trough.
* Dhaka = 44 % of sales; top 10 of 52 products = 38 %.
* Best monthly demand forecast cuts the error of "same as last year" planning by ~24 %;
  Holt-Winters forecasts company units with ~1.3 % MAPE.
* Large orders: ROC-AUC ≈ 0.71 for all models — channel is the dominant driver.

## Free hosting

See [guides/DEPLOYMENT.md](guides/DEPLOYMENT.md): dashboard on **Streamlit Community Cloud**
(`https://<name>.streamlit.app`) and website on **GitHub Pages** (`https://<user>.github.io/<repo>`).

## Report

See [guides/LATEX.md](guides/LATEX.md). Upload the `report/` folder (plus `sql/`) to Overleaf, or compile locally
with `pdflatex → bibtex → pdflatex ×2`. Edit your name, ID, supervisor and university at the top of `report/main.tex`.

## Reusing for a future project

See [guides/REUSE_FRAMEWORK.md](guides/REUSE_FRAMEWORK.md).
