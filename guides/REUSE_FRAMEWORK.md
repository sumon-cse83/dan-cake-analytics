# How to reuse this framework for your next project

This project is a template. A future project in inventory, finance, HR, customer or IT-service analytics
follows the **same 10 phases**. Only the data-specific parts change.

## The recipe

| Phase | File | What to do in a new project |
|---|---|---|
| 1. Problem definition | `report/chapters/01_introduction.tex` | Turn the business question into 2–4 analytical problems (forecast? classify? segment? detect?) |
| 2. Configuration | `src/dancake/config.py` | Set the data path, column names, domain rules, train/test dates |
| 3. Data validation & cleaning | `src/dancake/data.py` | Keep the order: placeholders → duplicates → essential fields → impossible values → derive → impute → flag outliers. Change the identities (e.g. `net = gross − discount`) to those of your data |
| 4. Storage | `data.py: build_warehouse` | One fact table (events) + dimension tables (who/what/when/where) |
| 5. EDA | `src/dancake/eda.py` | Trend, seasonality, Pareto, segment comparisons, correlation, tests **with effect sizes** |
| 6. Features | `src/dancake/features.py` | Use only information known *before* the prediction moment. Shift before rolling. Fit encoders/scalers on training data only |
| 7. Models | `forecasting.py`, `classification.py`, `clustering.py`, `anomaly.py` | Always include a naive baseline. Split by time. Tune on validation years, test once |
| 8. Explain | permutation importance, SHAP, simple regressions | Answer *why*, not just *how accurate* |
| 9. Present | `dashboard/app.py`, `website.py` | KPIs on top, global filters, intervals on forecasts, plain-language insights |
| 10. Document | `report/` | Numbers and tables are generated (`export.py`); text explains them |

## Ten rules learned in this project

1. **Look before you clean.** Validate first and report; change nothing until you know what is wrong.
2. **Derive before you impute.** Exact relationships between columns beat statistical guesses.
3. **Flag, don't delete, unusual values.** A bulk order is an outlier but not an error.
4. **Split by time** for anything that predicts the future. Random splits leak the future.
5. **Beat the naive baseline** ("same as last month / last year") or the model has no value.
6. **Simple models first.** Linear and logistic regression were as good as XGBoost here.
7. **Effect sizes, not just p-values.** With 500,000 rows everything is "significant".
8. **Beware confounding.** Price and sales both rose; only year fixed effects revealed elasticity ≈ 0.
9. **Generate every number from code** so report, dashboard and website always agree.
10. **Show uncertainty.** A forecast without an interval invites over-confidence.

## Adapting to a new dataset: minimum changes

1. Copy the project folder and replace the CSV.
2. Edit `config.py` (paths, `CATEGORICAL`, `NUMERIC`, `DOMAIN_RULES`, `TRAIN_END`, `TEST_START`).
3. In `data.py` update the identity rules in `validate()` and `clean()`.
4. In `features.py` change the entity (product → store, employee, ticket…) and the target column.
5. Run `python -m pytest tests -q`, then `python run_pipeline.py`, then fix any chart labels.
