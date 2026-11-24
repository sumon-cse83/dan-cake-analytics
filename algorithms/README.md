# Machine-learning algorithms – one lesson per file

Each file teaches **one algorithm, step by step**, on the Dan Cake data (proposal section 7.4).
Every step is a `# %%` cell: in VS Code click **Run Cell** above it to run just that step and see the result,
or run the whole file in the terminal.

## Before you start (once)

The lessons use the files made by Step 6, so run that first if `features_train_2020_2024.csv` is missing.

Open the VS Code terminal (**Ctrl + `**) and run:

```powershell
cd F:\dan_cake_project
.\.venv\Scripts\Activate.ps1
python 06_feature_engineering.py
```

## Run the lessons (in this order)

| # | Command | Algorithm | Problem | Time |
|---|---|---|---|---|
| 1 | `python algorithms\01_linear_regression.py` | Linear Regression | A – monthly units | ~5 s |
| 2 | `python algorithms\02_logistic_regression.py` | Logistic Regression | B – large order? | ~5 s |
| 3 | `python algorithms\03_decision_tree.py` | Decision Tree | B (+ A) | ~10 s |
| 4 | `python algorithms\04_random_forest.py` | Random Forest | A (+ B) | ~20 s |
| 5 | `python algorithms\05_gradient_boosting.py` | Gradient Boosting | A (+ B) | ~20 s |
| 6 | `python algorithms\06_xgboost.py` | XGBoost | B (+ A) | ~10 s |
| 7 | `python algorithms\07_svm.py` | Support Vector Machine | B (+ A) | ~70 s |
| 8 | `python algorithms\08_knn.py` | K-Nearest Neighbours | B (+ A) | ~30 s |
| 9 | `python algorithms\09_kmeans_clustering.py` | K-Means | C – product groups | ~10 s |

`common.py` is not a lesson – it loads the data and scores the models for all lessons. Read it first.

## The three problems

- **A. Regression** – how many units will a product sell in a month? (one row per product per month)
- **B. Classification** – will an order be large (10+ units)? (15,000 orders 2020-24 to learn, 15,000 from 2025-26 to test)
- **C. Clustering** – which products sell in a similar way? (one row per product, no answer column)

Every lesson follows the same honest pattern:
**learn on 2020-2023 → choose settings on 2024 → learn again on 2020-2024 → test once on 2025-2026 → compare with a simple baseline.**

## What each lesson shows you

| Lesson | Key ideas you will see |
|---|---|
| 1 Linear | the learned equation, coefficients, residual plot |
| 2 Logistic | sigmoid, probabilities, odds ratios, threshold choice, confusion matrix, ROC curve |
| 3 Decision Tree | readable if/else rules, overfitting vs. depth, tree drawing |
| 4 Random Forest | 1 tree vs. many, out-of-bag score, permutation importance, prediction range |
| 5 Gradient Boosting | boosting written by hand in 6 lines, learning rate vs. number of trees |
| 6 XGBoost | early stopping, regularisation, rare-class weight, gain importance |
| 7 SVM | why scaling is needed, linear vs. RBF kernel, tuning C and gamma, SVR |
| 8 KNN | scaling, choosing k, looking at the actual neighbours of one order |
| 9 K-Means | K-Means written by hand, elbow and silhouette, naming clusters, placing a new product |

## Results (test 2025-2026, from `outputs/summary.json`)

**A. Monthly units per product** (baseline "same as last month": MAE 139.8)

| Algorithm | MAE ↓ | R² ↑ |
|---|---|---|
| XGBoost | 96.9 | 0.699 |
| Random Forest | 97.6 | 0.696 |
| SVM (linear SVR) | 99.8 | 0.698 |
| Linear Regression | 100.8 | 0.692 |
| Gradient Boosting | 102.6 | 0.663 |
| KNN | 102.6 | 0.663 |
| Decision Tree | 108.0 | 0.623 |

**B. Large order (10+ units)** (baseline "always small": ROC-AUC 0.50, F1 0)

| Algorithm | ROC-AUC ↑ | F1 ↑ |
|---|---|---|
| Logistic Regression | 0.718 | 0.459 |
| XGBoost | 0.717 | 0.469 |
| Decision Tree | 0.712 | 0.468 |
| Random Forest | 0.712 | 0.461 |
| SVM | 0.712 | 0.439 |
| Gradient Boosting | 0.693 | 0.445 |
| KNN | 0.690 | 0.440 |

**C. K-Means:** k = 2 (silhouette 0.54) – 48 cake/snack products vs. 4 bread/breakfast products.

**Lesson learned:** all models beat the baseline. On problem A the top four are within ~4 units of
each other, so a simple, explainable model (Linear Regression) is almost as good as XGBoost.
On problem B every model reaches only ~0.72 AUC: the size of one order is mostly random, so
better results need better *data* (e.g. customer history), not a fancier algorithm.

Charts for every lesson are saved in `algorithms/outputs/`.
