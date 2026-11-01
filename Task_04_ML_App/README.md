# Bank Customer Churn Prediction - End-to-End ML Application

Devixo Solutions AI/ML Internship - **Task 04: End-to-End ML Application with Deployment**

A working machine learning application: a user enters a bank customer's details, clicks **Predict**, and gets a churn
probability with an explanation. The model is trained on a bank-customer dataset of 10,000 rows, compared against
four other models, tuned, saved and served through a multi-page Streamlit app.

**Live app:** _not deployed yet - add the URL here after following "Deploy" below._

**Screenshots:** _add real screenshots of your running app in `reports/screenshots/` and link them here._

## What the app does

| Page | Features |
|---|---|
| **Predict** (main) | Form with 10 inputs, Predict button, verdict, churn probability (% + progress bar), comparison with the average customer, "what drives this prediction" chart, historical churn rates of similar customers, warnings for unusual inputs |
| **Data Visualization** | Filters (country, gender, age) and charts: churn overview, churn rate by group, numeric distributions, correlation |
| **Model Performance** | Model comparison table, before/after tuning, confusion matrices, ROC curves, probability calibration, feature-engineering ablation |
| **Feature Importance** | Permutation importance for the selected model and across all models |
| **Sidebar** | Multiple-model selection (all 5 tuned models) |

## Dataset

- **Name:** Bank Customer Churn Prediction (Kaggle), described as data for "ABC Multistate Bank".
- **Size:** 10,000 rows, 12 columns, target `churn` (1 = customer left). About 20% of customers churn.
- **Provenance:** the Kaggle page does **not** state whether the data is real or simulated, or give a licence. Check the licence
  before redistributing the CSV, and do not describe the data as real customer data unless you confirm it.
- **Data quality (checked):** no missing values, no duplicates, `customer_id` unique (and not used as a feature).

## Workflow (Jupyter notebooks run in this order)

| Notebook | What it does |
|---|---|
| `notebooks/01_data_audit_and_eda.ipynb` | Quality checks, target balance, outliers (IQR), churn by group, correlation; records every decision |
| `notebooks/02_preparation_and_features.ipynb` | Cleaning, stratified 80/20 split **before** fitting, feature engineering, feature ablation test, the preprocessing pipeline |
| `notebooks/03_modeling_tuning_and_saving.ipynb` | 5 baseline models, GridSearchCV / RandomizedSearchCV tuning, before/after comparison, permutation importance, model selection, error analysis, Joblib saving, reload verification |

Reusable code is in `src/` (`data_utils`, `features`, `pipeline`, `evaluation`, `plots`). The Streamlit app is in `app/` and reads only the
saved files in `models/`.

## Results (test set, 2,000 unseen customers, threshold 0.5)

Best model: **Gradient Boosting** (highest cross-validated ROC-AUC after tuning). Always predicting "stays" would give
79.7% accuracy, so accuracy alone is misleading.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC (train) |
|---|---|---|---|---|---|---|
| Gradient Boosting (best) | 0.868 | 0.802 | 0.467 | 0.590 | 0.871 | 0.864 |
| Random Forest | 0.864 | 0.789 | 0.450 | 0.573 | 0.861 | 0.861 |
| KNN | 0.849 | 0.818 | 0.332 | 0.472 | 0.844 | 0.842 |
| Logistic Regression | 0.861 | 0.781 | 0.437 | 0.561 | 0.849 | 0.842 |
| SVM | 0.840 | 0.746 | 0.324 | 0.452 | 0.846 | 0.833 |

**Before vs after tuning (test ROC-AUC):**

| Model | Search | Before | After | Best parameters |
|---|---|---|---|---|
| Gradient Boosting | RandomizedSearchCV | 0.870 | 0.871 | subsample=0.7, n_estimators=100, min_samples_leaf=30, max_depth=4, learning_rate=0.05 |
| Random Forest | RandomizedSearchCV | 0.853 | 0.861 | n_estimators=150, min_samples_leaf=8, max_features=0.5, max_depth=16 |
| KNN | GridSearchCV | 0.788 | 0.844 | n_neighbors=61, p=2, weights=distance |
| Logistic Regression | GridSearchCV | 0.849 | 0.849 | C=30 |
| SVM | RandomizedSearchCV | 0.836 | 0.846 | gamma=0.01, C=1 |

At the F1-optimised threshold (0.29, chosen on training data only) Gradient Boosting reaches recall 0.668,
precision 0.610, F1 0.638. Most influential inputs (permutation importance): `age`, `products_number`, `active_member`.
These describe what the model uses, not what causes churn. In probability bins with at least 50 customers, predicted and observed
churn rates differ by at most 7.4 percentage points (one test set). Full details: `reports/Task04_Bank_Churn_Report.pdf`.

## Technologies

Python 3.12.3, pandas 3.0.2, NumPy 2.4.4, scikit-learn 1.8.0, Matplotlib, Seaborn, Streamlit, Joblib, pytest, Jupyter, Git/GitHub.

## Project structure

```
bank-churn-app/
├── app/                    # Streamlit app (app.py + pages/ + app_utils.py)  - plain .py
├── data/raw/               # original CSV (never edited)
├── data/processed/         # train.csv / test.csv written by notebook 02
├── docs/                   # decisions log, viva notes, test results, EDA output
├── models/                 # saved pipelines, metrics.json, reference.json, error_analysis.json
├── notebooks/              # 01, 02, 03 (.ipynb, executed with outputs)
├── reports/                # PDF report + figures/
├── src/                    # data_utils, features, pipeline, evaluation, plots, build_report
├── tests/                  # automated tests (data, models, notebooks, app)
├── requirements.txt        # pinned versions (needed to load the saved models)
└── README.md
```

## How to run

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

To re-run the notebooks (needed only if you change something or your scikit-learn version differs):

```bash
pip install -r requirements-dev.txt
jupyter lab                                  # open notebooks/ and run 01, 02, 03 in order
# or, without the browser:
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling_tuning_and_saving.ipynb
```

Notebook 03 takes roughly 10-15 minutes (the tuning is the slow part) and rewrites `models/` and `reports/figures/`.

Tests and report:

```bash
python -m pytest                             # needs the trained models
python -m src.build_report --author "Your Name" --date "Month D, YYYY"
```

If loading a model fails with a scikit-learn version error, re-run notebook 03 to retrain with your installed version.

## Deploy (Streamlit Community Cloud, free)

1. Create a **public** GitHub repo and push this project (the `models/` folder must be included).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **New app**.
3. Choose the repo and branch, set **Main file path** to `app/app.py`.
4. In *Advanced settings* pick **Python 3.12** (the version used for training).
5. Deploy, open the URL, test a prediction, then paste the URL into this README.

## Limitations

- Data source/licence not stated; real-vs-simulated unknown. "Churn" has no stated time window.
- Customers with 3-4 products are few, so patterns there are less reliable.
- A single 2,000-row test set is a noisy estimate; probability calibration was checked on that one set.
- Gender and country are model inputs; real banks would need legal/ethical review before using such attributes.
- Educational project: predictions are estimates, not decisions.

## Credits

Dataset: Kaggle "Bank Customer Churn Prediction" (see the Kaggle page for author and licence).
