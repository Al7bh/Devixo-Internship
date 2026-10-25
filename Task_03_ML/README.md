# Used-Car Price Prediction: End-to-End Machine Learning Project

**Devixo Solutions AI/ML Internship, Task 03.** A complete regression project: raw data to a saved, tested model and a Streamlit prediction app.

**Goal:** predict the selling price of a used car from its age, kilometres driven, fuel, transmission, owner history, engine, power, mileage and brand.

| Headline result (held-out test set, 1,386 cars) | |
|---|---|
| Selected model | XGBoost, tuned (a tie with Gradient Boosting, see *Model selection*) |
| Median error | **12.3%** of the price |
| Typical error in rupees (MAE) | about 71,000 |
| RMSLE (relative error) | 0.215, vs 0.749 for "always guess the median price" |
| R2 | 0.93 (see the caution in *Results*) |

---

## 1. Dataset

- **Source:** Kaggle, *Vehicle dataset from CarDekho* (user `nehalbirla`), file `Car details v3.csv`: <https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho>. Check the licence on that page before reusing the data. A copy is included in `data/raw/car_details_v3.csv`.
- **Size:** 8,128 rows x 13 columns. After removing 1,202 exact duplicates: **6,926 rows** (task minimum: 3,000).
- **Target:** `selling_price` (assumed Indian rupees; the file does not state the currency). Range 29,999 to 10,000,000, median 400,000, heavily right-skewed (skewness 5.57; -0.16 after a log transform).
- **Features:** name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque, seats.

**Data-quality findings (from `notebooks/01_eda_and_data_audit.ipynb`)**

| Issue | What we found | What we did |
|---|---|---|
| Duplicates | 1,202 exact duplicate rows (14.8%) | Removed before the train/test split. *Assumption:* identical rows are repeated listings (the file has no listing ID) |
| Missing values | `mileage`, `engine`, `max_power`, `seats`, `torque`: about 3% each, missing together | Median imputation inside the pipeline, fitted on training data only |
| Text numbers | "23.4 kmpl", "1248 CC", "74 bhp" | Parsed to numbers |
| Invalid values | 15 rows with mileage 0; one blank " bhp" | Treated as missing |
| Mixed units | Mileage is km/kg for the 87 CNG/LPG cars, kmpl otherwise | Kept as numbers; `fuel` identifies the unit |
| Messy text | `torque` has mixed units (Nm, kgm, rpm ranges) | Not used |
| Suspicious values | 2,360,457 km on a 2007 car | Kept (log transform limits their effect); only 2 such rows |

## 2. Workflow

```
raw CSV -> remove duplicates -> 80/20 train/test split (seed 42) -> [everything below uses TRAIN only]
  -> feature engineering -> feature selection (experiments) -> scaling/encoding
  -> 5 models + 2 reference baselines, 5-fold CV -> randomized hyperparameter search
  -> pick best by CV -> evaluate ONCE on test -> error analysis -> feature importance
  -> save (Joblib + Pickle) -> reload checks -> Streamlit app
```

**One saved pipeline does everything.** The saved model accepts a *raw* car description and returns a price. It contains: feature engineering, median imputation, scaling, one-hot encoding, the model, and the log-price conversion. Because only training data is used to fit any step, there is no train/test leakage.

**Feature engineering:** `car_age` (2020 minus year), `brand` (from the name; "Land" fixed to "Land Rover"), `log_km_driven`, and the parsed numbers `mileage`, `engine`, `max_power`.

**Feature selection (experiments, not guesses; `notebooks/02_...`).** Final set of 9 features: `car_age`, `log_km_driven`, `mileage`, `engine`, `max_power`, `brand`, `fuel`, `transmission`, `owner`.
- Adding `brand` lowered the cross-validated error from 0.2408 to 0.2301.
- Removing one feature at a time showed `car_age` matters most (+0.0765 error when removed), then `max_power`, `brand`, `mileage`.
- `seats` and `seller_type` made no measurable difference (+0.0003 and +0.0002), also confirmed jointly on Random Forest, Gradient Boosting and SVR, so they were dropped.
- `km_per_year` and `bhp_per_litre` were tested and gave no benefit, so they are not used.

## 3. Model selection

Models trained (task sheet: at least four): **Decision Tree, Random Forest, Support Vector Machine (SVR), Gradient Boosting**, plus **XGBoost (optional in the task)**. We added two reference points: a Ridge regression and an "always predict the median" baseline.

**Metric:** RMSLE (error on the log-price scale, roughly a typical percentage error). Prices span 30 thousand to 10 million, so plain RMSE/MAE would be dominated by a few luxury cars.

**Tuning:** `RandomizedSearchCV`, 15 combinations per model, 5-fold cross-validation, training set only. **Selection rule (fixed in advance):** lowest tuned CV RMSLE.

| Model | CV RMSLE default | CV RMSLE tuned | Test R2 | Test MAE | Test RMSLE |
|---|---|---|---|---|---|
| **XGBoost** | 0.2284 | **0.2178** | 0.9275 | 71,361 | 0.2153 |
| Gradient Boosting | 0.2342 | 0.2209 | 0.9401 | 69,992 | 0.2164 |
| Random Forest | 0.2300 | 0.2227 | 0.9225 | 72,025 | 0.2250 |
| SVR | 0.2295 | 0.2299 | 0.8544 | 76,556 | 0.2279 |
| Decision Tree | 0.2990 | 0.2623 | 0.8987 | 84,069 | 0.2547 |
| Median baseline | 0.7678 | - | -0.0460 | 259,547 | 0.7488 |

**Honest reading:** XGBoost and Gradient Boosting are **effectively tied** (CV gap 0.0031, smaller than the 0.0113 fold-to-fold noise). The pre-set rule picked XGBoost. Changing the choice after seeing test scores would be selecting a model using the test set, so we did not. All tree-based models clearly beat Ridge regression (0.2626) and the baseline.

## 4. Results

- **Test set (never used for any decision before the final report):** median error 12.3%, MAE 71,361 rupees, RMSLE 0.2153, R2 0.9275.
- **Caution on R2:** cross-validation R2 was 0.88, lower than the test R2 of 0.93. The test set happens to contain only 6 cars above 30 lakh (the training set has 43), and R2 is very sensitive to expensive cars. RMSLE (0.2178 in CV vs 0.2153 on test) is the steadier measure.
- **Where it works and fails:** median error is 9% to 12% for cars priced 2 to 10 lakh, but 19% for cars under 2 lakh (it tends to over-price them) and the model under-prices cars above 10 lakh by a median 8%. The largest errors are cheap old cars.
- **Feature importance** (permutation, test set): car age (0.38) and max power (0.27) dominate, then brand (0.07), fuel (0.05) and engine (0.04). This shows what the model *uses*, not what *causes* prices.
- **Prediction range in the app:** predicted price x 0.78 to x 1.27, the middle 80% of out-of-fold errors; 79.9% of test cars fell inside.
- Figures are in `reports/figures/`; tables in `reports/`.

## 5. Limitations

- **Old market data:** the newest car is from 2020 and prices are listing prices from that era. The model is **not valid for current prices**.
- **Weak on very cheap old cars and luxury cars** (see Results); only 80 test cars are above 10 lakh.
- **Rare categories** (Test Drive Car owners, LPG/CNG, 11 rare brands) have too few rows to learn from; they are grouped as "infrequent".
- **Duplicate removal is an assumption**, and near-duplicate listings (same model, similar details) may still sit on both sides of the split, which could make scores slightly optimistic.
- **Random split, not a time split:** it does not test how the model behaves on future years.
- **`torque`, `seats`, `seller_type` were not used**; `torque` might help if parsed carefully.
- **Tuning was modest** (15 random combinations per model); SVR's best value was at the edge of its search range.
- Currency is assumed to be rupees.
- No screenshot of the app is included; the app is verified by automated tests (see below).

## 6. Future improvements

- Parse `torque` into Nm and test it; extract the car *model* (not only the brand) and test it.
- Use a time-aware split and newer data; add location and condition features if available.
- Wider or Bayesian hyperparameter search; try quantile models for a principled price range.
- Group duplicates by car model for cross-validation to reduce near-duplicate leakage.
- Deploy the app (for example on Streamlit Community Cloud) and add monitoring of input ranges.

## 7. Repository structure

```
.
├── app.py                         Streamlit prediction app (bonus)
├── data/raw/car_details_v3.csv    original data
├── data/processed/                train.csv / test.csv (the exact split used)
├── notebooks/
│   ├── 01_eda_and_data_audit.ipynb
│   ├── 02_feature_engineering_and_selection.ipynb
│   └── 03_modeling_tuning_and_evaluation.ipynb
├── src/
│   ├── car_features.py            feature-engineering transformer
│   └── modeling.py                pipeline builder, metrics, final feature lists
├── models/                        best_model.joblib, best_model.pkl, model_metadata.json
├── reports/                       figures and result tables (CSV)
├── tests/                         30 automated tests
├── report/
│   ├── Task_03_Report.pdf          18-page project report (same format as the Task 02 report)
│   └── build_report_task3.py       rebuilds the report from the notebook outputs
├── docs/mentor_viva_guide.md      questions and answers about this project
└── requirements.txt
```

## 8. How to run

Requires **Python 3.11 or newer** (built and tested with 3.12). Package versions are pinned because saved models must be loaded with the scikit-learn version they were saved with.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook                    # run notebooks 01, 02, 03 in order
pytest                              # 30 tests
streamlit run app.py                # prediction app
python report/build_report_task3.py # rebuild report/Task_03_Report.pdf from the notebook outputs
```

Notebook 03 runs the hyperparameter search; it took roughly 15 minutes on a single-CPU machine (faster on more cores). The trained model files are already included, so the app and tests run without retraining.

## 9. Task requirement checklist

| Task 03 requirement | Status | Where |
|---|---|---|
| Kaggle dataset, at least 3,000 rows | Done: 6,926 rows after cleaning | Notebook 01 |
| Import, missing values, duplicates | Done | Notebooks 01-02 |
| Feature engineering, feature selection, scaling | Done, with experiments | Notebook 02 |
| At least four models (RF, SVM, Gradient Boosting, Decision Tree) | Done, plus XGBoost (optional) | Notebook 03 |
| Hyperparameter tuning, cross-validation, feature importance | Done | Notebook 03 |
| Save the best model with Joblib and Pickle | Done, reload-verified | `models/` |
| Documentation: dataset, workflow, model selection, results, limitations, future improvements | Done | this README and `report/Task_03_Report.pdf` |
| Bonus: Streamlit interface | Done, covered by automated tests | `app.py` |

## 10. Quality checks

`pytest` runs 30 tests: parsing and brand extraction, missing-column errors, no train/test overlap, size requirement, Joblib and Pickle giving identical predictions, a regression guard on test quality, handling of unseen brands and missing specs, a sanity check that a newer low-mileage car is priced above an older one, and headless tests of the Streamlit app.
