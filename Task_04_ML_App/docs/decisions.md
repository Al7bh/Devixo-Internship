# Decision log

| # | Decision | Reason / evidence | Outcome |
|---|---|---|---|
| 1 | Project: Customer churn (classification) | Supports probability, ROC-AUC, confusion matrix, feature importance bonuses | Done |
| 2 | Rejected the 2-file "customer_churn_dataset" archive | Training file had groups with exactly 100% churn and the test file behaved very differently (looked rule-generated and inconsistent) | Not used |
| 3 | Used Bank Customer Churn Prediction | One consistent file, subtle signal, realistic scores | Done |
| 4 | Source stated honestly | Kaggle page gives no origin or licence | In README and report |
| 5 | Drop `customer_id` | Identifier, listed as unused on the dataset page | Done |
| 6 | Split before fitting; everything inside a Pipeline | Prevents data leakage | Done, tested |
| 7 | Judge by ROC-AUC / recall / F1, not accuracy | 79.7% accuracy from always predicting "stays" | Done |
| 8 | Missing values / duplicates | None found; steps still implemented and unit-tested | Done |
| 9 | Keep outliers | Ages up to 92 and credit scores from 350 are plausible; old customers carry real information | Done |
| 10 | 4 engineered features, tested by ablation | Ablation (CV, train only) in `models/feature_ablation.json` | Kept all four; balance features add little on their own |
| 11 | No class weights; decision threshold tuned instead | Keeps probabilities closer to real frequencies; threshold chosen from out-of-fold training predictions | Done |
| 12 | Best model chosen on CV score, not test score | Avoids picking the winner using the test set | Gradient Boosting |
| 13 | SVM tuned with `probability=False`, final model refit with `probability=True` | Probability estimation is ~5x slower inside a search | Done |
| 14 | Permutation importance on raw input columns | Works for every model type; easy to explain | Done |
| 15 | Pin library versions | Saved models only load reliably with the same scikit-learn | `requirements.txt` |
| 16 | Training code lives in notebooks, app stays in .py | Notebooks show outputs step by step; the app must be a script | notebooks/01-03 |
| 17 | Calibration and per-group error analysis added | Tests the claim that probabilities can be trusted instead of assuming it | `models/error_analysis.json` |
| 18 | Bugs found while testing | Deprecated Streamlit `use_container_width` replaced; EDA chart given a shared y-axis; highlighted table cells had unreadable text in dark mode (now explicit colours) | Fixed |
