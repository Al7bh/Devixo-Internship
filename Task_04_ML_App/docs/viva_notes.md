# Viva / interview notes

Use these to practise explaining the project **in your own words**. Check every number against `models/metrics.json`.

**1. Why this dataset and what is its limitation?**
Bank customer churn: binary target, mixed numeric/categorical columns, supports probability and ROC-AUC. The Kaggle page does not say whether
the data is real or simulated or give a licence, so I do not claim it is real. I rejected another archive because its training file had groups
with exactly 100% churn and its test file behaved very differently.

**2. What data cleaning did you do?**
I checked missing values, blank strings, duplicates (also ignoring the ID) and uniqueness of the ID: none were found, so nothing needed fixing.
I still built the steps into the pipeline (imputers, duplicate removal) and tested them, but I did not invent problems.

**3. Why did you keep the outliers?**
IQR flagged ages and credit scores, but those values (ages up to 92, scores from 350) are plausible customers. Customers aged 70+ have a
different churn rate, so removing them would delete information. Trees ignore such values; the other models use standardised inputs.

**4. How did you prevent data leakage?**
I split first (stratified 80/20), then put imputing, scaling, encoding and the model in one Pipeline fitted on the training data only
(notebook 02 shows the scaler mean equals the *training* mean, not the whole-data mean). Engineered features use only values from the same row.
The best model is picked by cross-validation on the training set, not on the test set.

**5. Why not use accuracy?**
Only about 20% of customers churn, so always predicting "stays" scores 79.7%.
My best model's accuracy is 0.868, only a few points higher, while its recall at 0.5 is 0.467. So I use ROC-AUC, recall and F1.

**6. Which model won and why?**
Gradient Boosting, with the highest cross-validated ROC-AUC after tuning (0.864); test ROC-AUC 0.871. Another model was close, and on this test set it may score slightly
higher on some metric. That gap is small compared with test-set noise, so I kept the rule I fixed in advance.

**7. Did tuning help?**
It depends on the model. KNN improved a lot (test ROC-AUC 0.788 to 0.844; its best n_neighbors was 61, and my first grid stopped at 41, so I widened it).
Logistic Regression barely changed (0.849 to 0.849). F1 at threshold 0.5 even dropped for some models because I tuned for ROC-AUC, not F1 at 0.5.

**8. What is the decision threshold and why change it?**
The model outputs a probability. 0.5 is only a convention. I chose 0.29 by maximising F1 on out-of-fold predictions from the training data, which
raised recall from 0.467 to 0.668 with precision going from 0.802 to 0.610. The right threshold depends on the business cost of a missed churner versus a false alarm, and
this dataset has no cost information.

**9. What did the feature-engineering test show?**
The ablation (cross-validation, training data only) showed `many_products` raised Logistic Regression's ROC-AUC a lot, while the balance features added almost nothing on their own,
and for Random Forest all changes were within noise. I kept all four because none hurt, and I report that honestly.

**10. What does feature importance show?**
Permutation importance: shuffle one column on the test set and see how much ROC-AUC drops. The top inputs were age (0.125), products_number (0.111), active_member (0.030).
It shows what the model relies on, not what causes churn, and correlated columns can share importance.

**11. Can the probabilities be trusted?**
I checked calibration on the test set: in probability bins with at least 50 customers, predicted and observed churn rates differ by at most 7.4 percentage points
(Brier score 0.099). That is one test set, so it is evidence, not proof; the probabilities are best used to rank customers.

**12. How does the app explain one prediction?**
It replaces each of the customer's values, one at a time, with the typical training value and shows how the probability changes. This is a simple model-based
explanation; it ignores interactions and is not causal. It also shows how similar customers behaved historically.

**13. What are the weaknesses?**
Unknown data source and churn time window; small groups (3-4 products); one test split of 2,000 rows; gender and country as inputs need ethical review in a real bank;
and the app has not been deployed or checked in a real browser by the tests.

**14. How would you improve it?**
Confirm the data source, calibrate probabilities, use repeated cross-validation or a larger holdout, use cost-based thresholds with real costs, monitor drift after deployment,
and review fairness across groups.

**15. How do you know the app works?**
The project has automated tests covering data checks, feature edge cases, model loading, edge/unseen/missing inputs, a check that stored metrics equal a fresh evaluation,
that the notebooks are executed without errors, and Streamlit AppTest runs of every page and every model. I also started the real Streamlit server and confirmed it responds.
Tests do not replace looking at the app myself.

**16. Why are the training steps in notebooks and the app in a .py file?**
Notebooks show each step with its output and explanation, which is what you want when exploring and training. A Streamlit app is a program that must run top to bottom
without you, so it is a script. Reusable logic lives in `src/` so notebooks, tests and the app all use the same code.
