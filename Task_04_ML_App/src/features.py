"""Feature engineering.

`add_features` is used as the first step of the sklearn Pipeline. It only uses
values from the same row (no statistics learned from other rows), so it cannot
leak information from the test set.
"""
import numpy as np
import pandas as pd

AGE_BINS = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ["<=30", "31-40", "41-50", "51-60", "60+"]

# Names of the engineered columns, grouped by how the preprocessor treats them.
ENGINEERED_NUMERIC = ["balance_to_salary"]
ENGINEERED_BINARY = ["balance_zero", "many_products"]
ENGINEERED_CATEGORICAL = ["age_group"]
ALL_ENGINEERED = ENGINEERED_NUMERIC + ENGINEERED_BINARY + ENGINEERED_CATEGORICAL


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with four extra columns.

    balance_zero      1 if the account balance is exactly 0. Zero-balance
                      customers are a large group (~36%) with a different churn
                      rate, and 'balance == 0' is a different signal from
                      'balance is small'.
    balance_to_salary balance divided by salary: how much of a customer's income
                      is held at this bank (+1 avoids any division by zero).
    many_products     1 if the customer has 3 or more products. These groups are
                      small but behave very differently from 1-2 products.
    age_group         age in bands (<=30, 31-40, 41-50, 51-60, 60+). Churn does
                      not rise steadily with age (it peaks in the 50s), so bands
                      let simple models such as Logistic Regression capture that.
    """
    X = X.copy()
    X["balance_zero"] = (X["balance"] == 0).astype(int)
    X["balance_to_salary"] = X["balance"] / (X["estimated_salary"] + 1)
    X["many_products"] = (X["products_number"] >= 3).astype(int)
    X["age_group"] = pd.cut(X["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    X["balance_to_salary"] = X["balance_to_salary"].replace([np.inf, -np.inf], np.nan)
    return X
