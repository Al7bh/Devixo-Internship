"""
Reusable feature-engineering code for the used-car price project.

Why this file exists
--------------------
The raw CSV stores numbers as text ("23.4 kmpl", "1248 CC", "74 bhp") and the
car `name` contains the brand. We need the SAME cleaning to happen
  (a) when we train the model, and
  (b) when the Streamlit app predicts for a new car.
Putting it in ONE transformer inside the saved scikit-learn Pipeline guarantees
both places behave identically, and it means the saved model accepts RAW input.

The transformer is "stateless": it learns nothing from the data, so it cannot
leak information from the test set. Anything that DOES learn from data (median
imputation, scaling, one-hot categories) lives in the ColumnTransformer that
comes after it and is fitted on training data only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# The newest car in the dataset is from 2020, so we measure "age" from 2020.
# (Any constant works for tree models; we fix it and document it.)
REFERENCE_YEAR = 2020

# Raw dataset columns we read (everything except the target and `torque`,
# which we deliberately do not use - see notebook 02).
RAW_INPUT_COLUMNS = [
    "name", "year", "km_driven", "fuel", "seller_type",
    "transmission", "owner", "mileage", "engine", "max_power", "seats",
]
# `seats` and `seller_type` were tested in notebook 02 and did not improve
# the model, so the final model does not use them. They are OPTIONAL inputs:
# if they are missing the transformer fills them with NaN instead of failing.
OPTIONAL_INPUT_COLUMNS = ["seller_type", "seats"]
REQUIRED_INPUT_COLUMNS = [c for c in RAW_INPUT_COLUMNS if c not in OPTIONAL_INPUT_COLUMNS]

# Columns the transformer OUTPUTS (a superset of what the model finally uses).
ENGINEERED_NUMERIC = [
    "car_age", "km_driven", "log_km_driven", "km_per_year",
    "mileage", "engine", "max_power", "seats", "bhp_per_litre",
]
ENGINEERED_CATEGORICAL = ["brand", "fuel", "seller_type", "transmission", "owner"]
ALL_OUTPUT_COLUMNS = ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL

# Brands whose name has two words, so "first word" alone would be wrong.
_TWO_WORD_BRANDS = {"Land": "Land Rover"}

_NUMBER_AT_START = r"^\s*(\d+(?:\.\d+)?)"


def _leading_number(series: pd.Series) -> pd.Series:
    """'23.4 kmpl' -> 23.4 ; '1248 CC' -> 1248.0 ; ' bhp' -> NaN ; 74 -> 74.0"""
    text = series.astype("string")
    return pd.to_numeric(text.str.extract(_NUMBER_AT_START)[0], errors="coerce").astype(float)


def extract_brand(name: pd.Series) -> pd.Series:
    """'Maruti Swift Dzire VDI' -> 'Maruti' ; 'Land Rover Discovery' -> 'Land Rover'"""
    first = name.astype("string").str.strip().str.split().str[0]
    return first.replace(_TWO_WORD_BRANDS).astype(object)


class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """Turn raw car rows into clean numeric + categorical columns."""

    def fit(self, X, y=None):
        self._check_columns(X)
        self.n_features_in_ = len(REQUIRED_INPUT_COLUMNS)
        return self

    @staticmethod
    def _check_columns(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CarFeatureEngineer expects a pandas DataFrame with raw car columns.")
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")

    def transform(self, X):
        self._check_columns(X)
        out = pd.DataFrame(index=X.index)

        year = pd.to_numeric(X["year"], errors="coerce").astype(float)
        km = pd.to_numeric(X["km_driven"], errors="coerce").astype(float)
        km = km.where(km >= 0)                       # negative km is impossible -> missing

        out["car_age"] = (REFERENCE_YEAR - year).clip(lower=0)
        out["km_driven"] = km
        out["log_km_driven"] = np.log1p(km)
        out["km_per_year"] = km / out["car_age"].clip(lower=1)

        # Text -> number. Zero or blank values are not real measurements -> NaN.
        out["mileage"] = _leading_number(X["mileage"]).where(lambda s: s > 0)
        out["engine"] = _leading_number(X["engine"]).where(lambda s: s > 0)
        out["max_power"] = _leading_number(X["max_power"]).where(lambda s: s > 0)
        seats = X["seats"] if "seats" in X.columns else pd.Series(np.nan, index=X.index)
        out["seats"] = pd.to_numeric(seats, errors="coerce").astype(float)
        out["bhp_per_litre"] = out["max_power"] / (out["engine"] / 1000.0)

        out["brand"] = extract_brand(X["name"])
        for col in ["fuel", "seller_type", "transmission", "owner"]:
            values = X[col] if col in X.columns else pd.Series(np.nan, index=X.index)
            out[col] = values.astype("string").str.strip().astype(object)

        return out[ALL_OUTPUT_COLUMNS]

    def get_feature_names_out(self, input_features=None):
        return np.array(ALL_OUTPUT_COLUMNS, dtype=object)
