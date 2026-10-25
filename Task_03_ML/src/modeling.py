"""
Shared modeling helpers used by the notebooks, tests and the Streamlit app.

Design in one paragraph
-----------------------
Every model is wrapped in the SAME pipeline:

    raw car rows
      -> CarFeatureEngineer      (text -> numbers, brand, car_age ... ; learns nothing)
      -> ColumnTransformer       (median-impute + scale numbers, impute + one-hot categories;
                                  LEARNS from training data only)
      -> model                   (Decision Tree / Random Forest / SVR / Gradient Boosting ...)

and the whole thing is wrapped in a TransformedTargetRegressor that trains on
log(price) but returns predictions in rupees. Because it is ONE object, saving
it with joblib/pickle saves the preprocessing too.
"""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.car_features import CarFeatureEngineer

RANDOM_STATE = 42
TARGET = "selling_price"

# ---- Final feature lists (justified in notebooks/02_feature_engineering_and_selection.ipynb)
NUMERIC_FEATURES = ["car_age", "log_km_driven", "mileage", "engine", "max_power"]
CATEGORICAL_FEATURES = ["brand", "fuel", "transmission", "owner"]

# Categories seen fewer than this many times in the training data are grouped
# into one "infrequent" bucket (stops one-hot columns for 1-row brands).
MIN_CATEGORY_FREQUENCY = 10


def build_preprocessor(numeric_features=None, categorical_features=None):
    """Imputation + scaling + one-hot encoding. Fitted on training data only."""
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),          # needed by SVR; harmless for trees
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=MIN_CATEGORY_FREQUENCY,
            sparse_output=False,
        )),
    ])
    transformers = [("num", numeric_steps, list(numeric_features))]
    if categorical_features:
        transformers.append(("cat", categorical_steps, list(categorical_features)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(model, numeric_features=None, categorical_features=None, log_target=True):
    """Full raw-input -> price pipeline around any scikit-learn regressor."""
    inner = Pipeline([
        ("features", CarFeatureEngineer()),
        ("preprocess", build_preprocessor(numeric_features, categorical_features)),
        ("model", model),
    ])
    if not log_target:
        return inner
    return TransformedTargetRegressor(regressor=inner, func=np.log1p, inverse_func=np.expm1)


def regression_metrics(y_true, y_pred):
    """Metrics in rupees (R2, MAE, RMSE) plus relative-error metrics.

    RMSLE  = error on the log scale  (~ typical % error; treats cheap and
             expensive cars fairly)
    MedAPE = median absolute % error (robust, easy to explain)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "RMSLE": float(np.sqrt(mean_squared_log_error(y_true, y_pred))),
        "MedAPE_%": float(np.median(np.abs(y_true - y_pred) / y_true) * 100),
    }


# Scorers for cross_validate (all computed on the ORIGINAL rupee scale)
CV_SCORING = {
    "r2": "r2",
    "neg_mae": "neg_mean_absolute_error",
    "neg_rmse": "neg_root_mean_squared_error",
    "neg_rmsle": "neg_root_mean_squared_log_error",
}
# The ONE metric we use to choose between models / settings:
SELECTION_SCORE = "neg_root_mean_squared_log_error"
