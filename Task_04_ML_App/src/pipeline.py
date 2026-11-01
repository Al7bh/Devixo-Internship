"""Builds the full sklearn Pipeline: feature engineering -> preprocessing -> model.

The saved model is this whole Pipeline, so the Streamlit app applies exactly the
same transformations as training. Imputers, scalers and encoders are fitted only
on the training data when `.fit()` is called on the training split.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.data_utils import RANDOM_STATE
from src.features import (ALL_ENGINEERED, ENGINEERED_BINARY, ENGINEERED_CATEGORICAL,
                          ENGINEERED_NUMERIC, add_features)

MODEL_NAMES = ["Logistic Regression", "KNN", "SVM", "Random Forest", "Gradient Boosting"]

BASE_NUMERIC = ["credit_score", "age", "tenure", "balance", "estimated_salary", "products_number"]
BASE_CATEGORICAL = ["country", "gender"]
BASE_BINARY = ["credit_card", "active_member"]


def build_preprocessor(engineered=None) -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categoricals.

    `engineered` chooses which engineered columns to include (default: all).
    The imputers do nothing on this dataset (it has no missing values) but make
    the saved model safe if a missing value ever arrives.
    """
    engineered = list(ALL_ENGINEERED if engineered is None else engineered)
    numeric = BASE_NUMERIC + [c for c in engineered if c in ENGINEERED_NUMERIC]
    categorical = BASE_CATEGORICAL + [c for c in engineered if c in ENGINEERED_CATEGORICAL]
    binary = BASE_BINARY + [c for c in engineered if c in ENGINEERED_BINARY]

    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ])


def make_model(name: str, **params):
    """Create an untrained model with sensible defaults (params override them)."""
    if name == "Logistic Regression":
        defaults = dict(max_iter=2000, random_state=RANDOM_STATE)
        return LogisticRegression(**{**defaults, **params})
    if name == "KNN":
        return KNeighborsClassifier(**params)
    if name == "SVM":
        defaults = dict(probability=True, cache_size=1000, random_state=RANDOM_STATE)
        return SVC(**{**defaults, **params})
    if name == "Random Forest":
        defaults = dict(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        return RandomForestClassifier(**{**defaults, **params})
    if name == "Gradient Boosting":
        defaults = dict(random_state=RANDOM_STATE)
        return GradientBoostingClassifier(**{**defaults, **params})
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str, engineered=None, **model_params) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor(engineered)),
        ("model", make_model(name, **model_params)),
    ])
