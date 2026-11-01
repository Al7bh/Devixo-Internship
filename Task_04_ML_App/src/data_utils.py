"""Loading, validating, cleaning and splitting the bank churn dataset.

Every function here is shared by the notebooks, the training script, the tests
and the Streamlit app, so all of them see exactly the same data logic.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Bank Customer Churn Prediction.csv"

TARGET = "churn"
ID_COL = "customer_id"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Raw input columns used by the model (customer_id and churn are NOT inputs).
RAW_FEATURES = [
    "credit_score", "country", "gender", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member", "estimated_salary",
]
NUMERIC_COLS = ["credit_score", "age", "tenure", "balance", "estimated_salary"]
EXPECTED_COLUMNS = [ID_COL] + RAW_FEATURES + [TARGET]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the original CSV without changing anything."""
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> dict:
    """Run data-quality checks. Returns the results; does not modify df."""
    return {
        "shape": tuple(df.shape),
        "missing_total": int(df.isnull().sum().sum()),
        "blank_strings": int(
            df.select_dtypes(exclude="number").apply(lambda c: c.astype(str).str.strip().eq("")).sum().sum()
        ),
        "full_duplicates": int(df.duplicated().sum()),
        "duplicates_ignoring_id": int(df.drop(columns=ID_COL).duplicated().sum()),
        "id_is_unique": bool(df[ID_COL].is_unique),
        "target_values": sorted(df[TARGET].dropna().unique().tolist()),
        "missing_columns": [c for c in EXPECTED_COLUMNS if c not in df.columns],
    }


def iqr_outlier_counts(df: pd.DataFrame, cols=NUMERIC_COLS) -> pd.DataFrame:
    """Count values outside 1.5*IQR for each numeric column (detection only)."""
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[c] < low) | (df[c] > high)).sum())
        rows.append({
            "column": c, "lower_bound": float(low), "upper_bound": float(high),
            "n_outliers": n, "pct": round(100 * n / len(df), 2),
            "min": float(df[c].min()), "max": float(df[c].max()),
        })
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning steps that are safe to do BEFORE the train/test split.

    - drop rows with a missing target (none in this dataset)
    - drop duplicate customers (ignoring the ID column; none in this dataset)
    Outliers are deliberately NOT removed (see docs/decisions.md).
    Anything that learns from the data (imputing, scaling, encoding) lives in the
    sklearn Pipeline instead, so it is fitted on the training split only.
    """
    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates(subset=[c for c in df.columns if c != ID_COL])
    return df.reset_index(drop=True)


def get_split(df: pd.DataFrame):
    """Stratified 80/20 split. Returns X_train, X_test, y_train, y_test.

    X has only the raw model inputs (no customer_id). The same call (same
    random_state) is used in training, tests and the app, so results match.
    """
    X = df[RAW_FEATURES]
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
