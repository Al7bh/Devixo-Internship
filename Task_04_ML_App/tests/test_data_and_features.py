"""Tests for data loading, cleaning, splitting and feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.data_utils import (EXPECTED_COLUMNS, ID_COL, RAW_FEATURES, TARGET, clean,
                            get_split, iqr_outlier_counts, load_raw, validate)
from src.features import AGE_LABELS, add_features


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_schema_has_all_expected_columns(raw):
    assert validate(raw)["missing_columns"] == []
    assert list(raw.columns) == EXPECTED_COLUMNS


def test_dataset_is_clean_as_documented(raw):
    report = validate(raw)
    assert report["missing_total"] == 0
    assert report["blank_strings"] == 0
    assert report["full_duplicates"] == 0
    assert report["duplicates_ignoring_id"] == 0
    assert report["id_is_unique"]
    assert report["target_values"] == [0, 1]


def test_clean_keeps_all_rows_when_data_is_clean(raw):
    assert len(clean(raw)) == len(raw)


def test_clean_removes_duplicates_and_missing_target():
    df = load_raw().head(20)
    dup = df.iloc[[0]].copy()
    dup[ID_COL] = 999999999            # same customer details, different ID
    bad = df.iloc[[1]].copy()
    bad[ID_COL] = 999999998
    bad[TARGET] = np.nan
    out = clean(pd.concat([df, dup, bad], ignore_index=True))
    assert len(out) == 20


def test_split_is_stratified_disjoint_and_has_no_id(raw):
    df = clean(raw)
    X_train, X_test, y_train, y_test = get_split(df)
    assert ID_COL not in X_train.columns and TARGET not in X_train.columns
    assert list(X_train.columns) == RAW_FEATURES
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))      # no leakage between splits
    assert abs(y_train.mean() - y_test.mean()) < 0.01             # stratified


def test_split_is_reproducible(raw):
    df = clean(raw)
    a = get_split(df)[1].index
    b = get_split(df)[1].index
    assert list(a) == list(b)


def test_outlier_report_is_detection_only(raw):
    before = raw.copy()
    rep = iqr_outlier_counts(raw)
    pd.testing.assert_frame_equal(before, raw)       # data not modified
    assert set(rep["column"]) == {"credit_score", "age", "tenure", "balance", "estimated_salary"}


# ----------------------------------------------------------- feature engineering
def _row(**kw):
    base = dict(credit_score=650, country="France", gender="Male", age=40, tenure=5,
                balance=50000.0, products_number=2, credit_card=1, active_member=1,
                estimated_salary=60000.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_features_values_are_correct():
    out = add_features(_row(balance=0.0, products_number=3, age=61))
    assert out["balance_zero"].iloc[0] == 1
    assert out["many_products"].iloc[0] == 1
    assert out["age_group"].iloc[0] == "60+"
    assert out["balance_to_salary"].iloc[0] == 0


@pytest.mark.parametrize("age,group", [(18, "<=30"), (30, "<=30"), (31, "31-40"), (40, "31-40"),
                                       (41, "41-50"), (50, "41-50"), (51, "51-60"), (60, "51-60"),
                                       (61, "60+"), (100, "60+")])
def test_age_group_boundaries(age, group):
    assert add_features(_row(age=age))["age_group"].iloc[0] == group
    assert group in AGE_LABELS


def test_features_do_not_mutate_input_and_have_no_nan_or_inf():
    df = load_raw()[RAW_FEATURES]
    before = df.copy()
    out = add_features(df)
    pd.testing.assert_frame_equal(before, df)
    new_cols = ["balance_zero", "balance_to_salary", "many_products", "age_group"]
    assert not out[new_cols].isnull().any().any()
    assert np.isfinite(out["balance_to_salary"]).all()


def test_zero_salary_does_not_divide_by_zero():
    out = add_features(_row(estimated_salary=0.0, balance=1000.0))
    assert np.isfinite(out["balance_to_salary"].iloc[0])
