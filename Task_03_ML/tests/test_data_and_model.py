"""Data-quality and saved-model tests. Run notebooks 02 and 03 first (they create the files)."""
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.car_features import RAW_INPUT_COLUMNS
from src.modeling import TARGET, regression_metrics

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "car_details_v3.csv"
TRAIN = ROOT / "data" / "processed" / "train.csv"
TEST = ROOT / "data" / "processed" / "test.csv"
JOBLIB = ROOT / "models" / "best_model.joblib"
PICKLE = ROOT / "models" / "best_model.pkl"
META = ROOT / "models" / "model_metadata.json"

needs_model = pytest.mark.skipif(not (JOBLIB.exists() and PICKLE.exists()), reason="model not trained yet")


# ---------------- data ----------------
def test_dataset_meets_task_size_requirement_after_cleaning():
    df = pd.read_csv(RAW).drop_duplicates()
    assert len(df) >= 3000                       # task sheet: minimum 3,000 rows


def test_no_duplicates_remain_and_no_overlap_between_train_and_test():
    train, test = pd.read_csv(TRAIN), pd.read_csv(TEST)
    assert not train.duplicated().any() and not test.duplicated().any()
    overlap = train.merge(test, how="inner")     # rows identical in ALL columns
    assert len(overlap) == 0, "identical rows found in both train and test (leakage)"


def test_split_sizes_and_columns():
    train, test = pd.read_csv(TRAIN), pd.read_csv(TEST)
    assert 0.18 < len(test) / (len(train) + len(test)) < 0.22   # 80/20 split
    assert set(RAW_INPUT_COLUMNS + [TARGET]) <= set(train.columns)


def test_target_is_positive_and_complete():
    train = pd.read_csv(TRAIN)
    assert train[TARGET].notna().all() and (train[TARGET] > 0).all()


# ---------------- saved model ----------------
@needs_model
def test_joblib_and_pickle_give_identical_predictions():
    test = pd.read_csv(TEST)[RAW_INPUT_COLUMNS]
    a = joblib.load(JOBLIB).predict(test)
    with open(PICKLE, "rb") as f:
        b = pickle.load(f).predict(test)
    assert np.allclose(a, b)


@needs_model
def test_predictions_are_positive_and_finite():
    pred = joblib.load(JOBLIB).predict(pd.read_csv(TEST)[RAW_INPUT_COLUMNS])
    assert np.isfinite(pred).all() and (pred > 0).all()


@needs_model
def test_test_set_quality_has_not_regressed():
    """Guards against a broken retrain. Thresholds are deliberately below the achieved scores."""
    test = pd.read_csv(TEST)
    m = regression_metrics(test[TARGET], joblib.load(JOBLIB).predict(test[RAW_INPUT_COLUMNS]))
    assert m["R2"] > 0.80
    assert m["MedAPE_%"] < 20


@needs_model
def test_model_beats_the_always_median_baseline_by_a_lot():
    train, test = pd.read_csv(TRAIN), pd.read_csv(TEST)
    base = regression_metrics(test[TARGET], np.full(len(test), train[TARGET].median()))
    model = regression_metrics(test[TARGET], joblib.load(JOBLIB).predict(test[RAW_INPUT_COLUMNS]))
    assert model["RMSLE"] < 0.6 * base["RMSLE"]


@needs_model
def test_model_handles_missing_specs_and_unseen_brand():
    odd = pd.DataFrame([{"name": "Tesla Model S", "year": 2019, "km_driven": 30000, "fuel": "Petrol",
                         "transmission": "Automatic", "owner": "First Owner",
                         "mileage": np.nan, "engine": np.nan, "max_power": np.nan}])
    price = joblib.load(JOBLIB).predict(odd)[0]
    assert np.isfinite(price) and price > 0


@needs_model
def test_newer_lower_km_car_is_predicted_dearer_than_older_one():
    base = {"name": "Maruti Swift VXI", "fuel": "Petrol", "transmission": "Manual", "owner": "First Owner",
            "mileage": "20 kmpl", "engine": "1197 CC", "max_power": "82 bhp"}
    new = {**base, "year": 2018, "km_driven": 20000}
    old = {**base, "year": 2006, "km_driven": 150000}
    p_new, p_old = joblib.load(JOBLIB).predict(pd.DataFrame([new, old]))
    assert p_new > p_old


@needs_model
def test_model_rejects_input_missing_required_column():
    bad = pd.read_csv(TEST)[RAW_INPUT_COLUMNS].drop(columns=["year"]).head(2)
    with pytest.raises(ValueError):
        joblib.load(JOBLIB).predict(bad)


@needs_model
def test_metadata_is_complete_and_consistent():
    meta = json.load(open(META))
    for key in ["selected_model", "best_hyperparameters", "test_metrics", "features", "library_versions",
                "input_choices", "input_ranges", "indicative_range_ratio"]:
        assert key in meta
    assert meta["training_rows"] + meta["test_rows"] >= 3000


@needs_model
def test_retrain_script_rebuilds_a_model_that_matches_the_original(tmp_path):
    """`python -m src.retrain` is the recovery path for library-version problems; it must really work."""
    import subprocess, sys
    done = subprocess.run([sys.executable, "-W", "ignore", "-m", "src.retrain", "--output-dir", str(tmp_path)],
                          cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-800:]
    assert (tmp_path / "best_model.joblib").exists() and (tmp_path / "best_model.pkl").exists()
    test = pd.read_csv(TEST)
    original = json.load(open(META))["test_metrics"]
    m = regression_metrics(test[TARGET], joblib.load(tmp_path / "best_model.joblib").predict(test[RAW_INPUT_COLUMNS]))
    assert abs(m["RMSLE"] - original["RMSLE"]) < 0.01
