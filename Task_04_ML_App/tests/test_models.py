"""QA tests for the saved models. Requires `python -m src.train` to have been run."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.data_utils import ROOT, RAW_FEATURES, clean, get_split, load_raw
from src.pipeline import MODEL_NAMES

MODELS_DIR = ROOT / "models"


def slug(n):
    return n.lower().replace(" ", "_")


@pytest.fixture(scope="module")
def metrics():
    return json.loads((MODELS_DIR / "metrics.json").read_text())


@pytest.fixture(scope="module")
def split():
    return get_split(clean(load_raw()))


@pytest.fixture(scope="module", params=MODEL_NAMES)
def model(request):
    return request.param, joblib.load(MODELS_DIR / "all_models" / f"{slug(request.param)}.joblib")


def customer(**kw):
    base = dict(credit_score=650, country="France", gender="Male", age=40, tenure=5,
                balance=50000.0, products_number=2, credit_card=1, active_member=1,
                estimated_salary=60000.0)
    base.update(kw)
    return pd.DataFrame([base])[RAW_FEATURES]


def test_all_models_and_final_model_exist():
    for n in MODEL_NAMES:
        assert (MODELS_DIR / "all_models" / f"{slug(n)}.joblib").exists()
    assert (MODELS_DIR / "final_model.joblib").exists()
    assert (MODELS_DIR / "metrics.json").exists()


def test_final_model_is_the_best_model(metrics):
    final = joblib.load(MODELS_DIR / "final_model.joblib")
    best = joblib.load(MODELS_DIR / "all_models" / f"{slug(metrics['best_model'])}.joblib")
    X = customer()
    assert final.predict_proba(X)[0, 1] == pytest.approx(best.predict_proba(X)[0, 1])


def test_predict_proba_shape_and_range(model, split):
    _, pipe = model
    p = pipe.predict_proba(split[1])
    assert p.shape == (len(split[1]), 2)
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(p.sum(axis=1), 1)


@pytest.mark.parametrize("kw", [
    dict(credit_score=300, age=18, tenure=0, balance=0.0, estimated_salary=0.0),     # minimum values
    dict(credit_score=900, age=100, tenure=10, balance=1_000_000.0, estimated_salary=1_000_000.0,
         products_number=4),                                                          # maximum values
    dict(balance=0.0, products_number=4, active_member=0, country="Germany"),         # risky combination
    dict(country="Italy"),                                                            # unseen category
])
def test_edge_inputs_do_not_crash_and_return_valid_probability(model, kw):
    _, pipe = model
    p = pipe.predict_proba(customer(**kw))[0, 1]
    assert np.isfinite(p) and 0 <= p <= 1


def test_missing_value_in_input_is_handled(model):
    """The pipeline imputes, so a missing value must not crash prediction."""
    _, pipe = model
    X = customer()
    X.loc[0, "credit_score"] = np.nan
    assert np.isfinite(pipe.predict_proba(X)[0, 1])


def test_customer_id_is_not_a_model_input(model):
    _, pipe = model
    cols = set()
    for _, _, c in pipe.named_steps["prep"].transformers_:
        if isinstance(c, list):
            cols.update(c)
    assert "customer_id" not in cols and "churn" not in cols


def test_saved_metrics_match_a_fresh_evaluation(model, metrics, split):
    """Numbers shown in the app/report must equal what the saved model really scores."""
    name, pipe = model
    _, X_test, _, y_test = split
    fresh = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    saved = metrics["models"][name]["tuned"]["test_default_threshold"]["roc_auc"]
    assert fresh == pytest.approx(saved, abs=1e-6)


def test_confusion_matrix_totals_match_test_size(metrics):
    for n in MODEL_NAMES:
        for key in ("test_default_threshold", "test_tuned_threshold"):
            cm = np.array(metrics["models"][n]["tuned"][key]["confusion_matrix"])
            assert cm.sum() == metrics["meta"]["n_test"]


def test_best_model_has_highest_cv_score(metrics):
    cv = {n: metrics["models"][n]["tuned"]["cv_roc_auc_mean"] for n in MODEL_NAMES}
    assert metrics["best_model"] == max(cv, key=cv.get)


def test_every_model_beats_random_guessing(metrics):
    for n in MODEL_NAMES:
        assert metrics["models"][n]["tuned"]["test_default_threshold"]["roc_auc"] > 0.6


def test_at_least_four_models_compared(metrics):
    assert len(metrics["models"]) >= 4
