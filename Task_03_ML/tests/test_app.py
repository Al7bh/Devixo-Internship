"""Tests for the Streamlit app (uses Streamlit's built-in headless AppTest)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
MODEL_READY = (ROOT / "models" / "best_model.joblib").exists() and (ROOT / "models" / "model_metadata.json").exists()

pytestmark = pytest.mark.skipif(not MODEL_READY, reason="model not trained yet")


def test_build_input_frame_matches_training_format():
    import app
    row = app.build_input_frame("Maruti", 2015, 60000, "CNG", "Manual", "First Owner", 26.0, 998, 58.0)
    assert row.loc[0, "mileage"] == "26.0 km/kg"            # gas fuels use km/kg
    assert row.loc[0, "engine"] == "998 CC"
    assert app.build_input_frame("Other brand", 2015, 1, "Petrol", "Manual", "First Owner", 19, 1200, 80).loc[0, "name"] == "Other"


def test_app_starts_without_error_and_predicts():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) == 1
    assert "Rs" in at.metric[0].value


def test_app_warns_for_unrealistic_inputs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    next(w for w in at.number_input if w.label == "Kilometres driven").set_value(900_000)
    at.button[0].click().run()
    assert not at.exception
    assert any("Kilometres" in w.value for w in at.warning)


def test_version_problems_detects_a_mismatch():
    import app
    saved_ok = {"library_versions": {}}
    assert app.version_problems(saved_ok) == []
    fake = {"library_versions": {"scikit-learn": "0.0.1", "xgboost": "0.0.1"}}
    found = app.version_problems(fake)
    assert any("scikit-learn" in m for m in found) and any("xgboost" in m for m in found)


def test_app_shows_a_helpful_message_when_the_model_file_is_unreadable(tmp_path):
    """Simulates the real-world failure (model saved with other library versions): the app must explain the fix, not crash."""
    import shutil
    from streamlit.testing.v1 import AppTest
    shutil.copy(APP, tmp_path / "app.py")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    (tmp_path / "models").mkdir()
    shutil.copy(ROOT / "models" / "model_metadata.json", tmp_path / "models" / "model_metadata.json")
    (tmp_path / "models" / "best_model.joblib").write_bytes(b"this is not a valid model file")
    import streamlit as st
    st.cache_resource.clear()                    # earlier tests cached the REAL model under the same function name
    st.cache_data.clear()
    try:
        at = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=60).run()
        assert not at.exception                  # no crash / traceback
        assert any("could not be loaded" in e.value for e in at.error)
        assert any("src.retrain" in c.value for c in at.code)
    finally:
        st.cache_resource.clear()
        st.cache_data.clear()
