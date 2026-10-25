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
