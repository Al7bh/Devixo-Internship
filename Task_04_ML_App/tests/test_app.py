"""End-to-end tests of the Streamlit app using Streamlit's AppTest runner."""
import pytest
from streamlit.testing.v1 import AppTest

from src.data_utils import ROOT
from src.pipeline import MODEL_NAMES

APP = str(ROOT / "app" / "app.py")
PAGES = sorted((ROOT / "app" / "pages").glob("*.py"))
TIMEOUT = 120


def run_main():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def widget(elements, label):
    """Find a widget by its visible label (more robust than a position number)."""
    found = [e for e in elements if getattr(e, "label", "") == label]
    assert found, f"widget {label!r} not found"
    return found[0]


def verdicts(at):
    return list(at.error) + list(at.success)


def click_predict(at):
    btn = [b for b in at.get("button") if "Predict" in getattr(b, "label", "")]
    assert btn, "Predict button not found"
    btn[0].click()
    at.run()
    return at


def test_main_page_loads_without_error():
    at = run_main()
    assert not at.exception
    assert "Bank Customer Churn Predictor" in at.title[0].value


def test_form_has_all_inputs_and_predict_button():
    at = run_main()
    assert len(at.number_input) == 3        # credit score, balance, salary
    assert len(at.slider) == 2              # age, tenure
    assert len(at.selectbox) >= 3           # model, country, products
    assert len(at.radio) == 3               # gender, credit card, active
    assert any("Predict" in getattr(b, "label", "") for b in at.get("button"))


def test_no_result_before_clicking_predict():
    at = run_main()
    assert len(at.error) == 0 and len(at.success) == 0


def test_prediction_shows_result_probability_and_explanations():
    at = click_predict(run_main())
    assert not at.exception
    assert len(verdicts(at)) == 1                       # exactly one verdict
    verdict = verdicts(at)[0].value
    assert "probability" in verdict.lower() and "%" in verdict
    assert len(at.dataframe) >= 1                       # segment table / input table
    assert len(at.get("progress")) == 1                 # probability bar is shown


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_can_be_selected_and_used(model_name):
    at = run_main()
    at.sidebar.selectbox[0].set_value(model_name)
    at.run()
    at = click_predict(at)
    assert not at.exception
    assert len(verdicts(at)) == 1


def test_extreme_input_gives_warning_not_crash():
    at = run_main()
    widget(at.number_input, "Credit score").set_value(900)          # form maximum
    widget(at.number_input, "Account balance").set_value(900000.0)  # far outside the training data
    at = click_predict(at)
    assert not at.exception
    assert len(at.warning) >= 1
    assert len(verdicts(at)) == 1


def test_high_risk_profile_scores_higher_than_low_risk_profile():
    def prob_for(country, age, products, active):
        at = run_main()
        widget(at.selectbox, "Country").set_value(country)
        widget(at.slider, "Age").set_value(age)
        widget(at.selectbox, "Number of bank products").set_value(products)
        widget(at.radio, "Active member?").set_value("Yes" if active else "No")
        at = click_predict(at)
        assert not at.exception
        verdict = verdicts(at)[0].value
        return float(verdict.split("**")[1].strip("%")) / 100 if "**" in verdict else None

    low = prob_for("France", 28, 2, True)
    high = prob_for("Germany", 55, 1, False)
    assert low is not None and high is not None
    assert high > low


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_every_extra_page_runs_without_error(page):
    at = AppTest.from_file(str(page), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
