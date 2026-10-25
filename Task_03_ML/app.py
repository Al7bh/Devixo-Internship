"""
Streamlit app - Used-car price estimator (Task 03 bonus).

Run from the project root:
    streamlit run app.py

How it works: the saved model (models/best_model.joblib) is a complete
scikit-learn pipeline. It accepts a raw car description and returns a price,
so this file only has to collect inputs, build one row, and call predict().
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # lets joblib find src.car_features when loading the model

MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
OTHER_BRAND = "Other brand"
GAS_FUELS = ("CNG", "LPG")             # mileage of these is stored in km/kg in the dataset


# ----------------------------------------------------------------------------
# Plain functions (kept separate from the UI so they can be unit-tested)
# ----------------------------------------------------------------------------
def build_input_frame(brand, year, km_driven, fuel, transmission, owner, mileage, engine, max_power):
    """Build the one-row DataFrame in the same raw format as the training CSV."""
    unit = "km/kg" if fuel in GAS_FUELS else "kmpl"
    return pd.DataFrame([{
        "name": "Other" if brand == OTHER_BRAND else brand,
        "year": int(year),
        "km_driven": float(km_driven),
        "fuel": fuel,
        "transmission": transmission,
        "owner": owner,
        "mileage": f"{mileage} {unit}",
        "engine": f"{engine} CC",
        "max_power": f"{max_power} bhp",
    }])


def input_warnings(meta, km_driven, mileage, engine, max_power):
    """Messages when an input is outside the range the model saw in training."""
    r = meta["input_ranges"]
    msgs = []
    if km_driven > r["km_driven_p99"]:
        msgs.append(f"Kilometres driven is higher than 99% of training cars ({r['km_driven_p99']:,.0f} km).")
    for label, value, key in [("Mileage", mileage, "mileage_p01_p99"),
                              ("Engine size", engine, "engine_cc_p01_p99"),
                              ("Max power", max_power, "max_power_bhp_p01_p99")]:
        low, high = r[key]
        if not (low <= value <= high):
            msgs.append(f"{label} ({value}) is outside the usual range {low:.0f}-{high:.0f}; the estimate may be less reliable.")
    return msgs


def format_rupees(value):
    return f"Rs {value:,.0f}  (about {value / 1e5:.2f} lakh)"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(META_PATH) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Used-Car Price Estimator", page_icon="🚗")
st.title("🚗 Used-Car Price Estimator")
st.caption("Task 03 - Devixo Solutions AI/ML Internship. Estimates a used car's selling price from its details.")

if not (MODEL_PATH.exists() and META_PATH.exists()):
    st.error("Model files not found. Run notebook `03_modeling_tuning_and_evaluation.ipynb` first "
             "(it creates the `models/` folder contents).")
    st.stop()

model = load_model()
meta = load_metadata()
choices = meta["input_choices"]
y_min, y_max = meta["input_ranges"]["year"]

st.subheader("Car details")
col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", choices["brand"] + [OTHER_BRAND])
    year = st.number_input("Manufacturing year", min_value=1990, max_value=int(y_max), value=2015, step=1)
    km_driven = st.number_input("Kilometres driven", min_value=0, max_value=1_000_000, value=60_000, step=1_000)
    fuel = st.selectbox("Fuel type", choices["fuel"], index=choices["fuel"].index("Petrol") if "Petrol" in choices["fuel"] else 0)
with col2:
    transmission = st.selectbox("Transmission", choices["transmission"])
    owner = st.selectbox("Owner", choices["owner"])
    mileage_label = "Mileage (km/kg)" if fuel in GAS_FUELS else "Mileage (kmpl)"
    mileage = st.number_input(mileage_label, min_value=1.0, max_value=60.0, value=19.0, step=0.5)
    engine = st.number_input("Engine size (CC)", min_value=500, max_value=6000, value=1200, step=50)
    max_power = st.number_input("Max power (bhp)", min_value=20.0, max_value=700.0, value=80.0, step=1.0)

if st.button("Estimate price", type="primary"):
    row = build_input_frame(brand, year, km_driven, fuel, transmission, owner, mileage, engine, max_power)
    price = float(model.predict(row)[0])
    ratio = meta["indicative_range_ratio"]
    low, high = price * ratio["low"], price * ratio["high"]

    st.metric("Estimated selling price", format_rupees(price))
    st.write(f"**Likely range:** {format_rupees(low)} to {format_rupees(high)}")
    st.caption(f"The range covers the middle 80% of the model's past errors on training data "
               f"(on the untouched test set, {ratio['test_coverage']:.0%} of cars fell inside such a range).")
    for message in input_warnings(meta, km_driven, mileage, engine, max_power):
        st.warning(message)

with st.expander("About this model"):
    tm = meta["test_metrics"]
    st.write(
        f"- **Model:** {meta['selected_model']} (chosen by cross-validation)\n"
        f"- **Trained on:** {meta['training_rows']:,} used-car listings (CarDekho data, newest car year {y_max})\n"
        f"- **Test-set accuracy:** R2 = {tm['R2']:.2f}; median error = {tm['MedAPE_%']:.1f}% of the price\n"
        f"- **Currency:** assumed Indian rupees. Prices reflect the older market in the dataset, "
        f"so treat the result as an estimate, not a valuation."
    )
