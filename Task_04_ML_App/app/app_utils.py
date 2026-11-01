"""Helpers shared by all Streamlit pages (loading, caching, prediction logic)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # makes `import src...` work locally and on Streamlit Cloud
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.data_utils import RAW_FEATURES, load_raw, clean
from src.features import AGE_BINS, AGE_LABELS

MODELS_DIR = ROOT / "models"
ALL_MODELS_DIR = MODELS_DIR / "all_models"


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


@st.cache_data
def load_metrics() -> dict:
    return json.loads((MODELS_DIR / "metrics.json").read_text())


@st.cache_data
def load_reference() -> dict:
    return json.loads((MODELS_DIR / "reference.json").read_text())


@st.cache_data
def load_ablation() -> dict:
    path = MODELS_DIR / "feature_ablation.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_error_analysis() -> dict:
    path = MODELS_DIR / "error_analysis.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return clean(load_raw())


@st.cache_resource
def load_model(name: str):
    """Load one saved pipeline. Raises a clear error if the file is missing or the
    scikit-learn version differs (re-run `python -m src.train` in that case)."""
    return joblib.load(ALL_MODELS_DIR / f"{slug(name)}.joblib")


def safe_setup():
    """Load metrics once; stop the page with a helpful message if the files are missing."""
    try:
        return load_metrics(), load_reference()
    except FileNotFoundError:
        st.error("Model files not found. Run `python -m src.train` from the project root first.")
        st.stop()


def sidebar_model_picker(metrics: dict, key: str = "model_choice") -> str:
    """Shared model selector. The choice is kept across pages."""
    names = metrics["model_order"]
    best = metrics["best_model"]
    if key not in st.session_state:
        st.session_state[key] = best
    st.sidebar.header("Model")
    choice = st.sidebar.selectbox(
        "Choose a model", names, key=key,
        format_func=lambda n: f"{n} (best)" if n == best else n,
        help="All models were tuned. The one marked 'best' had the highest "
             "cross-validated ROC-AUC on the training data.",
    )
    m = metrics["models"][choice]["tuned"]["test_default_threshold"]
    st.sidebar.caption(f"Test ROC-AUC: **{m['roc_auc']:.3f}**  \n"
                       f"Recall: **{m['recall']:.3f}** | Precision: **{m['precision']:.3f}**")
    return choice


def to_frame(inputs: dict) -> pd.DataFrame:
    """One-row DataFrame in the exact column order the model was trained on."""
    return pd.DataFrame([inputs])[RAW_FEATURES]


def predict(model, inputs: dict) -> float:
    """Churn probability (0-1) for one customer."""
    return float(model.predict_proba(to_frame(inputs))[0, 1])


def range_warnings(inputs: dict, ref: dict) -> list:
    """Warn when a value is outside the range seen in the training data
    (the model has never seen such customers, so the prediction is less reliable)."""
    warnings = []
    for col in ["credit_score", "age", "tenure", "balance", "estimated_salary"]:
        lo, hi = ref["numeric_min"][col], ref["numeric_max"][col]
        if inputs[col] < lo or inputs[col] > hi:
            warnings.append(f"**{col.replace('_', ' ')}** = {inputs[col]:,.0f} is outside the range "
                            f"in the training data ({lo:,.0f} to {hi:,.0f}).")
    return warnings


def reference_inputs(ref: dict) -> dict:
    """A 'typical customer' built from training medians/modes."""
    med, mode = ref["numeric_median"], ref["mode"]
    return {
        "credit_score": int(round(med["credit_score"])), "country": mode["country"],
        "gender": mode["gender"], "age": int(round(med["age"])),
        "tenure": int(round(med["tenure"])), "balance": float(med["balance"]),
        "products_number": int(mode["products_number"]), "credit_card": int(mode["credit_card"]),
        "active_member": int(mode["active_member"]), "estimated_salary": float(med["estimated_salary"]),
    }


def factor_analysis(model, inputs: dict, ref: dict) -> pd.DataFrame:
    """Approximate 'what pushes this customer's risk up or down'.

    For each input we swap ONLY that value for the typical training value and see
    how the predicted probability changes. A positive change means the customer's
    own value raises the risk compared with a typical customer. This is a
    model-based explanation (not a causal one) and ignores interactions.
    """
    typical = reference_inputs(ref)
    p_orig = predict(model, inputs)
    rows = []
    for col in RAW_FEATURES:
        if inputs[col] == typical[col]:
            continue
        tmp = dict(inputs)
        tmp[col] = typical[col]
        rows.append({"feature": col, "your_value": inputs[col], "typical_value": typical[col],
                     "effect_on_probability": p_orig - predict(model, tmp)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.reindex(out["effect_on_probability"].abs().sort_values(ascending=False).index)


def age_group(age: float) -> str:
    return str(pd.cut([age], bins=AGE_BINS, labels=AGE_LABELS)[0])


def segment_table(inputs: dict, ref: dict) -> pd.DataFrame:
    """How customers in the same segments behaved in the training data."""
    seg = ref["segment_churn"]
    keys = {"country": inputs["country"], "gender": inputs["gender"],
            "products_number": inputs["products_number"], "active_member": inputs["active_member"],
            "credit_card": inputs["credit_card"], "age_group": age_group(inputs["age"])}
    rows = []
    for col, val in keys.items():
        info = seg[col].get(str(val))
        if info:
            rows.append({"segment": f"{col.replace('_', ' ')} = {val}",
                         "historical churn rate": info["churn_rate"],
                         "customers in training data": info["count"]})
    return pd.DataFrame(rows)
