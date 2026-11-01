"""Feature importance page (permutation importance, computed during training)."""
import app_utils as U
import pandas as pd
import streamlit as st

from src import plots

st.set_page_config(page_title="Feature Importance", page_icon="🔎", layout="wide")

metrics, ref = U.safe_setup()
model_name = U.sidebar_model_picker(metrics)

st.title("🔎 Feature Importance")
st.write(
    "Which customer details matter most to the model? This page uses **permutation importance**: "
    "one column at a time is randomly shuffled on the test set, and we measure how much the model's "
    "ROC-AUC drops. A bigger drop means the model relied more on that column.")

st.pyplot(plots.feature_importance(metrics, model_name, top_n=10))

imp = metrics["models"][model_name]["tuned"]["permutation_importance"]
table = (pd.DataFrame(imp).T.rename(columns={"mean": "Importance (AUC drop)", "std": "Std over repeats"})
         .sort_values("Importance (AUC drop)", ascending=False))
st.dataframe(table.style.format("{:.4f}"), width="stretch")

st.subheader("Compare across models")
comp = pd.DataFrame({n: {k: v["mean"] for k, v in
                         metrics["models"][n]["tuned"]["permutation_importance"].items()}
                     for n in metrics["model_order"]})
comp = comp.loc[comp.mean(axis=1).sort_values(ascending=False).index]
st.dataframe(comp.style.format("{:.4f}").background_gradient(cmap="Greens", axis=None),
             width="stretch")

st.info(
    "**How to read this carefully.** Importance shows what the *model* relies on, not what *causes* "
    "customers to leave. Features that are related to each other (for example balance and "
    "zero-balance) can share importance. Values near zero mean the model barely used that column; "
    "tiny negative values are just noise.")
