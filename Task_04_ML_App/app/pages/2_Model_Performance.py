"""Model performance page - everything is read from models/metrics.json."""
import app_utils as U
import pandas as pd
import streamlit as st

from src import plots

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

metrics, ref = U.safe_setup()
model_name = U.sidebar_model_picker(metrics)
meta = metrics["meta"]
models = metrics["models"]

st.title("📈 Model Performance")
st.write(
    f"Five models were trained on **{meta['n_train']:,}** customers and tested on **{meta['n_test']:,}** "
    f"unseen customers. Each model was tuned, and the best one was chosen using cross-validation on the "
    f"training data only, so the test set stayed untouched until the final report.")
st.success(f"**Best model: {metrics['best_model']}.** {metrics['selection_reason']}")

# ---------------------------------------------------------------- comparison table
st.subheader("1. Model comparison (test set, threshold 0.5)")
rows = []
for n in metrics["model_order"]:
    m = models[n]["tuned"]["test_default_threshold"]
    rows.append({"Model": n, "Accuracy": m["accuracy"], "Precision": m["precision"],
                 "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"],
                 "CV ROC-AUC (train)": models[n]["tuned"]["cv_roc_auc_mean"]})
table = pd.DataFrame(rows)
st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns if c != "Model"})
             .highlight_max(subset=[c for c in table.columns if c != "Model"],
                            props="background-color: #1b5e20; color: #ffffff; font-weight: 700;"),
             hide_index=True, width="stretch")
st.caption(f"Always predicting 'stays' would score **{meta['majority_class_accuracy_test']:.1%}** accuracy "
           "but catch no churners, so accuracy alone is misleading here. Dark green = best value in the column.")
st.pyplot(plots.model_comparison(metrics))

# ---------------------------------------------------------------- tuning
st.subheader("2. Before vs after hyperparameter tuning")
tune_rows = []
for n in metrics["model_order"]:
    b, t = models[n]["baseline"], models[n]["tuned"]
    tune_rows.append({
        "Model": n, "Search": t["search_type"], "Candidates": t["n_candidates"],
        "CV AUC before": b["cv_roc_auc_mean"], "CV AUC after": t["cv_roc_auc_mean"],
        "Test AUC before": b["test_default_threshold"]["roc_auc"],
        "Test AUC after": t["test_default_threshold"]["roc_auc"],
        "Test F1 before": b["test_default_threshold"]["f1"],
        "Test F1 after": t["test_default_threshold"]["f1"],
        "Best parameters": ", ".join(f"{k}={v}" for k, v in t["best_params"].items()),
    })
tdf = pd.DataFrame(tune_rows)
fmt = {c: "{:.3f}" for c in tdf.columns if "AUC" in c or "F1" in c}
st.dataframe(tdf.style.format(fmt), hide_index=True, width="stretch")
st.caption("Tuning is not guaranteed to help. Where the 'after' column is no better than 'before', "
           "that is a real result and is reported as it is.")
st.pyplot(plots.tuning_before_after(metrics, "roc_auc"))

# ---------------------------------------------------------------- selected model detail
st.subheader(f"3. Details for: {model_name}")
t = models[model_name]["tuned"]
d, tt = t["test_default_threshold"], t["test_tuned_threshold"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{d['accuracy']:.3f}")
c2.metric("Precision", f"{d['precision']:.3f}")
c3.metric("Recall", f"{d['recall']:.3f}")
c4.metric("F1", f"{d['f1']:.3f}")
c5.metric("ROC-AUC", f"{d['roc_auc']:.3f}")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.pyplot(plots.confusion_matrix_plot(metrics, model_name, "default"))
    st.caption("Threshold 0.5 (the standard default).")
with col_b:
    st.pyplot(plots.confusion_matrix_plot(metrics, model_name, "tuned"))
    st.caption(f"Threshold {t['tuned_threshold']:.2f}: chosen to maximise F1 on the training data "
               f"(used by the prediction page). Recall {tt['recall']:.3f}, precision {tt['precision']:.3f}.")
with col_c:
    st.pyplot(plots.roc_curves(metrics, highlight=model_name))

with st.expander("Tuned hyperparameters"):
    st.json(t["best_params"])
with st.expander("How to read a confusion matrix"):
    st.markdown("Rows are the **actual** outcome, columns the **predicted** outcome. "
                "Top-left = correctly predicted to stay, bottom-right = churners correctly caught, "
                "top-right = false alarms, bottom-left = churners the model missed.")

# ---------------------------------------------------------------- calibration
ea = U.load_error_analysis()
if ea:
    st.subheader(f"4. Can the probabilities be trusted? ({ea['model']}, test set)")
    cal = pd.DataFrame(ea["calibration"]).rename(columns={
        "bin": "Predicted probability", "n": "Customers",
        "mean_predicted": "Mean predicted", "observed_rate": "Observed churn rate"})
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(plots.calibration_plot(ea))
    with c2:
        st.dataframe(cal.style.format({"Mean predicted": "{:.1%}", "Observed churn rate": "{:.1%}"}, na_rep="-"),
                     hide_index=True, width="stretch")
        st.caption(f"Brier score: {ea['brier_score']:.3f} (lower is better). If 'mean predicted' and 'observed' are close, "
                   "the probability can be read as a rough frequency. Bins with few customers are noisy. "
                   "This analysis was run for the best model only.")

# ---------------------------------------------------------------- feature engineering
ablation = U.load_ablation()
if ablation:
    st.subheader("5. Did the engineered features help?")
    st.pyplot(plots.feature_ablation_plot(ablation))
    st.caption("5-fold cross-validated ROC-AUC on the training data. Differences smaller than the error "
               "bars should be treated as noise.")

st.caption(f"Trained with scikit-learn {meta['sklearn']} on Python {meta['python']}.")
