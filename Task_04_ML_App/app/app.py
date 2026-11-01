"""Bank Customer Churn Predictor - main page (prediction).

Run from the project root:   streamlit run app/app.py
"""
import app_utils as U            # also puts the project root on sys.path
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")

metrics, ref = U.safe_setup()
model_name = U.sidebar_model_picker(metrics)
model = U.load_model(model_name)
tuned = metrics["models"][model_name]["tuned"]
threshold = tuned["tuned_threshold"]
base_rate = ref["overall_churn_rate_train"]

st.sidebar.divider()
st.sidebar.info(
    "Use the pages above to explore the data, compare models and see which "
    "features matter most.")
st.sidebar.caption("Educational project. Predictions are statistical estimates, not decisions.")

st.title("🏦 Bank Customer Churn Predictor")
st.write(
    "Enter a customer's details and click **Predict**. The app estimates how likely "
    "this customer is to leave the bank, and explains what is driving that estimate.")

typical = U.reference_inputs(ref)

# ----------------------------------------------------------------- input form
with st.form("customer_form"):
    st.subheader("Customer details")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.number_input("Credit score", 300, 900, typical["credit_score"], step=1,
                                       help="Typical range 300-850.")
        country = st.selectbox("Country", ["France", "Germany", "Spain"],
                               index=["France", "Germany", "Spain"].index(typical["country"]))
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True,
                          index=["Female", "Male"].index(typical["gender"]))
    with c2:
        age = st.slider("Age", 18, 100, typical["age"])
        tenure = st.slider("Tenure (years with the bank)", 0, 10, typical["tenure"])
        products = st.selectbox("Number of bank products", [1, 2, 3, 4],
                                index=[1, 2, 3, 4].index(typical["products_number"]))
    with c3:
        balance = st.number_input("Account balance", 0.0, 1_000_000.0, float(round(typical["balance"], 2)),
                                  step=1000.0, format="%.2f")
        salary = st.number_input("Estimated annual salary", 0.0, 1_000_000.0,
                                 float(round(typical["estimated_salary"], 2)), step=1000.0, format="%.2f")
        has_card = st.radio("Has a credit card?", ["Yes", "No"], horizontal=True,
                            index=0 if typical["credit_card"] == 1 else 1)
        is_active = st.radio("Active member?", ["Yes", "No"], horizontal=True,
                             index=0 if typical["active_member"] == 1 else 1)
    submitted = st.form_submit_button("🔮 Predict churn risk", type="primary", width="stretch")

# ----------------------------------------------------------------- prediction
if submitted:
    inputs = {
        "credit_score": int(credit_score), "country": country, "gender": gender, "age": int(age),
        "tenure": int(tenure), "balance": float(balance), "products_number": int(products),
        "credit_card": 1 if has_card == "Yes" else 0, "active_member": 1 if is_active == "Yes" else 0,
        "estimated_salary": float(salary),
    }
    for w in U.range_warnings(inputs, ref):
        st.warning("Unusual input: " + w + " The model has not seen customers like this, "
                   "so treat the result with extra caution.")

    prob = U.predict(model, inputs)
    st.divider()
    st.subheader("Result")
    left, right = st.columns([1, 1])
    with left:
        if prob >= threshold:
            st.error(f"### ⚠️ Likely to churn\nEstimated churn probability: **{prob:.1%}**")
        else:
            st.success(f"### ✅ Likely to stay\nEstimated churn probability: **{prob:.1%}**")
        st.progress(min(max(prob, 0.0), 1.0), text=f"Churn probability {prob:.1%}")
        st.caption(
            f"The model flags a customer as 'likely to churn' when the probability is at least "
            f"**{threshold:.0%}** (chosen to balance precision and recall on the training data). "
            f"For comparison, **{base_rate:.1%}** of customers in the training data churned.")
        rel = prob / base_rate if base_rate else float("nan")
        st.metric("Compared with the average customer", f"{rel:.1f}x the average risk")
    with right:
        st.markdown(f"**Model used:** {model_name}")
        m = tuned["test_default_threshold"]
        st.markdown(
            f"On the held-out test set this model reached ROC-AUC **{m['roc_auc']:.3f}**, "
            f"recall **{m['recall']:.3f}** and precision **{m['precision']:.3f}** "
            f"(at the default 0.5 cut-off).")
        st.caption("Probabilities are model estimates. They are not guaranteed to be perfectly "
                   "calibrated, and the dataset does not state what time window 'churn' refers to.")

    st.subheader("What is driving this prediction?")
    st.caption("Approximate effect of each of this customer's values compared with a typical customer "
               "(model-based, not proof of cause).")
    fa = U.factor_analysis(model, inputs, ref)
    if fa.empty:
        st.info("This customer matches the typical profile on every input, so there is nothing to compare.")
    else:
        top = fa.head(6).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(top) + 1.2))
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in top["effect_on_probability"]]
        ax.barh([f"{r.feature} = {r.your_value:,.0f}" if isinstance(r.your_value, (int, float))
                 else f"{r.feature} = {r.your_value}" for r in top.itertuples()],
                top["effect_on_probability"] * 100, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Change in churn probability (percentage points)\nred = raises risk, green = lowers risk")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("How similar customers behaved (training data)")
    seg = U.segment_table(inputs, ref)
    st.dataframe(seg.style.format({"historical churn rate": "{:.1%}"}), hide_index=True,
                 width="stretch")

    with st.expander("Show the exact input sent to the model"):
        st.dataframe(U.to_frame(inputs), hide_index=True)
else:
    st.info("Fill in the form above and click **Predict churn risk**. "
            "The form starts with a typical customer from the training data.")
