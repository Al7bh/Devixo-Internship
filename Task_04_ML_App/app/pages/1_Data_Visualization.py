"""Data visualization page: explore the dataset behind the model."""
import app_utils as U
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Data Visualization", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid")

metrics, ref = U.safe_setup()
df = U.load_dataset()

st.title("📊 Data Visualization")
st.write("Explore the customer data the model was trained on. "
         "Patterns here show **association**, not cause and effect.")

# ---- optional filters ------------------------------------------------------
st.sidebar.header("Filters")
countries = st.sidebar.multiselect("Country", sorted(df["country"].unique()), default=sorted(df["country"].unique()))
genders = st.sidebar.multiselect("Gender", sorted(df["gender"].unique()), default=sorted(df["gender"].unique()))
age_lo, age_hi = st.sidebar.slider("Age range", int(df["age"].min()), int(df["age"].max()),
                                   (int(df["age"].min()), int(df["age"].max())))
view = df[df["country"].isin(countries) & df["gender"].isin(genders) & df["age"].between(age_lo, age_hi)]

if view.empty:
    st.warning("No customers match these filters. Widen the filters in the sidebar.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers shown", f"{len(view):,}")
k2.metric("Churned", f"{int(view['churn'].sum()):,}")
k3.metric("Churn rate", f"{view['churn'].mean():.1%}")
k4.metric("Average age", f"{view['age'].mean():.1f}")

tab1, tab2, tab3, tab4 = st.tabs(["Churn overview", "Churn rate by group", "Numeric features", "Correlation"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        counts = view["churn"].value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(["Stayed", "Churned"], counts.values, color=["#4c72b0", "#c44e52"])
        for b in bars:
            ax.annotate(int(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom")
        ax.set_title("Customers who stayed vs churned")
        st.pyplot(fig); plt.close(fig)
    with c2:
        st.markdown("**Why this matters:** the classes are imbalanced (roughly 4 stay for every 1 churn). "
                    "A model that always says 'stays' would look accurate while catching no churners, so "
                    "this project judges models by recall, F1 and ROC-AUC, not accuracy alone.")

with tab2:
    group_col = st.selectbox("Group customers by", ["country", "gender", "products_number",
                                                    "active_member", "credit_card", "tenure"])
    g = view.groupby(group_col)["churn"].agg(["mean", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=g, x=group_col, y="mean", ax=ax, color="#4c72b0")
    ax.axhline(view["churn"].mean(), color="red", ls="--", lw=1, label="Overall churn rate")
    for i, r in g.iterrows():
        ax.annotate(f"n={int(r['count'])}", (i, r["mean"]), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Churn rate"); ax.set_ylim(0, min(1.1, g["mean"].max() * 1.25 + 0.05)); ax.legend()
    ax.set_title(f"Churn rate by {group_col}")
    st.pyplot(fig); plt.close(fig)
    st.caption("`n` is the number of customers in each bar. Small groups give less reliable rates.")

with tab3:
    num_col = st.selectbox("Numeric feature", ["age", "credit_score", "balance", "estimated_salary", "tenure"])
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=view, x=num_col, hue="churn", bins=30, element="step", stat="density",
                     common_norm=False, ax=ax)
        ax.set_title(f"{num_col} distribution by churn")
        st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=view, x="churn", y=num_col, ax=ax)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Stayed", "Churned"])
        ax.set_title(f"{num_col} by churn (boxplot)")
        st.pyplot(fig); plt.close(fig)

with tab4:
    num = view.drop(columns=["customer_id"]).select_dtypes("number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation between numeric columns")
    st.pyplot(fig); plt.close(fig)
    st.caption("Correlation measures linear association only. A weak value does not prove a feature is "
               "useless, and a strong value does not prove cause.")

with st.expander("Look at the raw data"):
    st.dataframe(view.drop(columns=["customer_id"]).head(200), width="stretch")
