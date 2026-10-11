"""
Streamlit dashboard - Diabetes 130-US Hospitals (1999-2008)
Devixo Solutions AI/ML Internship - Task 01 (bonus)

Run from the Task_01_AI_ML folder:
    streamlit run app/app.py

The app reads the CLEANED dataset created by the notebook (dataset/diabetic_data_cleaned.csv).
All numbers shown here are calculated from that file; nothing is typed in by hand
except the "Key insights" tab, which summarises the notebook analysis.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Page settings and constants
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Diabetes Hospital Readmission Dashboard", page_icon="🏥", layout="wide")

DATA_PATH = Path(__file__).resolve().parent.parent / "dataset" / "diabetic_data_cleaned.csv"

AGE_ORDER = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
AGE_MIDPOINT = dict(zip(AGE_ORDER, range(5, 100, 10)))   # [0-10) -> 5, [10-20) -> 15, ...
PRIOR_ORDER = ["0", "1", "2", "3+"]

# Readable names for the columns shown on the dashboard
NUMERIC_LABELS = {
    "time_in_hospital": "Days in hospital",
    "num_lab_procedures": "Lab procedures",
    "num_procedures": "Procedures",
    "num_medications": "Medications",
    "number_outpatient": "Outpatient visits (previous year)",
    "number_emergency": "Emergency visits (previous year)",
    "number_inpatient": "Inpatient visits (previous year)",
    "number_diagnoses": "Number of diagnoses",
}
GROUP_LABELS = {
    "age": "Age bracket",
    "gender": "Gender",
    "race": "Race",
    "prior_inpatient_group": "Earlier inpatient visits (previous year)",
    "change": "Diabetes medication changed",
    "diabetes_med": "Diabetes medication prescribed",
    "a1c_result": "HbA1c test result",
    "medical_specialty": "Admitting medical specialty",
    "readmitted": "Readmission outcome",
}
ORDINAL_ORDERS = {"age": AGE_ORDER, "prior_inpatient_group": PRIOR_ORDER, "readmitted": ["NO", ">30", "<30"]}
REQUIRED_COLUMNS = (["encounter_id", "patient_id", "race", "gender", "age", "readmitted", "change", "diabetes_med",
                     "a1c_result", "medical_specialty", "number_inpatient"] + list(NUMERIC_LABELS))


# ----------------------------------------------------------------------------
# Data loading (cached, so the file is only read once)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(path: str) -> pd.DataFrame:
    """Read the cleaned CSV and rebuild the helper columns created in the notebook (Part 3)."""
    code_columns = ["primary_diagnosis", "secondary_diagnosis", "additional_diagnosis",
                    "admission_type_id", "discharge_disposition_id", "admission_source_id"]
    # keep_default_na=False: the text "None"/"NA" must stay text, never become missing values
    df = pd.read_csv(path, keep_default_na=False, dtype={c: str for c in code_columns}, low_memory=False)
    df["age"] = pd.Categorical(df["age"], categories=AGE_ORDER, ordered=True)
    df["age_midpoint"] = df["age"].astype(str).map(AGE_MIDPOINT)
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)            # 1 = readmitted within 30 days
    df["prior_inpatient_group"] = np.minimum(df["number_inpatient"], 3).astype(str).replace("3", "3+")
    return df


@st.cache_data
def dataset_facts(path: str) -> dict:
    """Facts about the whole file (not affected by the filters)."""
    header_columns = pd.read_csv(path, nrows=0).shape[1]
    df = load_data(path)
    return {
        "columns": header_columns,
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "unique_patients": int(df["patient_id"].nunique()),
    }


if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: `{DATA_PATH}`\n\n"
        "Run the notebook first. Its last cleaning step saves `diabetic_data_cleaned.csv`. "
        "Place that file in the `dataset/` folder, then reload this page."
    )
    st.stop()

data = load_data(str(DATA_PATH))
missing_columns = [c for c in REQUIRED_COLUMNS if c not in data.columns]
if missing_columns:
    st.error(f"The dataset is missing expected columns: {missing_columns}. Re-create it with the notebook.")
    st.stop()
facts = dataset_facts(str(DATA_PATH))

overall_rate = data["readmitted_30d"].mean() * 100
overall_days = data["time_in_hospital"].mean()
overall_meds = data["num_medications"].mean()

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")
st.sidebar.caption("All charts and numbers update when you change a filter.")

age_sel = st.sidebar.multiselect("Age bracket", AGE_ORDER, default=AGE_ORDER)
gender_sel = st.sidebar.multiselect("Gender", sorted(data["gender"].unique()), default=sorted(data["gender"].unique()))
race_sel = st.sidebar.multiselect("Race", sorted(data["race"].unique()), default=sorted(data["race"].unique()))
change_sel = st.sidebar.multiselect(
    "Diabetes medication changed", ["No", "Ch"], default=["No", "Ch"],
    format_func=lambda v: "No change" if v == "No" else "Changed")
days_sel = st.sidebar.slider("Days in hospital", 1, 14, (1, 14))

mask = (
    data["age"].isin(age_sel)
    & data["gender"].isin(gender_sel)
    & data["race"].isin(race_sel)
    & data["change"].isin(change_sel)
    & data["time_in_hospital"].between(days_sel[0], days_sel[1])
)
view = data[mask]

st.sidebar.markdown("---")
st.sidebar.metric("Visits matching filters", f"{len(view):,}", help=f"Out of {len(data):,} visits in total")

# ----------------------------------------------------------------------------
# Title and KPIs
# ----------------------------------------------------------------------------
st.title("🏥 Diabetes Hospital Readmission Dashboard")
st.caption("Dataset: Diabetes 130-US Hospitals for Years 1999-2008 (Kaggle: brandao/diabetes). "
           "Cleaned version created in the Task 01 notebook.")

if view.empty:
    st.warning("No visits match the current filters. Widen the filters in the sidebar.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Visits in selection", f"{len(view):,}", help=f"Total visits in the dataset: {len(data):,}")
k2.metric("Columns in dataset", facts["columns"])
k3.metric("Average days in hospital", f"{view['time_in_hospital'].mean():.2f}",
          delta=f"{view['time_in_hospital'].mean() - overall_days:+.2f} vs all visits", delta_color="off")
k4.metric("Average medications", f"{view['num_medications'].mean():.1f}",
          delta=f"{view['num_medications'].mean() - overall_meds:+.1f} vs all visits", delta_color="off")
k5.metric("30-day readmission rate", f"{view['readmitted_30d'].mean() * 100:.2f}%",
          delta=f"{view['readmitted_30d'].mean() * 100 - overall_rate:+.2f} pp vs all visits", delta_color="inverse",
          help="Share of visits where the patient came back within 30 days. pp = percentage points.")

tab_overview, tab_dist, tab_rel, tab_readm, tab_stats, tab_insights = st.tabs(
    ["📋 Overview", "📊 Distributions", "🔗 Relationships", "🔁 Readmission", "📈 Statistics", "💡 Key insights"])

# ----------------------------------------------------------------------------
# Tab 1 - Dataset overview
# ----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(data):,}")
    c2.metric("Unique patients", f"{facts['unique_patients']:,}")
    c3.metric("Missing values (cleaned data)", facts["missing"])
    c4.metric("Duplicate rows", facts["duplicates"])
    st.write(
        "Each row is one hospital visit of a patient with diabetes. The data comes from 130 US hospitals "
        "between 1999 and 2008. The same patient can appear in several rows, so visits are not fully independent."
    )
    st.markdown("**First rows of the current selection**")
    st.dataframe(view.head(100))

    st.markdown("**Column summary (whole dataset)**")
    column_summary = pd.DataFrame({
        "column": data.columns,
        "type": [str(t) for t in data.dtypes],
        "distinct values": [data[c].nunique() for c in data.columns],
    })
    st.dataframe(column_summary, hide_index=True)

# ----------------------------------------------------------------------------
# Tab 2 - Distributions
# ----------------------------------------------------------------------------
with tab_dist:
    st.subheader("How are the values distributed?")
    left, right = st.columns(2)

    with left:
        num_col = st.selectbox("Numeric variable", list(NUMERIC_LABELS), format_func=NUMERIC_LABELS.get, key="dist_num")
        fig = px.histogram(view, x=num_col, nbins=int(min(40, view[num_col].nunique())), marginal="box",
                           labels={num_col: NUMERIC_LABELS[num_col]}, template="plotly_white",
                           title=f"Distribution of {NUMERIC_LABELS[num_col].lower()}")
        fig.update_layout(yaxis_title="Number of visits", bargap=0.05)
        st.plotly_chart(fig)
        st.caption(f"Mean {view[num_col].mean():.2f} | Median {view[num_col].median():.1f} | "
                   f"Standard deviation {view[num_col].std():.2f}")

    with right:
        cat_col = st.selectbox("Categorical variable", list(GROUP_LABELS), format_func=GROUP_LABELS.get, key="dist_cat")
        counts = view[cat_col].value_counts()
        if cat_col in ORDINAL_ORDERS:
            counts = counts.reindex([o for o in ORDINAL_ORDERS[cat_col] if o in counts.index])
        counts = counts.head(15)
        fig = px.bar(x=counts.index.astype(str), y=counts.values, template="plotly_white",
                     labels={"x": GROUP_LABELS[cat_col], "y": "Number of visits"},
                     title=f"Visits by {GROUP_LABELS[cat_col].lower()}")
        st.plotly_chart(fig)
        if view[cat_col].nunique() > 15:
            st.caption("Only the 15 most common values are shown.")
        else:
            st.caption(f"Most common value: {view[cat_col].value_counts().idxmax()}")

# ----------------------------------------------------------------------------
# Tab 3 - Relationships
# ----------------------------------------------------------------------------
with tab_rel:
    st.subheader("How are two variables related?")
    s1, s2, s3 = st.columns(3)
    x_col = s1.selectbox("X axis", list(NUMERIC_LABELS), index=0, format_func=NUMERIC_LABELS.get, key="rel_x")
    y_col = s2.selectbox("Y axis", list(NUMERIC_LABELS), index=3, format_func=NUMERIC_LABELS.get, key="rel_y")
    colour_by = s3.selectbox("Colour points by", ["None", "gender", "change", "readmitted"],
                             format_func=lambda c: "None" if c == "None" else GROUP_LABELS[c], key="rel_col")

    sample = view.sample(min(5000, len(view)), random_state=42)
    fig = px.scatter(sample, x=x_col, y=y_col, color=None if colour_by == "None" else colour_by, opacity=0.35,
                     labels={x_col: NUMERIC_LABELS[x_col], y_col: NUMERIC_LABELS[y_col]}, template="plotly_white",
                     title=f"{NUMERIC_LABELS[y_col]} vs {NUMERIC_LABELS[x_col]} (random sample of up to 5,000 visits)")
    st.plotly_chart(fig)

    if x_col == y_col:
        st.info("Both axes show the same variable, so the correlation is 1 by definition.")
    elif len(view) > 2:
        r = view[x_col].corr(view[y_col])
        st.metric("Correlation (all visits in selection)", f"{r:.2f}")
    st.caption("Correlation measures how closely two variables move together in a straight line. "
               "It shows association, not cause.")

# ----------------------------------------------------------------------------
# Tab 4 - Readmission
# ----------------------------------------------------------------------------
with tab_readm:
    st.subheader("Which groups are readmitted most often?")
    r1, r2, r3 = st.columns(3)
    group_col = r1.selectbox("Compare groups of", [g for g in GROUP_LABELS if g != "readmitted"],
                             format_func=GROUP_LABELS.get, key="readm_group")
    metric_name = r2.selectbox("Measure", ["30-day readmission rate (%)", "Average days in hospital", "Average medications"],
                               key="readm_metric")
    min_visits = r3.slider("Minimum visits per group", 1, 2000, 100, step=50,
                           help="Small groups give unreliable averages, so they can be hidden.")

    metric_source = {"30-day readmission rate (%)": ("readmitted_30d", 100),
                     "Average days in hospital": ("time_in_hospital", 1),
                     "Average medications": ("num_medications", 1)}[metric_name]
    grouped = (view.groupby(group_col, observed=True)
               .agg(visits=("readmitted_30d", "size"), value=(metric_source[0], "mean")))
    grouped["value"] = grouped["value"] * metric_source[1]
    grouped = grouped[grouped["visits"] >= min_visits]

    if grouped.empty:
        st.info("No group has enough visits. Lower the minimum visits per group.")
    else:
        if group_col in ORDINAL_ORDERS:
            grouped = grouped.reindex([o for o in ORDINAL_ORDERS[group_col] if o in grouped.index])
        else:
            grouped = grouped.sort_values("value", ascending=False)
        grouped = grouped.reset_index()
        grouped[group_col] = grouped[group_col].astype(str)
        fig = px.bar(grouped, x=group_col, y="value", text=grouped["value"].round(1).astype(str),
                     hover_data={"visits": ":,"}, template="plotly_white",
                     labels={group_col: GROUP_LABELS[group_col], "value": metric_name},
                     title=f"{metric_name} by {GROUP_LABELS[group_col].lower()}")
        reference = {"30-day readmission rate (%)": overall_rate, "Average days in hospital": overall_days,
                     "Average medications": overall_meds}[metric_name]
        fig.add_hline(y=reference, line_dash="dash", line_color="firebrick",
                      annotation_text=f"All visits: {reference:.1f}", annotation_position="top left")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig)
        st.caption("Hover over a bar to see the number of visits in the group. "
                   "Differences between groups are associations; groups with few visits are less reliable.")
        st.dataframe(grouped.rename(columns={group_col: GROUP_LABELS[group_col], "value": metric_name}),
                     hide_index=True)

# ----------------------------------------------------------------------------
# Tab 5 - Statistics and correlation
# ----------------------------------------------------------------------------
with tab_stats:
    st.subheader("Summary statistics")
    numeric_view = view[list(NUMERIC_LABELS)]
    summary = pd.DataFrame({
        "mean": numeric_view.mean(), "median": numeric_view.median(), "std": numeric_view.std(),
        "min": numeric_view.min(), "max": numeric_view.max()}).round(2)
    summary.index = [NUMERIC_LABELS[c] for c in summary.index]
    st.dataframe(summary)

    st.subheader("Correlation heatmap")
    corr_cols = list(NUMERIC_LABELS) + ["age_midpoint", "readmitted_30d"]
    corr_names = list(NUMERIC_LABELS.values()) + ["Age (midpoint)", "Readmitted within 30 days"]
    if len(view) < 30:
        st.info("Too few visits in the current selection for a reliable correlation.")
    else:
        corr = view[corr_cols].corr()
        corr.index = corr_names
        corr.columns = corr_names
        fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                        aspect="auto", template="plotly_white", title="Correlation between numeric variables")
        fig.update_layout(height=650)
        st.plotly_chart(fig)
        st.caption("+1 = move together perfectly, 0 = no straight-line relationship, -1 = move in opposite directions. "
                   "A variable with no variation in the selection appears blank.")

# ----------------------------------------------------------------------------
# Tab 6 - Key insights
# ----------------------------------------------------------------------------
with tab_insights:
    st.subheader("Key insights from the notebook analysis")
    st.caption("These findings come from the full cleaned dataset in the notebook. They do not change with the filters.")
    st.markdown("""
1. **Stays are short.** 62.0% of stays last 1 to 4 days (mean 4.40, median 4 days).
2. **Earlier inpatient visits matter most for early readmission.** The 30-day readmission rate rises from 8.44% (no earlier visit) to 25.65% (three or more).
3. **Overall readmission.** 11.16% of visits were followed by a readmission within 30 days; 53.9% were never readmitted.
4. **Stay length and medications move together** (correlation 0.47, the strongest among numeric pairs).
5. **Age.** Average stay grows from 2.55 days (ages 0-10) to 4.81 days (80-90); average medications peak at ages 60-70.
6. **Medication changes** go with more medications (median 17 vs 13) and a slightly higher readmission rate (11.82% vs 10.59%).
7. **Mostly older patients.** 81.6% of visits are for ages 50-89.
8. **Data quality.** 192,849 hidden `?` placeholders were found and handled; the `weight` column (96.86% missing) was dropped.
""")
    st.info("All of these are associations in this dataset. They do not prove that one variable causes another.")

    st.subheader("Current selection")
    top_group = (view.groupby("age", observed=True)["readmitted_30d"].agg(["mean", "size"]))
    top_group = top_group[top_group["size"] >= 100]
    text = (f"**{len(view):,}** visits are selected. Their 30-day readmission rate is "
            f"**{view['readmitted_30d'].mean() * 100:.2f}%** (all visits: {overall_rate:.2f}%).")
    if not top_group.empty:
        best = top_group["mean"].idxmax()
        text += f" Among age brackets with at least 100 visits, the highest rate in this selection is **{best}** ({top_group['mean'].max() * 100:.1f}%)."
    st.markdown(text)
