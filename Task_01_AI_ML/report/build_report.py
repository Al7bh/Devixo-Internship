"""
Builds the Task 01 PDF report.

How to use
1. Edit the STUDENT DETAILS block below (name, date, GitHub link).
2. Save your screenshots as PNG files in  report/screenshots/  (file names are shown in the grey boxes of the draft report).
   A slot can hold up to 3 screenshots: name.png, name_2.png, name_3.png.
   (A missing screenshot is shown as a grey box with its file name, so you can see what is still needed.)
3. Run from the Task_01_AI_ML folder:   python report/build_report.py
   The report is saved as report/Task_01_Report.pdf

Numbers in the tables are calculated from dataset/diabetic_data_cleaned.csv.
Charts are taken from the images/ folder (created by the notebook).
"""
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

# =============================== STUDENT DETAILS (edit these) ===============================
STUDENT_NAME = "Muhammad Ali"
INTERNSHIP_DOMAIN = "AI/ML"
SUBMISSION_DATE = date.today().strftime("%B %d, %Y")
GITHUB_URL = "https://github.com/Al7bh/Devixo-Internship"
KAGGLE_URL = "https://www.kaggle.com/datasets/brandao/diabetes"
# ============================================================================================

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
SHOTS = ROOT / "report" / "screenshots"
DATA = ROOT / "dataset" / "diabetic_data_cleaned.csv"
OUT = ROOT / "report" / "Task_01_Report.pdf"

NUM_COLS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
AGE_MID = dict(zip(["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)",
                    "[80-90)", "[90-100)"], range(5, 100, 10)))

# ------------------------------------------------------------------ styles
base = getSampleStyleSheet()
NAVY = colors.HexColor("#1F3A5F")
S = {
    "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=10, leading=14.5, spaceAfter=6),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=NAVY,
                         spaceBefore=14, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, textColor=NAVY,
                         spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "cap": ParagraphStyle("cap", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
                          textColor=colors.HexColor("#444444"), alignment=TA_CENTER, spaceAfter=8),
    "cell": ParagraphStyle("cell", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11),
    "cellb": ParagraphStyle("cellb", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.5, leading=11,
                            textColor=colors.white),
    "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=10, leading=14,
                             leftIndent=14, bulletIndent=2, spaceAfter=3),
    "t_big": ParagraphStyle("t_big", fontName="Helvetica-Bold", fontSize=26, leading=32, alignment=TA_CENTER, textColor=NAVY),
    "t_mid": ParagraphStyle("t_mid", fontName="Helvetica", fontSize=14, leading=20, alignment=TA_CENTER),
    "t_small": ParagraphStyle("t_small", fontName="Helvetica", fontSize=12, leading=18, alignment=TA_CENTER),
}


def md(text):
    """Tiny markup: **bold** and `code`. Everything else is escaped so <, > and & are safe."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font name="Courier" size="9">\1</font>', text)
    return text


def P(text, style="body"):
    return Paragraph(md(text), S[style])


def bullets(items):
    return [Paragraph(md(t), S["bullet"], bulletText="-") for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(md(str(c)), S["cellb"] if (header and i == 0) else S["cell"]) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F2F5F9")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


MISSING = []   # collects everything still missing, printed at the end


def picture(path, caption, max_w=16.5 * cm, max_h=10.5 * cm, what="", as_list=False):
    """Image scaled to fit, or a visible grey box if the file does not exist yet."""
    path = Path(path)
    if path.exists():
        w, h = PILImage.open(path).size
        scale = min(max_w / w, max_h / h)
        items = [Image(str(path), width=w * scale, height=h * scale)] + ([Paragraph(md(caption), S["cap"])] if caption else [])
        return items if as_list else KeepTogether(items)
    MISSING.append(path.name)
    box = Table([[Paragraph(md(f"**SCREENSHOT MISSING:** `{path.name}`<br/>{what}".replace("<br/>", " - ")),
                            S["cell"])]], colWidths=[max_w], rowHeights=[2.2 * cm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#999999")),
                             ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEEEEE")),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    items = [box] + ([Paragraph(md(caption), S["cap"])] if caption else [])
    return items if as_list else KeepTogether(items)


def shot(name, caption, what):
    """One slot can hold up to 3 screenshots: name.png, name_2.png and name_3.png (shown one under the other)."""
    base = SHOTS / name
    found = [p for p in [base, base.with_name(f"{base.stem}_2.png"), base.with_name(f"{base.stem}_3.png")] if p.exists()]
    if len(found) <= 1:
        return picture(base, caption, what=what)
    items = []
    for p in found:
        items += picture(p, "", max_w=16.5 * cm, max_h=7.2 * cm, as_list=True)
        items.append(Spacer(1, 4))
    items.append(Paragraph(md(caption), S["cap"]))
    return KeepTogether(items)


# ------------------------------------------------------------------ data (all numbers come from the cleaned file)
if not DATA.exists():
    raise SystemExit(f"Missing {DATA}. Run the notebook and save diabetic_data_cleaned.csv into dataset/ first.")
df = pd.read_csv(DATA, keep_default_na=False, low_memory=False)
df["age_midpoint"] = df["age"].map(AGE_MID)
df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
df["prior_group"] = np.minimum(df["number_inpatient"], 3).astype(str).replace("3", "3+")
N = len(df)
overall = df["readmitted_30d"].mean() * 100

desc = df[NUM_COLS].agg(["mean", "median", "std", "min", "max"]).T.round(2)
corr = df[NUM_COLS + ["age_midpoint", "readmitted_30d"]].corr()
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().rename("r").to_frame()
pairs["abs"] = pairs["r"].abs()
top_pairs = pairs.sort_values("abs", ascending=False).head(6)
prior = (df.groupby("prior_group")["readmitted_30d"].agg(["mean", "size"]))
prior["mean"] *= 100

W = 17 * cm   # text width

# ------------------------------------------------------------------ story
st = []

# ---- title page
st += [Spacer(1, 3.5 * cm), P("DEVIXO SOLUTIONS", "t_big"), Spacer(1, 0.3 * cm), P("AI/ML Internship Program", "t_mid"),
       Spacer(1, 1.2 * cm), P("**Task 01**", "t_mid"),
       P("Python Programming, Data Analysis & Visualization", "t_mid"), Spacer(1, 2.2 * cm),
       P(f"**Student Name:** {STUDENT_NAME}", "t_small"), P(f"**Internship Domain:** {INTERNSHIP_DOMAIN}", "t_small"),
       P(f"**Submission Date:** {SUBMISSION_DATE}", "t_small"), Spacer(1, 2.5 * cm),
       P("Project: Hospital Readmission Analysis of the Diabetes 130-US Hospitals Dataset", "t_small"), PageBreak()]

# ---- 1-2
st += [P("1. Introduction", "h1"),
       P("This report documents Task 01 of the Devixo Solutions AI/ML Internship Program. It follows a complete data analysis "
         "workflow: loading a real-world dataset, understanding its structure, cleaning it, exploring it, calculating "
         "statistics, creating visualizations and writing insights. A bonus interactive dashboard was built with Streamlit."),
       P("The dataset describes hospital visits of patients with diabetes and records whether each patient returned to the "
         "hospital. The analysis focuses on what the data can say about hospital stays, treatment and readmission."),
       P("2. Objective", "h1"),
       P("The objective is to build practical skills in Python data analysis: dataset loading and exploration, data cleaning, "
         "exploratory data analysis, statistical analysis, visualization, insight generation, technical documentation and "
         "GitHub project organization, and to turn raw data into meaningful, evidence-based conclusions.")]

# ---- 3-6
st += [P("3. Dataset Name", "h1"), P("**Diabetes 130-US Hospitals for Years 1999-2008** (file: `diabetic_data.csv`)."),
       P("4. Dataset Source", "h1"),
       P(f"Kaggle: {KAGGLE_URL}. The data was originally collected from 130 US hospitals between 1999 and 2008 "
         "(Health Facts database) and was published for research on hospital readmission of patients with diabetes. "
         "The documentation file `description.pdf` supplied with the dataset describes the original study."),
       P("5. Dataset Description", "h1"),
       P("Each row is one hospital visit (an encounter) of a patient with diabetes. The columns describe the patient (race, "
         "gender, age bracket), the visit (days in hospital, numbers of lab procedures and medications, diagnoses), the "
         "patient's recent history (outpatient, emergency and inpatient visits in the previous year), diabetes medications, and "
         "the outcome: whether the patient was readmitted. A patient can appear in several rows."),
       P("6. Dataset Size", "h1")]
st += [table([["", "Raw dataset", "Cleaned dataset"],
              ["Rows (visits)", "101,766", f"{N:,}"],
              ["Columns", "50", "49"],
              ["Unique patients", "71,518", f"{df['patient_id'].nunique():,}"],
              ["File size", "about 19 MB", f"about {DATA.stat().st_size / 1e6:.0f} MB"]], [5 * cm, 6 * cm, 6 * cm]),
       Spacer(1, 4)]

# ---- 7
st += [P("7. Features/Columns Description", "h1"),
       P("Column meanings follow the dataset documentation (`description.pdf`). Names are shown as in the cleaned dataset."),
       table([["Group", "Columns", "Meaning"],
              ["Identifiers", "`encounter_id`, `patient_id`", "Unique ID of the visit and of the patient"],
              ["Patient", "`race`, `gender`, `age`", "Demographics; age is stored as 10-year brackets such as [70-80)"],
              ["Visit codes", "`admission_type_id`, `discharge_disposition_id`, `admission_source_id`", "Numeric codes (categories, not quantities) for how the patient arrived and left"],
              ["Hospital stay", "`time_in_hospital`", "Days in hospital (1 to 14)"],
              ["Insurance / doctor", "`payer_code`, `medical_specialty`", "Payer type and specialty of the admitting physician"],
              ["Visit workload", "`num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses`", "Number of lab tests, other procedures, medications and diagnoses recorded during the visit"],
              ["Recent history", "`number_outpatient`, `number_emergency`, `number_inpatient`", "Visits by the patient in the year before this visit"],
              ["Diagnoses", "`primary_diagnosis`, `secondary_diagnosis`, `additional_diagnosis`", "ICD-9 diagnosis codes (renamed from diag_1, diag_2, diag_3)"],
              ["Lab results", "`max_glucose_serum`, `a1c_result`", "Blood test results; 'Not tested' means the test was not done"],
              ["Medications", "23 columns (for example `metformin`, `insulin`)", "Whether each drug was prescribed and if the dose changed (No, Steady, Up, Down)"],
              ["Treatment flags", "`change`, `diabetes_med`", "Whether diabetes medication was changed; whether any was prescribed"],
              ["Outcome", "`readmitted`", "NO = not readmitted, >30 = readmitted after 30 days, <30 = readmitted within 30 days"]],
             [3 * cm, 6.2 * cm, 7.8 * cm])]

# ---- 8
st += [P("8. Methodology", "h1"),
       P("The work followed the required workflow, using Python 3, Pandas, NumPy, Matplotlib and Seaborn in a Jupyter notebook "
         "(Google Colab), and Streamlit with Plotly for the dashboard:")]
st += bullets(["**Data loading and understanding:** load the CSV, inspect rows, columns, data types and missing values.",
               "**Data cleaning:** handle hidden missing values, duplicates, data types, column names and invalid values on a copy of the data, with a log of every action.",
               "**Exploratory analysis and statistics:** descriptive statistics, correlation and group comparisons.",
               "**Visualization:** 11 charts, each with a written interpretation.",
               "**Insights and conclusion:** 14 insights, each based on a number from the notebook.",
               "**Bonus:** interactive Streamlit dashboard connected to the cleaned dataset."])

# ---- 9
st += [P("9. Data Loading", "h1"),
       P("The CSV was loaded into a Pandas DataFrame with `keep_default_na=False`. By default Pandas converts the text 'None' "
         "into a missing value, but in this dataset 'None' in `max_glu_serum` and `A1Cresult` means the test was not performed, "
         "which is real information. An integrity check confirms the file has the expected 101,766 rows and 50 columns before the "
         "analysis continues."),
       P("The first inspection showed that `isna()` found **0** missing values, but 192,849 cells contained the placeholder `?` in 7 "
         "columns (for example `weight`, 96.86% of rows). Pandas does not count `?` as missing, so this had to be handled explicitly.")]
st += [P("*Screenshots: code in Figure 14.1; outputs in Figures 15.1 and 15.2.*".replace("*", ""))]

# ---- 10
st += [P("10. Data Cleaning", "h1"),
       P("All cleaning was done on a copy (`df_clean`); the raw data was never changed. Every action is recorded in a cleaning log:")]
st += [table([["Step", "Action", "Affected", "Reason"],
              ["2.1", "Converted `?` placeholders to missing values", "192,849 cells", "Pandas cannot count `?` as missing"],
              ["2.2", "Dropped column `weight`", "96.86% missing", "Almost empty; filling would invent data"],
              ["2.2", "Filled `medical_specialty` with 'Unknown'", "49,949 (49.08%)", "A mode would invent a specialty for half the patients"],
              ["2.2", "Filled `payer_code` with 'Unknown'", "40,256 (39.56%)", "Category; cannot be averaged"],
              ["2.2", "Filled `race` with 'Unknown'", "2,273 (2.23%)", "Guessing a race with the mode would be wrong"],
              ["2.2", "Filled `diag_2`, `diag_3` with 'Not recorded'", "358 and 1,423", "Most of these patients had very few diagnoses"],
              ["2.2", "Removed rows with missing `diag_1`", "21 rows", "Primary diagnosis is essential; a negligible loss"],
              ["2.3", "Checked duplicates", "0 found", "No identical rows and no repeated encounter_id"],
              ["2.4", "Converted codes, text and age to category types", "39 columns", "Codes are not quantities; age brackets have an order"],
              ["2.5", "Renamed 12 columns (for example diag_1 to primary_diagnosis)", "12 columns", "Clearer, consistent names"],
              ["2.6", "Removed rows with gender 'Unknown/Invalid'", "3 rows", "Not a valid category; cannot be inferred"],
              ["2.6", "Outliers inspected but kept", "see EDA", "Plausible hospital data; removing them would bias results"]],
             [1.3 * cm, 6.3 * cm, 3.4 * cm, 6 * cm]), Spacer(1, 6),
       P("**Result:** 24 rows (0.024%) and 1 column were removed. The cleaned dataset has 101,742 rows, 49 columns, 0 missing values and 0 duplicate rows. "
         "Repeated patients (71,509 patients in 101,745 visits at that stage) were kept because each row is a different hospital stay; this is "
         "noted as a limitation. The meaning of the numeric code columns was not checked against the ID-mapping file, which is also a limitation."),
       P("Screenshots: code in Figure 14.2; outputs in Figures 15.3 to 15.5.")]

# ---- 11
st += [P("11. Exploratory Data Analysis", "h1"),
       P("The EDA looked at what the data contains, the most common categories, extreme values and relationships:")]
st += bullets([f"The typical visit lasts about {df['time_in_hospital'].median():.0f} days with about {df['num_medications'].median():.0f} medications.",
               "The most common categories are Caucasian (74.8%), female (53.8%), age 70 to 80 (25.6%) and 'not readmitted' (53.9%).",
               "49.1% of visits have an unknown specialty and 39.6% an unknown payer, which limits analysis of those columns.",
               "Extreme values exist (up to 81 medications, 132 lab procedures, 76 emergency visits). They were checked and kept: the highest "
               "medication counts occur in long stays, and the five highest emergency counts all belong to one patient.",
               "The three visit-count columns are very skewed: most patients had no earlier visits (median 0) but a few had many."])

# ---- 12
st += [P("12. Statistical Analysis", "h1"), P("Descriptive statistics of the numeric columns (cleaned data):")]
rows = [["Variable", "Mean", "Median", "Std. dev.", "Min", "Max"]]
for c in NUM_COLS:
    r = desc.loc[c]
    rows.append([f"`{c}`", f"{r['mean']:.2f}", f"{r['median']:.1f}", f"{r['std']:.2f}", f"{r['min']:.0f}", f"{r['max']:.0f}"])
st += [table(rows, [5.5 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm, 2 * cm, 2.5 * cm]), Spacer(1, 6),
       P("For the three visit-count columns the median is 0 while the mean is between 0.20 and 0.64, because a small number of patients "
         "with many visits pull the mean up. This is an example of why the median is more useful than the mean when extreme values exist."),
       P("**Strongest correlations** (Pearson, association only):")]
rows = [["Pair", "Correlation"]]
for (a, b), r in top_pairs["r"].items():
    rows.append([f"`{a}` and `{b}`", f"{r:.2f}"])
st += [table(rows, [11 * cm, 3 * cm]), Spacer(1, 6),
       P(f"No pair exceeds 0.5. The correlation of 30-day readmission with earlier inpatient visits is {corr.loc['number_inpatient', 'readmitted_30d']:.2f} "
         "(weak); all other variables are 0.06 or lower. The group comparison below shows a clearer pattern:"),
       table([["Earlier inpatient visits", "Visits", "30-day readmission rate"]] +
             [[g, f"{int(prior.loc[g, 'size']):,}", f"{prior.loc[g, 'mean']:.2f}%"] for g in ["0", "1", "2", "3+"]],
             [6 * cm, 4 * cm, 6 * cm]), Spacer(1, 6),
       P("Screenshots: code in Figure 14.3; outputs in Figures 15.6 to 15.8.")]

# ---- 13
CHARTS = [
 ("visualization_01_histogram_time_in_hospital.png", "Visualization 1 - Histogram of hospital stay length",
  "Most stays are short: 62.0% last 1 to 4 days. The mean (4.40) is above the median (4) because of a long tail of lengthy stays."),
 ("visualization_02_bar_readmission_by_prior_visits.png", "Visualization 2 - Readmission rate by earlier inpatient visits",
  "The 30-day readmission rate rises from 8.4% with no earlier inpatient visit to 25.6% with three or more, roughly three times higher (an association, not a cause)."),
 ("visualization_03_pie_readmission_outcome.png", "Visualization 3 - Readmission outcome",
  "53.9% were not readmitted, 34.9% were readmitted after 30 days and 11.2% within 30 days: about 1 visit in 9."),
 ("visualization_04_scatter_stay_vs_medications.png", "Visualization 4 - Hospital stay vs number of medications",
  "Longer stays go with more medications (correlation 0.47, about 1.27 more medications per extra day), but the spread is wide."),
 ("visualization_05_line_trend_across_age.png", "Visualization 5 - Stay length and medications across age groups",
  "Average stay rises from 2.55 days (ages 0-10) to 4.81 days (80-90); average medications peak at ages 60-70 and then fall. This is a trend across age, not time."),
 ("visualization_06_box_medications_by_change.png", "Visualization 6 - Medications by medication change",
  "Visits with a medication change have a higher median (17 vs 13). Both groups have many high values that are plausible long stays."),
 ("visualization_07_heatmap_correlation.png", "Visualization 7 - Correlation heatmap",
  "No pair of variables is strongly correlated. Stay length and medications are the strongest pair (0.47); readmission is only weakly linked to earlier inpatient visits (0.17)."),
 ("visualization_08_countplot_age.png", "Visualization 8 - Visits by age bracket",
  "Visits are concentrated in older adults: ages 50 to 89 make up 81.6%, and only 161 visits are for ages 0-10."),
 ("visualization_09_pairplot_numeric.png", "Visualization 9 - Pair plot of key numeric variables",
  "Apart from stay and medications, the variables are weakly connected. Diagnoses pile up at 9 (48.6% of visits), possibly a recording limit (not verified)."),
 ("visualization_10_violin_diagnoses_by_age.png", "Visualization 10 - Diagnoses by age group",
  "Median diagnoses rise from 6 (under 40) to 9 (60 and over). Older groups look alike partly because many visits sit at the maximum of 9."),
 ("visualization_11_bar_readmission_by_specialty.png", "Visualization 11 (extra) - Readmission by specialty",
  "Nephrology has the highest rate (15.3%) and Orthopedics-Reconstructive the lowest (7.5%), but only 9 specialties qualify and about half of visits have an unknown specialty."),
]
st += [P("13. Data Visualizations", "h1"),
       P("Eleven charts were created (the 10 required types plus one extra) and saved in the `images/` folder. Each is shown with its interpretation.")]
for fname, title, text in CHARTS:
    st.append(KeepTogether([P(title, "h2")] +
                           picture(IMAGES / fname, "", max_w=15.5 * cm, max_h=8.8 * cm,
                                   what="Run the notebook to create this chart", as_list=True) +
                           [P(text)]))

# ---- 14-16
st += [PageBreak(), P("14. Code Screenshots", "h1"),
       P("Screenshots of the main Python code from the notebook:"),
       shot("code_01_imports_loading.png", "Figure 14.1: imports, loading and integrity check", "Section 1.1-1.2 code cells"),
       shot("code_02_cleaning.png", "Figure 14.2: data cleaning code", "Section 2.1-2.2 code cells"),
       shot("code_03_statistics.png", "Figure 14.3: statistical analysis code", "Section 3.1-3.6 code cells"),
       shot("code_04_visualization.png", "Figure 14.4: visualization code (example)", "One chart cell, for example Visualization 2"),
       P("15. Output Screenshots", "h1"),
       P("Screenshots of the notebook outputs, in the order of the analysis:"),
       shot("output_01_head_shape.png", "Figure 15.1: first rows, last rows and shape of the dataset", "Output of head(), tail(), sample() and shape (section 1.3-1.5)"),
       shot("output_02_info_missing.png", "Figure 15.2: dataset information and hidden missing values", "Output of info() and the '?' count table (section 1.4)"),
       shot("output_03_missing_table.png", "Figure 15.3: missing values per column", "Missing-values table from section 2.1"),
       shot("output_04_cleaning_log.png", "Figure 15.4: cleaning log", "Cleaning log table from section 2.7"),
       shot("output_05_verification.png", "Figure 15.5: verification of the cleaned dataset", "Verification output from section 2.7 (shape, missing, duplicates, dtypes)"),
       shot("output_06_descriptive_stats.png", "Figure 15.6: descriptive statistics", "Descriptive statistics table from section 3.1"),
       shot("output_07_correlation.png", "Figure 15.7: correlation analysis", "Correlation matrix and strongest pairs from section 3.6"),
       shot("output_08_group_comparisons.png", "Figure 15.8: group comparisons", "Group comparison tables from section 3.7"),
       shot("output_09_insight_numbers.png", "Figure 15.9: numbers behind the insights", "Output of the first cell in Part 5"),
       P("16. Graph Screenshots", "h1"),
       P("The 11 notebook charts are shown in section 13. The following screenshots show the Streamlit dashboard (bonus), "
         "which is in `app/app.py` and reads the cleaned dataset."),
       shot("dashboard_01_overview.png", "Figure 16.1: dashboard - KPIs and filters", "Whole page with KPI row and sidebar filters"),
       shot("dashboard_02_distributions.png", "Figure 16.2: dashboard - Distributions tab", "Distributions tab"),
       shot("dashboard_03_relationships.png", "Figure 16.3: dashboard - Relationships tab", "Relationships tab"),
       shot("dashboard_04_readmission.png", "Figure 16.4: dashboard - Readmission tab", "Readmission tab"),
       shot("dashboard_05_statistics.png", "Figure 16.5: dashboard - correlation heatmap", "Statistics tab with the heatmap"),
       shot("dashboard_06_insights.png", "Figure 16.6: dashboard - Key insights tab", "Key insights tab")]

# ---- 17
st += [PageBreak(), P("17. Major Findings", "h1")]
st += bullets([f"The dataset has {N:,} cleaned visits; 11.16% ended in a readmission within 30 days and 53.9% in no readmission.",
               "The clearest pattern is earlier inpatient visits: the 30-day readmission rate rises from 8.44% (none) to 25.65% (three or more).",
               "Hospital stays are short (62.0% last 1 to 4 days), and stay length is moderately related to the number of medications (0.47).",
               "No numeric variable is strongly related to readmission; the highest correlation is 0.17.",
               "The patients are mainly older adults (81.6% are aged 50 to 89).",
               "The raw data hid 192,849 missing values behind `?` placeholders, and `weight` was missing in 96.86% of rows.",
               "**Limitations:** repeated patients, unverified meaning of some code columns, and about half of the specialty values being unknown. "
               "All findings are associations, not causes."])

# ---- 18
INSIGHTS = [
 ("Most stays are short.", "62.0% of stays last 1 to 4 days; mean 4.40, median 4 (Visualization 1)."),
 ("Earlier inpatient visits are the clearest warning sign for early readmission.", "Rate rises from 8.44% to 25.65% (Visualization 2)."),
 ("Quick readmission is the minority outcome.", "11.16% within 30 days; 53.9% never readmitted (Visualization 3)."),
 ("Longer stays go with more medications.", "Correlation 0.47, the strongest among numeric pairs (Visualizations 4 and 7)."),
 ("Stay length rises with age, but medications peak at 60-70.", "Average stay 2.55 to 4.81 days; medications peak at 17.15 (Visualization 5)."),
 ("Visits with a medication change are more complex.", "Median medications 17 vs 13; readmission 11.82% vs 10.59% (Visualization 6)."),
 ("The patients are mostly older adults.", "81.6% are aged 50 to 89; only 161 visits are aged 0-10 (Visualization 8)."),
 ("No numeric variable is strongly related to readmission.", "Highest correlation 0.165; no pair above 0.5 (Visualization 7)."),
 ("Readmission differs by specialty, but data is incomplete.", "Nephrology 15.32% vs Orthopedics-Reconstructive 7.46%; 49.1% of visits have an unknown specialty (Visualization 11)."),
 ("Most patients had no HbA1c test.", "83.3% untested; readmission 11.42% without the test vs 9.84% with it (association only)."),
 ("One patient drives the extreme emergency-visit values.", "The five highest counts (46 to 76) belong to one patient; coefficient of variation 4.70."),
 ("Nearly half of the visits have exactly 9 diagnoses.", "48.6%; possibly a recording ceiling (not verified)."),
 ("Gender makes almost no difference to readmission.", "11.24% for females vs 11.06% for males."),
 ("The raw data hid a lot of missing information.", "192,849 `?` cells; `weight` missing in 96.86%."),
]
rows = [["#", "Insight", "Evidence"]] + [[str(i + 1), f"**{a}**", b] for i, (a, b) in enumerate(INSIGHTS)]
st += [P("18. At Least 10 Insights", "h1"),
       P("Every insight below comes from the results in the notebook and describes an association in this dataset, not a cause."),
       table(rows, [0.9 * cm, 7.6 * cm, 8.5 * cm])]

# ---- 19-20
st += [P("19. Conclusion", "h1"),
       P("For this task I analysed the Diabetes 130-US Hospitals for Years 1999-2008 dataset from Kaggle, which describes 101,766 hospital "
         "visits by patients with diabetes and records whether each patient was readmitted. I loaded the data, checked its structure and found "
         "that 192,849 cells held hidden `?` placeholders that Pandas does not count as missing. I cleaned the data by converting these placeholders, "
         "dropping the almost-empty `weight` column, labelling unknown categories, removing 24 rows (0.024%) with no primary diagnosis or an invalid "
         "gender, converting codes and age brackets to proper categories, and renaming 12 columns. Each decision was recorded in a cleaning log. "
         "I then calculated the mean, median, mode, range, standard deviation, variance and correlations, compared groups, and created 11 visualizations, "
         "each with a written interpretation."),
       P("The most important findings are that most hospital stays are short (62.0% last 1 to 4 days), that 11.16% of visits ended in a readmission "
         "within 30 days, and that the number of earlier inpatient visits is the clearest pattern: the readmission rate rises from 8.44% to 25.65%. "
         "Stay length and medications are moderately related (r = 0.47), but no numeric variable is strongly related to readmission. These results are "
         "associations, not causes, and have limits: some patients appear many times, the meaning of some code columns was not checked, and about half of "
         "the specialty values are unknown. From this project I learned to check for hidden missing values and verify the data after loading, to choose "
         "the median when values are skewed, to explain every cleaning decision, and to describe results carefully instead of claiming causes. The analysis "
         "suggests that hospitals could give extra attention to patients with several recent admissions when planning follow-up care, though that would "
         "need further study to confirm."),
       P("20. GitHub Repository Link", "h1"), P(f"{GITHUB_URL}"),
       P("The repository contains the notebook, the raw and cleaned datasets, the 11 visualization images, the Streamlit application, this report and the README.")]


# ------------------------------------------------------------------ build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "Devixo Solutions - AI/ML Internship Program - Task 01")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                        title="Task 01 - Python Programming, Data Analysis & Visualization", author=STUDENT_NAME)
doc.build(st, onFirstPage=footer, onLaterPages=footer)

print(f"Report written: {OUT}")
if STUDENT_NAME in ("Your Name", ""):
    print("WARNING: set STUDENT_NAME at the top of this script.")
if MISSING:
    print(f"WARNING: {len(MISSING)} image(s) still missing:")
    for name in sorted(set(MISSING)):
        print("  -", name)
