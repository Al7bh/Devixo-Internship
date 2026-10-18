"""
Builds the Task 02 PDF report.

How to use
1. Run the whole notebook (notebook/Task2_Model_Comparison.ipynb) in Colab, then download it with its outputs
   and save it in notebook/ (replace the old file).
2. Edit the STUDENT DETAILS block below if needed (name, GitHub link).
3. Run from the Task_02_ML folder:   python report/build_report_task2.py
   Optional:  python report/build_report_task2.py path/to/notebook.ipynb path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
   The report is saved as report/Task_02_Report.pdf

Where the numbers come from (nothing is typed in by hand)
- Model results, best parameters, charts and the Advantages/Limitations notes are read from the SAVED OUTPUTS and
  code of the notebook. If you re-run the notebook and rebuild, the report updates itself.
- Dataset facts (rows, blanks, duplicates, class balance, statistics) are calculated from the original CSV.
- Counts of true/false positives are derived from Precision, Recall and the number of churners in the test set
  (they match the confusion-matrix charts).
The written explanations (interpretation, conclusion, limitations) are the author's reasoning. If a re-run changes a
result noticeably, re-read those paragraphs.
"""
import base64
import json
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

# =============================== STUDENT DETAILS (edit these) ===============================
STUDENT_NAME = "Muhammad Ali"
INTERNSHIP_DOMAIN = "AI/ML"
SUBMISSION_DATE = date.today().strftime("%B %d, %Y")
GITHUB_URL = "https://github.com/Al7bh/Devixo-Internship"
GITHUB_FOLDER = "Task_02_ML"          # folder of this task inside the repository
KAGGLE_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
# ============================================================================================

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebook" / "Task2_Model_Comparison.ipynb"
CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = ROOT / "report" / "Task_02_Report.pdf"
TMP = Path(tempfile.mkdtemp(prefix="task02_report_"))

for p in (NOTEBOOK, CSV):
    if not p.exists():
        raise SystemExit(f"Missing {p}. Put the executed notebook in notebook/ and the Kaggle CSV in dataset/.")


# ------------------------------------------------------------------ read the notebook (single source of truth)
def _num(s):
    try:
        return float(s) if re.search(r"[.eE]", s) else int(s)
    except ValueError:
        return s


def parse_df_text(text):
    """Turn the text/plain of a displayed DataFrame (possibly wrapped over several blocks) into a DataFrame."""
    blocks, cur = [], []
    for line in text.split("\n"):
        line = re.sub(r"\s*\\$", "", line.rstrip())
        if line.strip():
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    parts = []
    for b in blocks:
        header = re.split(r"\s{2,}", b[0].strip())
        rows = {}
        for line in b[1:]:
            toks = line.split()
            idx, rest, vals = int(toks[0]), toks[1:], []
            for j, c in enumerate(header):
                if c == "Model":       # may contain spaces
                    need = sum(1 for k in header[j + 1:] if k != "Best params")
                    take = len(rest) - need
                    vals.append(" ".join(rest[:take]))
                    rest = rest[take:]
                elif c == "Best params":
                    vals.append(" ".join(rest))
                    rest = []
                else:
                    vals.append(_num(rest[0]))
                    rest = rest[1:]
            if rest:
                raise ValueError(f"cannot parse row {line!r}")
            rows[idx] = vals
        parts.append(pd.DataFrame.from_dict(rows, orient="index", columns=header))
    return pd.concat(parts, axis=1)


nb = json.load(open(NOTEBOOK, encoding="utf-8"))
CODE = [c for c in nb["cells"] if c["cell_type"] == "code"]
if len(CODE) != 11:
    raise SystemExit(f"Expected 11 code cells in the notebook, found {len(CODE)}.")
for i, marker in {0: "train_test_split", 1: "eda_overview", 2: "DummyClassifier", 3: "GridSearchCV", 4: "before_after",
                  5: "median_fit_time", 6: "MonthlyCharges", 7: "DummyRegressor", 8: "tuned_conf_mats",
                  9: "feature_importances_", 10: "clf_notes"}.items():
    assert marker in "".join(CODE[i]["source"]), f"code cell {i} is not the expected cell (looking for {marker})"


def cell_out(i):
    res = {"stream": "", "text": [], "png": []}
    for o in CODE[i].get("outputs", []):
        if "text" in o:
            res["stream"] += "".join(o["text"])
        elif "data" in o:
            if "text/plain" in o["data"]:
                res["text"].append("".join(o["data"]["text/plain"]))
            if "image/png" in o["data"]:
                res["png"].append(o["data"]["image/png"])
    return res


def save_png(i, name):
    data = cell_out(i)["png"]
    assert data, f"no image in code cell {i}"
    path = TMP / f"{name}.png"
    path.write_bytes(base64.b64decode(data[0]))
    return path


base = parse_df_text(cell_out(2)["text"][0])            # baseline classification (incl. Dummy)
tuned = parse_df_text(cell_out(3)["text"][0])           # tuned classification
comp = parse_df_text(cell_out(5)["text"][0])            # before/after with FAIR fit times
reg = parse_df_text(cell_out(7)["text"][0])             # regression baselines (incl. Dummy)
best_params = {}
for line in cell_out(3)["stream"].strip().split("\n"):
    m = re.match(r"(.+?): done in ([\d.]+)s -> (\{.*\})", line)
    best_params[m.group(1)] = m.group(3).replace("model__", "")
assert set(best_params) == set(tuned["Model"]), "best-parameter lines do not match the tuned table"

# feature importance (top 10)
imp = []
for line in cell_out(9)["text"][0].split("\n"):
    m = re.match(r"^(\S.*?)\s{2,}([\d.]+)$", line.rstrip())
    if m:
        imp.append((m.group(1), float(m.group(2))))
assert len(imp) == 10, "could not read the 10 feature importances"

# EDA numbers (cell 1 text output) and split numbers (cell 0 text output)
eda_txt = cell_out(1)["stream"]
contract_rate = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(Month-to-month|One year|Two year)\s+([\d.]+)$", eda_txt, re.M)}
tenure_median = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"^([01])\s+([\d.]+)$", eda_txt, re.M)}
split_txt = cell_out(0)["stream"]
n_train, n_test = map(int, re.search(r"Train: \((\d+), \d+\) \| Test: \((\d+), \d+\)", split_txt).groups())
rate_train, rate_test = map(float, re.search(r"Churn rate\s+train: ([\d.]+) \| test: ([\d.]+)", split_txt).groups())
reg_txt = cell_out(6)["stream"]
price_by_internet = {}
for m in re.finditer(r"^(DSL|Fiber optic|No)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", reg_txt, re.M):
    price_by_internet[m.group(1)] = dict(count=int(m.group(2)), mean=float(m.group(3)), std=float(m.group(4)),
                                         min=float(m.group(5)), max=float(m.group(6)))
price_stats = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(mean|std|min|max)\s+([\d.]+)$", reg_txt, re.M)}
assert contract_rate and tenure_median and price_by_internet and price_stats, "could not read EDA/regression numbers"

# notes written in the notebook (Advantages / Limitations)
src10 = "".join(CODE[10]["source"])
notes_ns = {}
exec(src10[: src10.index("# Classification table")], notes_ns)
clf_notes, reg_notes = notes_ns["clf_notes"], notes_ns["reg_notes"]

# ------------------------------------------------------------------ read the CSV (dataset facts)
raw = pd.read_csv(CSV)
RAW_ROWS, RAW_COLS = raw.shape
raw_nulls = int(raw.isnull().sum().sum())
tc_blank = int((raw["TotalCharges"].astype(str).str.strip() == "").sum())
tc_blank_tenure0 = int(((raw["TotalCharges"].astype(str).str.strip() == "") & (raw["tenure"] == 0)).sum())
raw_churn_yes = int((raw["Churn"] == "Yes").sum())
clean = raw.copy()
clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce").fillna(0)
clean = clean.drop(columns="customerID")
dups = int(clean.duplicated().sum())
clean = clean.drop_duplicates().reset_index(drop=True)
N = len(clean)
churn_yes = int((clean["Churn"] == "Yes").sum())
assert N == n_train + n_test, "CSV and notebook disagree about the number of clean rows"

# ------------------------------------------------------------------ derived confusion-matrix counts
N_POS = int(round(n_test * rate_test))      # churners in the test set
NEG = n_test - N_POS


def counts(recall, precision):
    tp = int(round(recall * N_POS))
    fn = N_POS - tp
    fp = int(round(tp / precision - tp)) if precision > 0 else 0
    return tp, fp, fn, NEG - fp


def row(df, model, stage=None):
    d = df[df["Model"] == model]
    if stage:
        d = d[d["Stage"] == stage]
    return d.iloc[0]


MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM (RBF)"]
for m in MODELS:
    r = row(tuned, m)
    tp, fp, fn, tn = counts(r["Recall"], r["Precision"])
    assert tn >= 0 and fp >= 0, "derived confusion counts are inconsistent"

f4 = lambda x: f"{x:.4f}"
f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{x * 100:.1f}%"

# ------------------------------------------------------------------ before/after chart (re-drawn from the notebook's numbers, legend outside the bars)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
x = np.arange(len(MODELS))
for ax, metric in zip(axes, ["F1", "Recall"]):
    b = [row(comp, m, "Baseline")[metric] for m in MODELS]
    t = [row(comp, m, "Tuned")[metric] for m in MODELS]
    r1 = ax.bar(x - 0.2, b, 0.4, label="Baseline", color="#1F77B4")
    r2 = ax.bar(x + 0.2, t, 0.4, label="Tuned", color="#E8751A")
    for rect in list(r1) + list(r2):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.012, f"{rect.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (RBF)", "").replace("Logistic Regression", "Logistic\nRegression").replace("Decision Tree", "Decision\nTree").replace("Random Forest", "Random\nForest") for m in MODELS], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{metric}: baseline vs tuned (test set)", fontsize=10)
    ax.set_ylabel(metric)
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
BA_PNG = TMP / "before_after_redrawn.png"
plt.savefig(BA_PNG, dpi=170)
plt.close()

FIG = {"eda": save_png(1, "eda_overview"), "base_cm": save_png(2, "baseline_cm"), "tuned_cm": save_png(8, "tuned_cm"),
       "imp": save_png(9, "importance"), "reg": save_png(7, "regression"), "ba": BA_PNG}

# ------------------------------------------------------------------ styles (same look as the Task 01 report)
basess = getSampleStyleSheet()
NAVY = colors.HexColor("#1F3A5F")
S = {
    "body": ParagraphStyle("body", parent=basess["BodyText"], fontName="Helvetica", fontSize=10, leading=14.5, spaceAfter=6),
    "h1": ParagraphStyle("h1", parent=basess["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=NAVY,
                         spaceBefore=14, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", parent=basess["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, textColor=NAVY,
                         spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "cap": ParagraphStyle("cap", parent=basess["BodyText"], fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
                          textColor=colors.HexColor("#444444"), alignment=TA_CENTER, spaceAfter=8),
    "cell": ParagraphStyle("cell", parent=basess["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11),
    "cellb": ParagraphStyle("cellb", parent=basess["BodyText"], fontName="Helvetica-Bold", fontSize=8.5, leading=11,
                            textColor=colors.white),
    "cell_s": ParagraphStyle("cell_s", parent=basess["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9.2),
    "cellb_s": ParagraphStyle("cellb_s", parent=basess["BodyText"], fontName="Helvetica-Bold", fontSize=7.4, leading=9.2,
                              textColor=colors.white),
    "bullet": ParagraphStyle("bullet", parent=basess["BodyText"], fontName="Helvetica", fontSize=10, leading=14,
                             leftIndent=14, bulletIndent=2, spaceAfter=3),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=6.2, leading=7.4, leftIndent=4),
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


def table(rows, widths, header=True, small=False):
    cs, hs = ("cell_s", "cellb_s") if small else ("cell", "cellb")
    data = [[Paragraph(md(str(c)), S[hs] if (header and i == 0) else S[cs]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F2F5F9")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return KeepTogether([t]) if len(rows) <= 12 else t     # short tables never split across pages


def picture(path, caption, max_w=16.5 * cm, max_h=10.5 * cm, lead=None):
    """Figure with caption; `lead` (e.g. a heading) is kept on the same page as the figure."""
    w, h = PILImage.open(path).size
    scale = min(max_w / w, max_h / h)
    return KeepTogether(list(lead or []) + [Image(str(path), width=w * scale, height=h * scale), Paragraph(md(caption), S["cap"])])


def code_listing(i, title, caption):
    src = "".join(CODE[i]["source"]).encode("cp1252", "replace").decode("cp1252").rstrip()
    return [KeepTogether([P(title, "h2"), Preformatted(src, S["code"]), Paragraph(md(caption), S["cap"])])]


W = 17 * cm
st = []

# ------------------------------------------------------------------ title page
st += [Spacer(1, 3.5 * cm), P("DEVIXO SOLUTIONS", "t_big"), Spacer(1, 0.3 * cm), P("AI/ML Internship Program", "t_mid"),
       Spacer(1, 1.2 * cm), P("**Task 02**", "t_mid"), P("Build and Compare Machine Learning Models", "t_mid"), Spacer(1, 2.2 * cm),
       P(f"**Student Name:** {STUDENT_NAME}", "t_small"), P(f"**Internship Domain:** {INTERNSHIP_DOMAIN}", "t_small"),
       P(f"**Submission Date:** {SUBMISSION_DATE}", "t_small"), Spacer(1, 2.5 * cm),
       P("Project: Customer Churn Classification and Monthly Charges Regression on the Telco Customer Churn Dataset", "t_small"),
       PageBreak()]

# ------------------------------------------------------------------ 1-2
st += [P("1. Introduction", "h1"),
       P("This report documents Task 02 of the Devixo Solutions AI/ML Internship Program. It follows the full machine learning workflow: "
         "loading a real dataset, preprocessing it, training several models, evaluating them with suitable metrics, comparing them, "
         "tuning them with cross-validation (bonus) and recommending the best model with evidence."),
       P("The dataset describes the customers of a telecom company. Two problems are solved on it: **Task A (main)** is a classification "
         "problem, predicting whether a customer leaves (`Churn`); **Task B** is a regression problem, predicting the `MonthlyCharges` of a customer."),
       P("2. Objective", "h1"),
       P("The objective is to learn why preprocessing, model comparison and model selection matter: to train at least three different "
         "algorithms, evaluate each with appropriate metrics, compare them on training time and performance, tune them, and choose a "
         "model on evidence instead of preference.")]

# ------------------------------------------------------------------ 3-6
st += [P("3. Dataset Name", "h1"), P("**Telco Customer Churn** (file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`)."),
       P("4. Dataset Source", "h1"),
       P(f"Kaggle: {KAGGLE_URL}. It is an IBM sample dataset about a **fictional** telecom company that sells phone and internet services. "
         "Because the company is fictional, the findings describe this dataset and may not transfer to the customers of a real company. "
         "This is stated again in the limitations."),
       P("5. Dataset Description", "h1"),
       P("Each row is one customer. The columns describe the customer (gender, senior citizen, partner, dependents), the account "
         "(tenure in months, contract, billing and payment method), the services the customer has (phone, internet, security, backup, "
         "device protection, tech support, streaming), the charges, and whether the customer left (`Churn`)."),
       P("6. Dataset Size", "h1")]
st += [table([["", "Raw dataset", "After preprocessing"],
              ["Rows (customers)", f"{RAW_ROWS:,}", f"{N:,}"],
              ["Columns", str(RAW_COLS), f"{RAW_COLS - 1} (`customerID` removed)"],
              ["Blank `TotalCharges` values (hidden missing)", str(tc_blank), "0 (filled with 0)"],
              ["Duplicate rows (ignoring `customerID`)", str(dups), "0 (removed)"],
              ["Customers who churned", f"{raw_churn_yes:,} ({raw_churn_yes / RAW_ROWS * 100:.1f}%)", f"{churn_yes:,} ({churn_yes / N * 100:.1f}%)"],
              ["File size", f"about {CSV.stat().st_size / 1e6:.2f} MB", "-"]], [7 * cm, 5 * cm, 5 * cm]),
       Spacer(1, 4)]

# ------------------------------------------------------------------ 7
st += [P("7. Features/Columns Description", "h1"),
       table([["Group", "Columns", "Meaning"],
              ["Identifier", "customerID", "Unique ID of the customer; dropped because it has no predictive meaning"],
              ["Demographics", "gender, SeniorCitizen, Partner, Dependents", "Customer profile; `SeniorCitizen` is already 0/1"],
              ["Account", "tenure, Contract, PaperlessBilling, PaymentMethod",
               "Months with the company; contract type (Month-to-month, One year, Two year); billing and payment method"],
              ["Services", "PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies",
               "Which services the customer has; `InternetService` is DSL, Fiber optic or No"],
              ["Charges", "MonthlyCharges, TotalCharges", "Current monthly bill and total billed so far (`TotalCharges` is stored as text in the raw file)"],
              ["Targets", "Churn (Task A), MonthlyCharges (Task B)", "`Churn` = Yes/No (encoded 1/0); `MonthlyCharges` is the regression target"]],
             [2.6 * cm, 6.4 * cm, 8 * cm])]

# ------------------------------------------------------------------ 8
st += [P("8. Methodology", "h1"),
       P("The work was done in Python 3 with Pandas, NumPy, Scikit-learn, Matplotlib and Seaborn in a Jupyter notebook (Google Colab). "
         "All scikit-learn models run on the CPU, so the Colab GPU was not needed. The random seed is 42 everywhere."),
       *bullets(["**Part 1 - Preprocessing:** load, fix hidden missing values, remove duplicates, encode, scale and split (split first, then fit the preprocessing on the training data only).",
                 "**Part 2 - Model development:** five algorithms for classification and five for regression, each with a Dummy model as a reference floor.",
                 "**Part 3 - Evaluation:** Accuracy, Precision, Recall, F1 and confusion matrices (classification); MAE, MSE, RMSE and R2 (regression).",
                 "**Bonus - Tuning:** `GridSearchCV` with 5-fold stratified cross-validation on the training set only, scored by F1.",
                 "**Part 4 - Comparison:** tables with training time, metrics, advantages and limitations.",
                 "**Part 5 - Conclusion:** a recommended model for each task and the reasons."])]

# ------------------------------------------------------------------ 9 preprocessing
st += [P("9. Part 1 - Data Preprocessing", "h1"),
       P("Every action is recorded below with its reason. All preprocessing was done on a copy of the data; the raw file was never changed."),
       table([["Step", "Action", "Affected", "Reason"],
              ["1", "Loaded the CSV", f"{RAW_ROWS:,} rows, {RAW_COLS} columns", "The task requires at least 1,000 records."],
              ["2", "Converted `TotalCharges` from text to numbers and filled blanks with 0", f"{tc_blank} values",
               f"`isnull()` found {raw_nulls} missing values because the blanks are text. All {tc_blank_tenure0} of them have `tenure` = 0, so these customers have not been billed yet and 0 is the logical value (a median would invent a bill)."],
              ["3", "Removed duplicate rows (ignoring `customerID`)", f"{dups} rows",
               "Identical in every feature and in the label. Removing them also stops the same row from appearing in both the training and the test set. Assumption: they are redundant records."],
              ["4", "Dropped `customerID`", "1 column", "A unique ID has no predictive meaning."],
              ["5", "Encoded the target: No = 0, Yes = 1", "`Churn`", "Churn is the positive class, so Precision and Recall describe churners."],
              ["6", "Split 80% / 20%, stratified, `random_state=42`", f"{n_train:,} train / {n_test:,} test",
               f"Stratifying keeps the churn rate the same in both sets ({rate_train:.4f} train, {rate_test:.4f} test)."],
              ["7", "Scaled `tenure`, `MonthlyCharges`, `TotalCharges` with `StandardScaler`", "3 columns",
               "Needed by Logistic Regression, KNN and SVM. Fitted on the training data only (inside a Pipeline) to avoid data leakage."],
              ["8", "One-hot encoded the other 16 categorical columns", "16 columns",
               "Models need numbers. `handle_unknown=\"ignore\"` keeps the pipeline safe for unseen categories."]],
             [1.0 * cm, 5.3 * cm, 3.2 * cm, 7.5 * cm]),
       Spacer(1, 4),
       P(f"Result: {N:,} clean rows (the requirement of at least 1,000 records is met), {RAW_COLS - 1} columns before encoding, and no missing values or duplicates.")]

# ------------------------------------------------------------------ 10 EDA
num_stats = clean[["tenure", "MonthlyCharges", "TotalCharges"]].agg(["mean", "median", "std", "min", "max"]).T.round(2)
rows = [["Variable", "Mean", "Median", "Std. dev.", "Min", "Max"]] + [[k] + [f"{v:,.2f}" for v in r.values] for k, r in num_stats.iterrows()]
contract_counts = clean["Contract"].value_counts()
internet_counts = clean["InternetService"].value_counts()
st += [P("10. Exploratory Data Analysis", "h1"),
       P("Statistics of the numeric columns (cleaned data, all rows):"), table(rows, [4.6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.4 * cm, 2.5 * cm]),
       Spacer(1, 6),
       P(f"Categories: contracts are Month-to-month ({contract_counts['Month-to-month']:,}), Two year ({contract_counts['Two year']:,}) and One year "
         f"({contract_counts['One year']:,}); internet service is Fiber optic ({internet_counts['Fiber optic']:,}), DSL ({internet_counts['DSL']:,}) and "
         f"No internet ({internet_counts['No']:,})."),
       P("The next figure uses the **training data only**, so no idea is formed by looking at the test set.")]
st += [picture(FIG["eda"], "Figure 10.1: churn distribution, churn rate by contract type and tenure by churn (training set)"),
       P(f"The churn rate is **{contract_rate['Month-to-month']:.1f}%** for month-to-month contracts, **{contract_rate['One year']:.1f}%** for one-year "
         f"and **{contract_rate['Two year']:.1f}%** for two-year contracts. The median tenure is **{tenure_median[0]:.0f} months** for customers who stayed and "
         f"**{tenure_median[1]:.0f} months** for customers who left. Churn is the minority class (about {churn_yes / N * 100:.0f}% of customers), so a "
         "stratified split is used and accuracy cannot be the only metric. These are associations, not causes, and contract type and tenure "
         "are probably related to each other.")]

# ------------------------------------------------------------------ 11 models
st += [P("11. Part 2 - Model Development", "h1"),
       P("Every model is a Scikit-learn `Pipeline` made of the shared preprocessing step and the model, trained on the same training set. "
         "Baseline models use default settings. A Dummy model that always predicts the majority class (classification) or the training mean "
         "(regression) is added as a reference floor."),
       table([["Model", "Task A (churn)", "Task B (price)", "Needs scaling?", "Baseline settings"],
              ["Logistic Regression", "Yes", "- (Linear Regression is used)", "Yes", "`max_iter=1000`"],
              ["Linear Regression", "-", "Yes", "Yes (helps)", "defaults"],
              ["Decision Tree", "Yes", "Yes", "No", "defaults (no depth limit)"],
              ["Random Forest", "Yes", "Yes", "No", "defaults (100 trees)"],
              ["KNN", "Yes", "Yes", "Yes", "defaults (5 neighbours)"],
              ["SVM / SVR (RBF kernel)", "Yes", "Yes", "Yes", "defaults"],
              ["Dummy", "Majority class", "Training mean", "-", "reference only"]],
             [4.4 * cm, 2.6 * cm, 3.6 * cm, 2.6 * cm, 3.8 * cm]),
       Spacer(1, 4),
       P("XGBoost (optional in the task) was not used, to keep every model explainable.")]

# ------------------------------------------------------------------ 12 evaluation
rows = [["Model", "Train time (s)", "Accuracy", "Precision", "Recall", "F1"]]
for _, r in base.iterrows():
    rows.append([r["Model"], f3(r["Train time (s)"]), f4(r["Accuracy"]), f4(r["Precision"]), f4(r["Recall"]), f4(r["F1"])])
dummy = row(base, "Dummy (majority class)")
b_rows = {m: row(base, m) for m in MODELS}
fn_base = {m: counts(b_rows[m]["Recall"], b_rows[m]["Precision"])[2] for m in MODELS}
rec_lo, rec_hi = min(b_rows[m]["Recall"] for m in MODELS), max(b_rows[m]["Recall"] for m in MODELS)
st += [P("12. Part 3 - Model Evaluation (Task A, baseline models)", "h1"),
       P("Metrics are calculated on the same test set for every model. The positive class is churn. Precision is the share of flagged customers who really churn; "
         "Recall is the share of real churners that are found; F1 balances the two. The training times in this first table were measured in single runs "
         "(a fairer timing is in section 13.3)."),
       table(rows, [5 * cm, 2.6 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.5 * cm]), Spacer(1, 6),
       picture(FIG["base_cm"], "Figure 12.1: confusion matrices of the baseline models on the test set", max_h=7.6 * cm),
       *bullets([f"The Dummy model has accuracy **{f4(dummy['Accuracy'])}**, which is the share of non-churners in the test set. All five real models beat it on F1.",
                 f"Logistic Regression has the best baseline accuracy (**{f4(b_rows['Logistic Regression']['Accuracy'])}**) and F1 (**{f4(b_rows['Logistic Regression']['F1'])}**).",
                 f"The unpruned Decision Tree has accuracy **{f4(b_rows['Decision Tree']['Accuracy'])}**, below the Dummy, but its recall ({f4(b_rows['Decision Tree']['Recall'])}) is higher "
                 "than KNN, SVM and Random Forest. Accuracy and recall tell different stories, so accuracy alone is misleading here.",
                 f"Every baseline model has recall between **{f4(rec_lo)} and {f4(rec_hi)}**: each finds only about half of the {N_POS} churners in the test set. "
                 f"Churners missed: Random Forest {fn_base['Random Forest']}, SVM {fn_base['SVM (RBF)']}, KNN {fn_base['KNN']}, "
                 f"Decision Tree {fn_base['Decision Tree']}, Logistic Regression {fn_base['Logistic Regression']}."])]

# ------------------------------------------------------------------ guards: warn if a re-run contradicts the written text
WARN = []


def check(cond, msg):
    if not cond:
        WARN.append(msg)


tb = {m: row(tuned, m) for m in MODELS}
cb = {m: row(comp, m, "Baseline") for m in MODELS}
ct = {m: row(comp, m, "Tuned") for m in MODELS}
tuned_cm = {m: counts(tb[m]["Recall"], tb[m]["Precision"]) for m in MODELS}
base_cm = {m: counts(b_rows[m]["Recall"], b_rows[m]["Precision"]) for m in MODELS}

# ------------------------------------------------------------------ 13 tuning
check(sum("'class_weight': 'balanced'" in best_params[m] for m in MODELS) == 4, "class_weight=balanced is no longer chosen by all four models")
check("'C': 10" in best_params["Logistic Regression"] and "'n_neighbors': 31" in best_params["KNN"]
      and "'min_samples_leaf': 30" in best_params["Decision Tree"] and "'n_estimators': 300" in best_params["Random Forest"]
      and "'min_samples_leaf': 10" in best_params["Random Forest"], "best parameters changed: the 'edge of the grid' sentence may be wrong")
check(all(ct[m]["Recall"] > cb[m]["Recall"] and ct[m]["F1"] > cb[m]["F1"] for m in MODELS), "recall/F1 no longer rise for every model after tuning")

rows = [["Model", "Best parameters", "CV F1 (train)", "Accuracy", "Precision", "Recall", "F1", "Search (s)"]]
for m in MODELS:
    r = tb[m]
    rows.append([m, best_params[m], f4(r["CV F1 (train)"]), f4(r["Accuracy"]), f4(r["Precision"]), f4(r["Recall"]), f4(r["F1"]), f"{r['Search time (s)']:.1f}"])
gaps = {m: tb[m]["CV F1 (train)"] - tb[m]["F1"] for m in MODELS}
edge = "Logistic Regression `C=10`, Decision Tree `min_samples_leaf=30`, Random Forest `n_estimators=300` and `min_samples_leaf=10`, KNN `n_neighbors=31`"
st += [P("13. Bonus - Hyperparameter Tuning (GridSearchCV)", "h1"),
       P("**Design.** `GridSearchCV` with small grids and **5-fold stratified cross-validation on the training set only**, scored by **F1** of the churn class "
         "(accuracy rewards predicting 'No' on imbalanced data). `class_weight` is part of the grid for the models that support it, because it makes a model pay "
         "more attention to the rare churn class. After the search, each model is refitted with the best settings and evaluated **once** on the test set, so the "
         "test set never influences the choice of settings."),
       P("13.1 Grids searched", "h2"),
       table([["Model", "Parameters searched"],
              ["Logistic Regression", "`C` in {0.01, 0.1, 1, 10}; `class_weight` in {None, balanced}"],
              ["Decision Tree", "`max_depth` in {3, 5, 7, 10}; `min_samples_leaf` in {1, 10, 30}; `class_weight` in {None, balanced}"],
              ["Random Forest", "`n_estimators` in {100, 300}; `max_depth` in {5, 10, None}; `min_samples_leaf` in {1, 5, 10}; `class_weight` in {None, balanced}"],
              ["KNN", "`n_neighbors` in {5, 11, 21, 31}; `weights` in {uniform, distance}"],
              ["SVM (RBF)", "`C` in {0.1, 1, 10}; `gamma` in {scale, 0.01}; `class_weight` in {None, balanced}"]], [4 * cm, 13 * cm]),
       P("13.2 Results and before/after comparison", "h2"),
       table(rows, [2.5 * cm, 4.9 * cm, 1.5 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm], small=True), Spacer(1, 6)]
rows = [["Model", "Stage", "Accuracy", "Precision", "Recall", "F1"]]
for m in MODELS:
    for stage, d in (("Baseline", cb[m]), ("Tuned", ct[m])):
        rows.append([m if stage == "Baseline" else "", stage, f4(d["Accuracy"]), f4(d["Precision"]), f4(d["Recall"]), f4(d["F1"])])
dl = lambda metric: ", ".join(f"{m} {ct[m][metric] - cb[m][metric]:+.4f}" for m in MODELS)
acc_down = [m for m in MODELS if ct[m]["Accuracy"] < cb[m]["Accuracy"]]
prec_down = [m for m in MODELS if ct[m]["Precision"] < cb[m]["Precision"]]
st += [table(rows, [4.4 * cm, 2.6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]), Spacer(1, 6),
       picture(FIG["ba"], "Figure 13.1: F1 and Recall before and after tuning (test set; re-drawn from the notebook's numbers)"),
       *bullets([f"**Recall rose for every model:** {dl('Recall')}.",
                 f"**F1 rose for every model:** {dl('F1')}.",
                 f"**Precision fell** for {', '.join(prec_down)} and rose for the others. **Accuracy fell** for {', '.join(acc_down)}.",
                 f"`class_weight=\"balanced\"` was chosen by all four models that support it. The Random Forest search took the longest ({tb['Random Forest']['Search time (s)']:.1f} s), then SVM ({tb['SVM (RBF)']['Search time (s)']:.1f} s).",
                 f"Cross-validated F1 on the training data is close to the test F1 (largest gap {max(gaps.values()):.4f}, for {max(gaps, key=gaps.get)}), so there is no sign of serious overfitting.",
                 f"Tuned Logistic Regression has accuracy {f4(ct['Logistic Regression']['Accuracy'])}, barely above the Dummy ({f4(dummy['Accuracy'])}), yet it finds {pct(ct['Logistic Regression']['Recall'])} of the churners where the Dummy finds none."]),
       P("**Interpretation.** The models trade precision for recall. With balanced class weights they flag more customers as churners, so they catch more real churners "
         "but also raise more false alarms. That is a business trade-off (a missed churner against a wasted retention offer), not a defect. KNN has no `class_weight` option, "
         f"which is why it changed the least. Several winning values sit at the edge of their grid ({edge}), so a wider grid might do slightly better.")]

# 13.3 fair timing
fit_all = {(m, s_): d["Fit time (s)"] for m in MODELS for s_, d in (("Baseline", cb[m]), ("Tuned", ct[m]))}
check(min(fit_all, key=fit_all.get)[0] == "KNN", "KNN is no longer the fastest model to fit")
rows = [["Model", "Baseline fit (s)", "Tuned fit (s)", "Search time (s)"]] + [[m, f3(cb[m]["Fit time (s)"]), f3(ct[m]["Fit time (s)"]), f"{tb[m]['Search time (s)']:.1f}"] for m in MODELS]
rf_ratio = ct["Random Forest"]["Fit time (s)"] / cb["Random Forest"]["Fit time (s)"]
st += [P("13.3 Fair training-time comparison", "h2"),
       P("The first timing tables are not fair: the baseline Random Forest used one CPU core and the tuned one used all cores, with 300 trees instead of 100. "
         "The times below refit every model with the **same CPU usage (one core)** and report the **median of 5 fits** (Colab CPU). The search time is the cost of the "
         "grid search itself (all cores)."),
       table(rows, [5 * cm, 4 * cm, 4 * cm, 4 * cm]), Spacer(1, 4),
       P(f"The tuned Random Forest is about **{rf_ratio:.1f} times slower** to fit than the baseline, which matches 300 trees instead of 100. KNN is the fastest model to fit in both stages; "
         "SVM and Random Forest are the slowest. Differences of about 0.05 s are within the noise of a shared Colab machine and should not be interpreted.")]

# 13.4 confusion matrices after tuning
rows = [["Model", "Churners found (TP)", "Churners missed (FN)", "False alarms (FP)"]]
for m in MODELS:
    rows.append([m, f"{base_cm[m][0]} to {tuned_cm[m][0]}", f"{base_cm[m][2]} to {tuned_cm[m][2]}", f"{base_cm[m][1]} to {tuned_cm[m][1]}"])
st += [picture(FIG["tuned_cm"], "Figure 13.2: confusion matrices of the tuned models on the test set", max_h=9 * cm,
               lead=[P("13.4 Confusion matrices after tuning", "h2")]),
       table(rows, [4.6 * cm, 4.1 * cm, 4.2 * cm, 4.1 * cm]),
       P(f"Counts are baseline to tuned. The test set has {N_POS} churners and {NEG} non-churners. After tuning, the models miss far fewer churners, "
         "at the price of more false alarms. The counts are derived from Precision and Recall and match the confusion-matrix charts.", "cap")]

# ------------------------------------------------------------------ 14 feature importance
check(imp[0][0].startswith("Contract_") and "tenure" in [k for k, _ in imp[:3]], "top features changed: re-read the feature-importance paragraph")
st += [picture(FIG["imp"], "Figure 14.1: ten most important features (impurity-based importance; one-hot columns appear as Feature_Category)", max_w=12.5 * cm, max_h=8 * cm,
               lead=[P("14. Feature Importance (tuned Random Forest)", "h1")]),
       table([["Feature", "Importance"]] + [[k, f"{v:.4f}"] for k, v in imp], [9 * cm, 3 * cm]), Spacer(1, 6),
       P(f"The most important features are `{imp[0][0]}` ({imp[0][1]:.4f}), `{imp[1][0]}` ({imp[1][1]:.4f}) and `{imp[2][0]}` ({imp[2][1]:.4f}). "
         f"{sum(k.startswith('Contract_') for k, _ in imp)} of the top ten columns come from `Contract`. This agrees with the exploratory analysis. Two cautions: the importance shows what the "
         "model **used**, not what **causes** churn; and impurity-based importance tends to favour numeric features with many distinct values, while a category such as Contract is split over its dummy columns.")]

# ------------------------------------------------------------------ 15 regression
real_reg = [m for m in reg["Model"] if not m.startswith("Dummy")]
best_by = {c: reg[reg["Model"].isin(real_reg)].sort_values(c, ascending=(c != "R2")).iloc[0]["Model"] for c in ["MAE", "MSE", "RMSE", "R2"]}
check(set(best_by.values()) == {"Linear Regression"}, "Linear Regression is no longer best on every regression metric")
dr = row(reg, "Dummy (mean)")
check(abs(dr["R2"]) < 0.01, "Dummy regressor R2 is not about 0")
rrow = {m: row(reg, m) for m in real_reg}
rows = [["Model", "Train time (s)", "MAE", "MSE", "RMSE", "R2"]]
for _, r in reg.iterrows():
    rows.append([r["Model"], f3(r["Train time (s)"]), f4(r["MAE"]), f4(r["MSE"]), f4(r["RMSE"]), f4(r["R2"])])
vrows = [["Internet service", "Customers", "Mean price", "Std. dev.", "Min", "Max"]] + [
    [k, f"{v['count']:,}", f"{v['mean']:.2f}", f"{v['std']:.2f}", f"{v['min']:.2f}", f"{v['max']:.2f}"] for k, v in
    sorted(price_by_internet.items(), key=lambda kv: kv[1]["mean"])]
worst = reg[reg["Model"].isin(real_reg)].sort_values("R2").iloc[0]
st += [P("15. Task B - Regression: Predicting MonthlyCharges", "h1"),
       P("The same dataset also supports an honest regression question: the monthly price should depend on which services a customer has. "
         "**Target:** `MonthlyCharges`. **Excluded features:** `TotalCharges` (roughly `tenure` times `MonthlyCharges`, so it would leak the answer) and `Churn` (an outcome, not a pricing input). "
         f"**Features kept:** `tenure` (scaled) and the 16 categorical columns (one-hot encoded). The split is 80/20 and not stratified ({n_train:,} training rows, {n_test:,} test rows, seed 42)."),
       P("**Metrics:** MAE is the average size of the error; MSE averages the squared errors; RMSE is the square root of MSE (in the units of the price, punishing large errors more); "
         "R2 is the share of the variation in price that the model explains (1 is perfect, 0 is no better than the mean)."),
       P(f"**Validation check.** Training prices range from {price_stats['min']:.2f} to {price_stats['max']:.2f} (mean {price_stats['mean']:.2f}). The average price depends strongly on the internet service, "
         "but a large spread remains inside each type, so other services also change the price:"),
       table(vrows, [4 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.3 * cm, 2.3 * cm]), Spacer(1, 8),
       P("**Regression results on the test set (default settings):**"),
       table(rows, [5 * cm, 2.6 * cm, 2.3 * cm, 2.4 * cm, 2.3 * cm, 2.4 * cm]), Spacer(1, 6),
       picture(FIG["reg"], "Figure 15.1: actual vs predicted MonthlyCharges on the test set (points near the red line are good predictions)", max_h=9.5 * cm),
       *bullets([f"The Dummy model has R2 = {f4(dr['R2'])} and MAE = {f4(dr['MAE'])}, so the metric calculation is wired correctly.",
                 f"**Linear Regression is best on all four metrics:** MAE {f4(rrow['Linear Regression']['MAE'])}, MSE {f4(rrow['Linear Regression']['MSE'])}, RMSE {f4(rrow['Linear Regression']['RMSE'])}, R2 {f4(rrow['Linear Regression']['R2'])}.",
                 (f"Random Forest and SVR both have R2 {f4(rrow['Random Forest']['R2'])}" if rrow['Random Forest']['R2'] == rrow['SVR (RBF)']['R2'] else f"Random Forest and SVR have R2 {f4(rrow['Random Forest']['R2'])} and {f4(rrow['SVR (RBF)']['R2'])}") + f", the Decision Tree {f4(rrow['Decision Tree']['R2'])}, and {worst['Model']} is the weakest real model (R2 {f4(worst['R2'])}, MAE {f4(worst['MAE'])})."]),
       P("**Interpretation (a hypothesis, not tested directly).** The price looks close to a sum of fixed charges for each service. If so, a linear model fits it almost perfectly and the flexible "
         "models have nothing extra to find. The table above and the fact that Linear Regression beats the flexible models agree with this idea, but they do not prove it. KNN probably suffers because "
         "distances are noisy across many one-hot columns. Every real model has R2 above 0.97, so this task is **easy** and the comparison is less informative than for churn. "
         "No tuning was done for regression; the bonus requirement is covered by Task A.")]

# ------------------------------------------------------------------ 16 Part 4 tables
rows = [["Model", "Fit time (s)", "Acc.", "Prec.", "Recall", "F1", "Advantages", "Limitations"]]
for m in MODELS:
    d = ct[m]
    rows.append([m, f"{d['Fit time (s)']:.3f}<br/>(search {tb[m]['Search time (s)']:.1f})".replace("<br/>", " "), f"{d['Accuracy']:.4f}", f"{d['Precision']:.4f}", f"{d['Recall']:.4f}", f"{d['F1']:.4f}",
                 clf_notes[m][0], clf_notes[m][1]])
rows2 = [["Model", "Train time (s)", "MAE", "RMSE", "R2", "Advantages", "Limitations"]]
for _, r in reg.iterrows():
    rows2.append([r["Model"], f3(r["Train time (s)"]), f4(r["MAE"]), f4(r["RMSE"]), f4(r["R2"]), reg_notes[r["Model"]][0], reg_notes[r["Model"]][1]])
st += [P("16. Part 4 - Model Comparison", "h1"),
       P("**Classification (Task A).** The table shows the final, tuned version of each model on the test set. Fit time is the median of 5 single-core fits; the search time is the cost of the grid "
         "search. The baseline results are in section 12 and the before/after comparison is in section 13.2."),
       table(rows, [2.2 * cm, 1.8 * cm, 1.15 * cm, 1.15 * cm, 1.2 * cm, 1.15 * cm, 4.1 * cm, 4.25 * cm], small=True), Spacer(1, 8),
       P("**Regression (Task B).** Results on the test set with default settings (no tuning); the Dummy row is the reference floor."),
       table(rows2, [2.4 * cm, 1.6 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 4.5 * cm, 4.3 * cm], small=True)]

# ------------------------------------------------------------------ 17 conclusion
rf_t, svm_t, lr_t = tb["Random Forest"], tb["SVM (RBF)"], tb["Logistic Regression"]
check(abs(rf_t["F1"] - max(tb[m]["F1"] for m in MODELS)) < 0.002, "tuned Random Forest is no longer (nearly) the best on F1")
check(rf_t["CV F1 (train)"] == max(tb[m]["CV F1 (train)"] for m in MODELS), "tuned Random Forest no longer has the best cross-validated F1")
high_rec = [m for m in MODELS if tb[m]["Recall"] >= 0.75]
check(max(high_rec, key=lambda m: tb[m]["Precision"]) == "Random Forest", "Random Forest no longer has the best precision among models with recall >= 0.75")
check(lr_t["Recall"] == max(tb[m]["Recall"] for m in MODELS), "Logistic Regression no longer has the highest recall")
speedup = ct["Random Forest"]["Fit time (s)"] / ct["Logistic Regression"]["Fit time (s)"]
check(speedup > 10, "Logistic Regression is no longer more than 10 times faster to fit than the tuned forest")
lrr = rrow["Linear Regression"]
st += [P("17. Part 5 - Conclusion and Recommendation", "h1"),
       P(f"I trained and compared five classification models and five regression models on the Telco Customer Churn dataset ({N:,} customers after cleaning). I fixed hidden missing values, removed "
         f"{dups} duplicate rows, split the data before fitting any scaler or encoder, and compared every model on the same test set. Because only about {churn_yes / N * 100:.0f}% of customers churn, "
         "I judged the churn models by Precision, Recall and F1 and not by accuracy alone."),
       P(f"**Classification: I recommend the tuned Random Forest** (`n_estimators=300`, `max_depth=10`, `min_samples_leaf=10`, `class_weight=balanced`). Its test F1 ({f4(rf_t['F1'])}) is tied for the best with SVM ({f4(svm_t['F1'])}); "
         f"the difference of {abs(rf_t['F1'] - svm_t['F1']):.4f} is far smaller than the noise to expect from {N_POS} test churners, so the two cannot be separated on F1. The tie is broken by three points: the Random Forest has the "
         f"highest cross-validated F1 on the training data ({f4(rf_t['CV F1 (train)'])}), which is a steadier estimate than a single test split; among the models with a recall of about 0.75 or more it has the best precision "
         f"({f4(rf_t['Precision'])}); and it provides feature importance. It finds {tuned_cm['Random Forest'][0]} of the {N_POS} churners, against {base_cm['Random Forest'][0]} for the baseline forest. "
         "A likely reason it did better is that a forest can combine several features at once (for example contract type, tenure and services) while a linear model adds up separate effects; the small margin over "
         "Logistic Regression suggests that most of the pattern is simple. I did not test this explanation directly."),
       P(f"An honest alternative is **Logistic Regression**: its F1 is {rf_t['F1'] - lr_t['F1']:.4f} lower ({f4(lr_t['F1'])}), but it has the highest recall ({f4(lr_t['Recall'])}), is easy to explain and fits more than ten times faster "
         f"({ct['Logistic Regression']['Fit time (s)']:.3f} s against {ct['Random Forest']['Fit time (s)']:.3f} s). If the business prefers a simple, explainable model, it is a defensible choice. The right choice also depends on the "
         "cost of a missed churner against a false alarm, which this project did not model."),
       P(f"**Regression: I recommend Linear Regression.** It has the lowest MAE ({f4(lrr['MAE'])}), MSE ({f4(lrr['MSE'])}) and RMSE ({f4(lrr['RMSE'])}) and the highest R2 ({f4(lrr['R2'])}), and it is simple and fast. "
         "The likely reason is that the monthly price is close to an additive sum of service charges (a hypothesis supported by the results, not proven)."),
       P("From this project I learned that `isnull()` can hide missing data, that accuracy is a poor metric on imbalanced data, that scaling and encoding must be fitted on training data only, that tuning can raise recall "
         "mainly by giving up precision, and that training times must be measured under equal conditions before they are compared.")]

# ------------------------------------------------------------------ 18 limitations + 19 findings
st += [P("18. Limitations", "h1"),
       *bullets(["**Fictional data:** the dataset describes a fictional telecom company, so the findings are for this dataset and not for real customers.",
                 f"**One train/test split** with only {N_POS} churners in the test set: differences of about 0.01 to 0.02 in F1 are within noise. Cross-validated scores were used to support the choice.",
                 "**Fixed 0.5 decision threshold and no cost model:** the business cost of a missed churner against a false alarm was not used.",
                 f"**Class weights trade precision for recall:** accuracy fell for {len(acc_down)} of the five tuned models, and tuned Logistic Regression's accuracy ({f4(ct['Logistic Regression']['Accuracy'])}) is barely above 'always No' ({f4(dummy['Accuracy'])}).",
                 "**Small grids with best values at the grid edge:** a wider search might improve the scores slightly.",
                 "**Timing on Colab is noisy:** only large differences are meaningful. Search times use all cores; fit times use one core.",
                 "**Assumptions in preprocessing:** identical rows are redundant, and customers who have not been billed yet owe nothing (`TotalCharges` = 0).",
                 "**Task B is easy:** all real models have R2 above 0.97, and no regression tuning was done.",
                 "**Feature importance** is impurity-based and not causal. XGBoost (optional in the task) was not included."]),
       P("19. Major Findings", "h1"),
       *bullets([f"After cleaning, the dataset has {N:,} customers; {churn_yes / N * 100:.1f}% churned.",
                 f"Churn is much higher on month-to-month contracts ({contract_rate['Month-to-month']:.1f}%) than on two-year contracts ({contract_rate['Two year']:.1f}%), and churners have a much shorter median tenure ({tenure_median[1]:.0f} against {tenure_median[0]:.0f} months). These are associations, not causes.",
                 f"Default models find only {pct(rec_lo)} to {pct(rec_hi)} of the churners; accuracy hides this because the Dummy model already scores {f4(dummy['Accuracy'])}.",
                 f"Tuning with balanced class weights raised recall for every model (the best reaches {f4(lr_t['Recall'])}) and F1 for every model, but lowered precision for {len(prec_down)} of five.",
                 f"The tuned Random Forest and SVM are tied on F1 ({f4(rf_t['F1'])} and {f4(svm_t['F1'])}); the forest is recommended for its cross-validated F1, precision and feature importance.",
                 f"`MonthlyCharges` can be predicted almost perfectly from the services (Linear Regression R2 {f4(lrr['R2'])}), which suggests an additive price structure."])]

# ------------------------------------------------------------------ 20 appendix
st += [P("20. Appendix - Key Code from the Notebook", "h1"),
       P("The code below is copied from the notebook cells (text, not screenshots). The notebook in the repository contains all cells with their saved outputs."),
       *code_listing(0, "A.1 Data loading, cleaning and split (Part 1)", "Listing A.1: preprocessing cell"),
       *code_listing(2, "A.2 Baseline models and evaluation (Parts 2 and 3)", "Listing A.2: baseline models"),
       *code_listing(3, "A.3 Hyperparameter tuning (Bonus)", "Listing A.3: GridSearchCV for all five models"),
       *code_listing(5, "A.4 Fair training-time measurement", "Listing A.4: median of 5 single-core fits")]

# ------------------------------------------------------------------ 21
st += [P("21. GitHub Repository Link", "h1"), P(f"{GITHUB_URL} (folder: `{GITHUB_FOLDER}`)"),
       P("The task folder contains the notebook with saved outputs, the dataset folder (or download link), the figures in `images/`, the result tables in `results/`, "
         "this report, the report builder script, `requirements.txt` and the README.")]


# ------------------------------------------------------------------ build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "Devixo Solutions - AI/ML Internship Program - Task 02")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                        title="Task 02 - Build and Compare Machine Learning Models", author=STUDENT_NAME)
doc.build(st, onFirstPage=footer, onLaterPages=footer)

print(f"Report written: {OUT}")
if WARN:
    print(f"\nWARNING: {len(WARN)} statement(s) in the written text may no longer match the notebook results. Re-read these parts:")
    for w in WARN:
        print("  -", w)
else:
    print("All consistency checks between the written text and the notebook results passed.")
