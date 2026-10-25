"""
Builds the Task 03 PDF report (same look and structure as the Task 02 report).

How to use
1. Run the three notebooks in order (notebooks/01_..., 02_..., 03_...) so they contain their saved outputs.
2. Edit the STUDENT DETAILS block below if needed (name, GitHub link).
3. Run from the project folder:   python report/build_report_task3.py
   The report is saved as report/Task_03_Report.pdf

Where the numbers come from (nothing is typed in by hand)
- Model results, best parameters, feature-selection experiments, error analysis and feature importance are read from the
  SAVED OUTPUTS of the notebooks. If you re-run the notebooks and rebuild, the report updates itself.
- Dataset facts (rows, duplicates, missing values, statistics) are calculated from the original CSV.
- File sizes and the model metadata are read from the models/ folder.
The written explanations (interpretation, conclusion, limitations) are the author's reasoning. The script prints a WARNING
when a re-run makes one of those statements untrue, so re-read those paragraphs in that case.
"""
import re
import json
import subprocess
import sys
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
GITHUB_FOLDER = "Task_03_ML"          # folder of this task inside the repository
KAGGLE_URL = "https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho"
# ============================================================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
NB1 = ROOT / "notebooks" / "01_eda_and_data_audit.ipynb"
NB2 = ROOT / "notebooks" / "02_feature_engineering_and_selection.ipynb"
NB3 = ROOT / "notebooks" / "03_modeling_tuning_and_evaluation.ipynb"
CSV = ROOT / "data" / "raw" / "car_details_v3.csv"
FIGDIR = ROOT / "reports" / "figures"
META = ROOT / "models" / "model_metadata.json"
OUT = ROOT / "report" / "Task_03_Report.pdf"

for p in (NB1, NB2, NB3, CSV, META):
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run the notebooks first (they create the saved outputs and the models/ folder).")


# ------------------------------------------------------------------ read notebooks (single source of truth)
def load_code(path):
    return [c for c in json.load(open(path, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def cell_out(code, i):
    res = {"stream": "", "text": []}
    for o in code[i].get("outputs", []):
        if "text" in o:
            res["stream"] += "".join(o["text"])
        elif "data" in o and "text/plain" in o["data"]:
            res["text"].append("".join(o["data"]["text/plain"]))
    return res


def src(code, i):
    return "".join(code[i]["source"])


def _num(s):
    try:
        return float(s)
    except ValueError:
        return s


def parse_df(text, n_num=None):
    """Turn the text/plain of a displayed DataFrame into a DataFrame indexed by its first (index) column.

    The last `n_num` values of each line are numbers (read from the right, because a negative number may be separated by
    only one space); whatever is left of them is the index followed by any text columns, separated by 2+ spaces."""
    lines = [l.rstrip() for l in text.split("\n") if l.strip()]
    header = re.split(r"\s{2,}", lines[0].strip())
    n_num = len(header) if n_num is None else n_num
    rows = {}
    for line in lines[1:]:
        if len(line.split()) == 1:                  # the index-name line, e.g. "actual"
            continue
        pieces = line.rstrip().rsplit(None, n_num)
        lead = re.split(r"\s{2,}", pieces[0].strip())
        if len(lead) != len(header) - n_num + 1 or len(pieces) != n_num + 1:
            raise ValueError(f"cannot parse row {line!r} (header {header})")
        rows[lead[0]] = lead[1:] + [_num(x) for x in pieces[1:]]
    return pd.DataFrame.from_dict(rows, orient="index", columns=header)


C2, C3 = load_code(NB2), load_code(NB3)
assert len(C2) == 13 and len(C3) == 14, "notebook 02 should have 13 and notebook 03 should have 14 code cells"
for code, i, marker in [(C2, 1, "drop_duplicates"), (C2, 3, "SimpleImputer"), (C2, 5, "cv_score"), (C2, 10, "ablation"),
                        (C2, 11, "reduced_num"), (C3, 1, "base_models"), (C3, 3, "RandomizedSearchCV"), (C3, 5, "ranking"),
                        (C3, 6, "final_models"), (C3, 8, "bands"), (C3, 9, "permutation_importance"), (C3, 11, "cross_val_predict"),
                        (C3, 12, "joblib.dump"), (C3, 13, "subprocess")]:
    assert marker in src(code, i), f"code cell {i} is not the expected cell (looking for {marker})"

# notebook 02
n_raw, n_dup, n_clean = map(int, re.search(r"Raw rows: (\d+) \| duplicates removed: (\d+) \| rows kept: (\d+)", cell_out(C2, 1)["stream"]).groups())
n_train, n_test = map(int, re.search(r"Train: \((\d+), \d+\) \| Test: \((\d+), \d+\)", cell_out(C2, 1)["stream"]).groups())
med_train, med_test = [int(x.replace(",", "")) for x in re.search(r"train: ([\d,]+) \| test: ([\d,]+)", cell_out(C2, 1)["stream"]).groups()]
miss = parse_df(cell_out(C2, 3)["text"][0])
medians = {k: float(v) for k, v in re.findall(r"'(\w+)': (?:np\.float64\()?([\d.]+)", cell_out(C2, 3)["stream"])}
skew = {k: float(v) for k, v in re.findall(r"Skewness of (\w+) \(train\):\s+(-?[\d.]+)", cell_out(C2, 4)["stream"])}
n_brands, n_rare = map(int, re.search(r"Brands in train: (\d+) \| brands with < 10 rows \(grouped as 'infrequent'\): (\d+)", cell_out(C2, 4)["stream"]).groups())
fe_tab = parse_df(cell_out(C2, 5)["text"][0])
tgt_tab = parse_df(cell_out(C2, 6)["text"][0])
c7 = cell_out(C2, 7)["stream"]
n_cols_out = int(re.search(r"matrix: \(\d+, (\d+)\)", c7).group(1))
n_onehot = int(re.search(r"One-hot columns created: (\d+)", c7).group(1))
corr_pairs = parse_df(cell_out(C2, 8)["text"][1], n_num=1)
ablation = parse_df(cell_out(C2, 10)["text"][0], n_num=2)
joint = parse_df(cell_out(C2, 11)["text"][0], n_num=4)

# notebook 03
base = parse_df(cell_out(C3, 1)["text"][0])
tuned = parse_df(cell_out(C3, 4)["text"][0])
comp = parse_df(cell_out(C3, 4)["text"][1])
c4 = cell_out(C3, 4)["stream"]
best_params = {m.group(1): m.group(2) for m in re.finditer(r"^\s{2}(.+?): (\{.*\})$", c4, re.M)}
search_secs = {m.group(1).strip(): int(m.group(2)) for m in re.finditer(r"^(.+?)\s+tuned CV RMSLE [\d.]+ \| (\d+)s$", cell_out(C3, 3)["stream"], re.M)}
c5 = cell_out(C3, 5)["stream"]
selected = re.search(r"Selected model: (.+)", c5).group(1).strip()
runner_up, ru_rmsle, gap = re.search(r"Runner-up: (.+?) \(CV RMSLE ([\d.]+)\); gap = ([\d.]+)", c5).groups()
sel_noise = float(re.search(r"\(\+/- ([\d.]+) across folds\)", c5).group(1))
gap, ru_rmsle = float(gap), float(ru_rmsle)
test = parse_df(cell_out(C3, 6)["text"][0])
bands = parse_df(cell_out(C3, 8)["text"][0])
worst = parse_df(cell_out(C3, 8)["text"][1], n_num=4)
perm = parse_df(cell_out(C3, 9)["text"][0])
builtin = [(k, float(v)) for k, v in re.findall(r"^(\w+)\s+([\d.]+)$", cell_out(C3, 10)["stream"], re.M)]
q_low, q_high = map(float, re.search(r"predicted x \[([\d.]+), ([\d.]+)\]", cell_out(C3, 11)["stream"]).groups())
coverage = float(re.search(r"([\d.]+)% of test cars", cell_out(C3, 11)["stream"]).group(1))
c13 = cell_out(C3, 13)["stream"]
assert c13.count("identical to in-memory model: True") == 2 and "return code: 0" in c13, "saved-model checks did not all pass"

meta = json.load(open(META))
ver = meta["library_versions"]
size = {p.name: p.stat().st_size / 1e6 for p in (ROOT / "models").glob("best_model.*")}

# ------------------------------------------------------------------ read the CSV (dataset facts)
from src.car_features import RAW_INPUT_COLUMNS, CarFeatureEngineer, _leading_number  # noqa: E402

raw = pd.read_csv(CSV)
RAW_ROWS, RAW_COLS = raw.shape
clean = raw.drop_duplicates().reset_index(drop=True)
N = len(clean)
assert N == n_clean == n_train + n_test, "CSV and notebooks disagree about the number of clean rows"
parsed = CarFeatureEngineer().fit_transform(clean[RAW_INPUT_COLUMNS])
price = clean["selling_price"]
zero_mileage = int((_leading_number(clean["mileage"]) == 0).sum())
blank_power = int((clean["max_power"].notna() & ~clean["max_power"].fillna("").str.contains(r"\d")).sum())
kmkg = int(clean["mileage"].fillna("").str.contains("km/kg").sum())
kmkg_fuels = clean.loc[clean["mileage"].fillna("").str.contains("km/kg"), "fuel"].value_counts().to_dict()
raw_missing = raw.isnull().sum()
miss_cols = ["mileage", "engine", "max_power", "seats", "torque"]
any_missing = int(clean[miss_cols].isna().any(axis=1).sum())
all_missing = int(clean[miss_cols].isna().all(axis=1).sum())
train_df, test_df = pd.read_csv(ROOT / "data" / "processed" / "train.csv"), pd.read_csv(ROOT / "data" / "processed" / "test.csv")
big_train, big_test = int((train_df["selling_price"] > 3_000_000).sum()), int((test_df["selling_price"] > 3_000_000).sum())
big_expected = big_train / len(train_df) * len(test_df)
med_by = {c: clean.groupby(c)["selling_price"].median() for c in ["transmission", "fuel", "owner"]}
corr_price = parsed[["car_age", "km_driven", "mileage", "engine", "max_power", "seats"]].corrwith(np.log1p(price), method="spearman")

try:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                         capture_output=True, text=True, timeout=120).stdout
    N_TESTS = int(re.search(r"(\d+) tests? collected", out).group(1))
except Exception:
    N_TESTS = None

f4 = lambda x: f"{x:.4f}"
f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{x:.1f}%"
nice = lambda m: m.replace(" (optional)", " (optional)").replace("SVR (Support Vector Machine)", "SVR (SVM)")
MODELS = ["Decision Tree", "Random Forest", "SVR (Support Vector Machine)", "Gradient Boosting", "XGBoost (optional)"]

WARN = []


def check(cond, msg):
    if not cond:
        WARN.append(msg)


# ------------------------------------------------------------------ chart re-drawn from the notebook numbers
fig, ax = plt.subplots(figsize=(9, 3.9))
x = np.arange(len(MODELS))
b = [comp.loc[m, "CV RMSLE default"] for m in MODELS]
t = [comp.loc[m, "CV RMSLE tuned"] for m in MODELS]
r1 = ax.bar(x - 0.2, b, 0.4, label="Default settings", color="#1F77B4")
r2 = ax.bar(x + 0.2, t, 0.4, label="Tuned", color="#E8751A")
for rect in list(r1) + list(r2):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.004, f"{rect.get_height():.3f}", ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([nice(m).replace(" (", "\n(").replace("Gradient Boosting", "Gradient\nBoosting").replace("Random Forest", "Random\nForest").replace("Decision Tree", "Decision\nTree") for m in MODELS], fontsize=8)
ax.set_ylim(0, 0.34)
ax.set_ylabel("5-fold CV RMSLE (lower is better)")
ax.set_title("Cross-validated error before and after tuning (training set)", fontsize=10)
ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
BA_PNG = ROOT / "report" / "_before_after.png"
plt.savefig(BA_PNG, dpi=170)
plt.close()

# ------------------------------------------------------------------ styles (same look as the Task 01 / Task 02 reports)
basess = getSampleStyleSheet()
NAVY = colors.HexColor("#1F3A5F")
S = {
    "body": ParagraphStyle("body", parent=basess["BodyText"], fontName="Helvetica", fontSize=10, leading=14.5, spaceAfter=6),
    "h1": ParagraphStyle("h1", parent=basess["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=NAVY, spaceBefore=14, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", parent=basess["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, textColor=NAVY, spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "cap": ParagraphStyle("cap", parent=basess["BodyText"], fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"), alignment=TA_CENTER, spaceAfter=8),
    "cell": ParagraphStyle("cell", parent=basess["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11),
    "cellb": ParagraphStyle("cellb", parent=basess["BodyText"], fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.white),
    "cell_s": ParagraphStyle("cell_s", parent=basess["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9.2),
    "cellb_s": ParagraphStyle("cellb_s", parent=basess["BodyText"], fontName="Helvetica-Bold", fontSize=7.4, leading=9.2, textColor=colors.white),
    "bullet": ParagraphStyle("bullet", parent=basess["BodyText"], fontName="Helvetica", fontSize=10, leading=14, leftIndent=14, bulletIndent=2, spaceAfter=3),
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


def table(rows, widths, header=True, small=False, hilite=None, keep=True):
    cs, hs = ("cell_s", "cellb_s") if small else ("cell", "cellb")
    data = [[Paragraph(md(str(c)), S[hs] if (header and i == 0) else S[cs]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F2F5F9")) for i in range(2, len(rows), 2)]
    if hilite:
        style += [("BACKGROUND", (0, hilite), (-1, hilite), colors.HexColor("#FFF1CC"))]
    t.setStyle(TableStyle(style))
    return KeepTogether([t]) if (keep and len(rows) <= 12) else t     # short tables never split across pages


def picture(path, caption, max_w=16.5 * cm, max_h=10.5 * cm, lead=None):
    w, h = PILImage.open(path).size
    scale = min(max_w / w, max_h / h)
    return KeepTogether(list(lead or []) + [Image(str(path), width=w * scale, height=h * scale), Paragraph(md(caption), S["cap"])])


def code_listing(text, title, caption):
    text = text.encode("cp1252", "replace").decode("cp1252").rstrip()
    return [KeepTogether([P(title, "h2"), Preformatted(text, S["code"]), Paragraph(md(caption), S["cap"])])]


st = []

# ------------------------------------------------------------------ title page
st += [Spacer(1, 3.5 * cm), P("DEVIXO SOLUTIONS", "t_big"), Spacer(1, 0.3 * cm), P("AI/ML Internship Program", "t_mid"),
       Spacer(1, 1.2 * cm), P("**Task 03**", "t_mid"), P("End-to-End Machine Learning Project", "t_mid"), Spacer(1, 2.2 * cm),
       P(f"**Student Name:** {STUDENT_NAME}", "t_small"), P(f"**Internship Domain:** {INTERNSHIP_DOMAIN}", "t_small"),
       P(f"**Submission Date:** {SUBMISSION_DATE}", "t_small"), Spacer(1, 2.5 * cm),
       P("Project: Used-Car Price Prediction (regression) on the CarDekho Vehicle Dataset", "t_small"), PageBreak()]

# ------------------------------------------------------------------ 1-2
st += [P("1. Introduction", "h1"),
       P("This report documents Task 03 of the Devixo Solutions AI/ML Internship Program: a complete machine learning project from data preparation to a saved "
         "model and a prediction interface. It covers feature engineering and selection, model development, hyperparameter tuning with cross-validation, feature "
         "importance, saving the best model with Joblib and Pickle, and a bonus Streamlit app."),
       P("The dataset lists used cars with their age, kilometres driven, fuel, transmission, owner history, engine, power, mileage and name. The problem is a "
         "**regression** problem: predict the selling price of a car. (Accuracy, precision and recall from Task 02 do not apply to a numeric target; error "
         "measures such as MAE, RMSE, RMSLE and R2 are used instead.)"),
       P("The next table links each part of the Task 03 sheet to the section of this report that answers it.", "body"),
       table([["Task 03 requirement", "Report section"],
              ["Part 1: import, missing values, duplicates, feature engineering, feature selection, scaling", "6, 9, 10, 11"],
              ["Part 2: at least four models (Random Forest, SVM, Gradient Boosting, Decision Tree; XGBoost optional)", "12"],
              ["Part 3: hyperparameter tuning, cross-validation, feature importance", "13, 16"],
              ["Part 4: save the best model (Joblib and Pickle)", "17"],
              ["Part 5 documentation: dataset, workflow, model selection, results, limitations, future improvements", "3-9, 14-15, 19, 20"],
              ["Bonus: Streamlit prediction interface", "18"]], [12 * cm, 5 * cm]),
       P("2. Objective", "h1"),
       P("The objective is to build a clean, reproducible machine learning pipeline on a real-world dataset: prepare the data without leaking test information, "
         "compare at least four algorithms fairly, tune them with cross-validation, choose a model by evidence, explain what it learned, and save it so that "
         "it can be reused.")]

# ------------------------------------------------------------------ 3-6
st += [P("3. Dataset Name", "h1"), P("**Vehicle dataset from CarDekho**, file `Car details v3.csv` (stored in the project as `data/raw/car_details_v3.csv`)."),
       P("4. Dataset Source", "h1"),
       P(f"Kaggle: {KAGGLE_URL}. The rows are used-car listings from the CarDekho website (India). The newest car is from "
         f"{int(clean['year'].max())}, so the listings are probably from around that year or earlier and the findings describe that market and period. The price column has no stated currency; it is assumed to be Indian rupees. "
         "The licence should be checked on the Kaggle page before the data is reused."),
       P("5. Dataset Description", "h1"),
       P("Each row is one used car. The columns describe the car (name, manufacturing year, fuel, transmission), its use (kilometres driven, number of owners, "
         "seller type), its specification (mileage, engine size, maximum power, torque, number of seats) and the selling price, which is the target."),
       P("6. Dataset Size", "h1")]
st += [table([["", "Raw dataset", "After preprocessing"],
              ["Rows (cars)", f"{RAW_ROWS:,}", f"{N:,}"],
              ["Columns", str(RAW_COLS), f"{RAW_COLS} (9 of them are used by the model)"],
              ["Exact duplicate rows", f"{n_dup:,} ({n_dup / RAW_ROWS * 100:.1f}%)", "0 (removed)"],
              ["Rows with a missing value in mileage, engine, max_power, seats or torque", f"{int(raw[miss_cols].isna().any(axis=1).sum())}", f"{any_missing} (about {any_missing / N * 100:.1f}%; imputed inside the pipeline)"],
              ["Train / test rows", "-", f"{n_train:,} / {n_test:,} (80% / 20%)"],
              ["File size", f"about {CSV.stat().st_size / 1e6:.2f} MB", "-"]], [7.2 * cm, 4.4 * cm, 5.4 * cm]),
       Spacer(1, 4),
       P(f"The task sheet requires at least 3,000 rows. After removing duplicates the dataset still has **{N:,}** rows, so the requirement is met.")]

# ------------------------------------------------------------------ 7
st += [P("7. Features/Columns Description", "h1"),
       table([["Group", "Columns", "Meaning"],
              ["Identity", "name", "Brand, model and variant in one text (for example 'Maruti Swift Dzire VDI'). Too many distinct values to use directly, so the brand is extracted."],
              ["Age and use", "year, km_driven, owner", "Manufacturing year, kilometres driven, and number of previous owners (First Owner to Test Drive Car)"],
              ["Type", "fuel, transmission, seller_type", "Diesel, Petrol, CNG or LPG; Manual or Automatic; Individual, Dealer or Trustmark Dealer"],
              ["Specification", "mileage, engine, max_power, torque, seats", "Stored as text with units ('23.4 kmpl', '1248 CC', '74 bhp'); torque is free text with mixed units"],
              ["Target", "selling_price", "Price of the car (assumed rupees); heavily right-skewed"]], [2.8 * cm, 4.6 * cm, 9.6 * cm])]

# ------------------------------------------------------------------ 8
st += [P("8. Methodology", "h1"),
       P(f"The work was done in Python {ver['python']} with Pandas {ver['pandas']}, NumPy {ver['numpy']}, Scikit-learn {ver['scikit-learn']}, XGBoost {ver.get('xgboost', '-')}, Matplotlib and "
         "Seaborn in Jupyter notebooks, with reusable code in `src/`. The random seed is 42 everywhere. The workflow is:"),
       *bullets(["**Notebook 01 - Data audit and EDA:** read-only inspection of the raw file; records every data problem and the decision it leads to.",
                 "**Notebook 02 - Preparation and feature work:** remove duplicates, split 80/20 **before** learning anything from the data, parse the text columns, engineer and select features, scale.",
                 "**Notebook 03 - Modelling:** compare five models and two reference baselines with 5-fold cross-validation, tune with `RandomizedSearchCV`, choose the best by cross-validation, evaluate once on the test set, explain the model, save it.",
                 "**One pipeline:** feature engineering, imputation, scaling, encoding, the model and the log-price conversion form one scikit-learn pipeline. Only training data is used to fit it, so the test set cannot leak, and the saved model accepts a raw car description.",
                 f"**Quality checks:** {N_TESTS if N_TESTS else 'a set of'} automated tests (`pytest`) check the parsing code, the data split, the saved model and the Streamlit app."])]

# ------------------------------------------------------------------ 9 preparation
st += [P("9. Part 1 - Data Preparation", "h1"),
       P("Every action is recorded below with its reason. The raw file was never changed; all work was done on copies."),
       table([["Step", "Action", "Affected", "Reason"],
              ["1", "Loaded the CSV", f"{RAW_ROWS:,} rows, {RAW_COLS} columns", "The task requires at least 3,000 rows."],
              ["2", "Removed exact duplicate rows", f"{n_dup:,} rows",
               "Identical in all 13 columns, including exact price and kilometres. Done **before** the split so that a listing cannot be in both train and test. Assumption: they are repeated listings (the file has no listing ID)."],
              ["3", "Split 80% / 20% (`random_state=42`)", f"{n_train:,} train / {n_test:,} test",
               f"Done before any statistic is learned. Median price is {med_train:,} in train and {med_test:,} in test. The test set is not used again until the final evaluation."],
              ["4", "Parsed `mileage`, `engine`, `max_power` from text to numbers", "3 columns",
               f"Models need numbers. Zero mileage ({zero_mileage} rows) and a blank power value ({blank_power} row in the clean data) are not real measurements, so they become missing."],
              ["5", "Filled missing numbers with the median (inside the pipeline, fitted on train only)",
               f"mileage {int(miss.loc['mileage', 'missing_after_parsing'])}, engine {int(miss.loc['engine', 'missing_after_parsing'])}, max_power {int(miss.loc['max_power', 'missing_after_parsing'])} (train)",
               f"About 3% of rows, missing together ({all_missing} rows have all five columns missing), so they cannot be rebuilt from each other and dropping them would waste data. Medians learned from training data: mileage {medians.get('mileage', 0):.1f}, engine {medians.get('engine', 0):,.0f}, max_power {medians.get('max_power', 0):.2f}."],
              ["6", "Engineered `car_age`, `brand`, `log_km_driven`", "3 new features", "Explained and tested in section 11."],
              ["7", "Selected 9 features, dropped `seats`, `seller_type`, `torque`, `name`, `year`, `km_driven`", "see section 11", "Chosen by experiment, not by guess."],
              ["8", "Scaled the 5 numeric features with `StandardScaler`", "5 columns", "Needed by SVR; harmless for trees. Fitted on training data only, inside the pipeline."],
              ["9", "One-hot encoded `brand`, `fuel`, `transmission`, `owner`", f"{n_onehot} new columns",
               f"Categories with fewer than 10 training cars are grouped as 'infrequent' ({n_rare} of {n_brands} brands). Unseen categories are handled safely. After preprocessing the training matrix has {n_cols_out} columns."],
              ["10", "Trained on log(price) and converted predictions back to rupees", "target", "Price is very skewed; checked by experiment in section 11."]],
             [1.2 * cm, 4.9 * cm, 3.3 * cm, 7.6 * cm]),
       Spacer(1, 4),
       P(f"Result: {N:,} clean rows (the requirement of at least 3,000 is met) and a single pipeline that turns a raw car description into a model-ready row without using the test set.")]

# ------------------------------------------------------------------ 10 EDA
num = clean[["year", "km_driven", "selling_price"]].agg(["mean", "median", "std", "min", "max"]).T
rows = [["Variable", "Mean", "Median", "Std. dev.", "Min", "Max"]] + [[k] + [f"{v:,.0f}" for v in r.values] for k, r in num.iterrows()]
check(med_by["transmission"]["Automatic"] > med_by["transmission"]["Manual"], "automatic cars are no longer dearer than manual ones")
check(med_by["fuel"]["Diesel"] > med_by["fuel"]["Petrol"], "diesel cars are no longer dearer than petrol cars")
check(corr_price["car_age"] < -0.6 and corr_price["max_power"] > 0.5, "car_age / max_power correlations changed")
st += [P("10. Exploratory Data Analysis", "h1"),
       P("Statistics of the numeric columns (cleaned data, all rows):"), table(rows, [4 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]), Spacer(1, 6),
       P(f"Fuel: Diesel ({int((clean['fuel'] == 'Diesel').sum()):,}), Petrol ({int((clean['fuel'] == 'Petrol').sum()):,}), CNG ({int((clean['fuel'] == 'CNG').sum())}), LPG ({int((clean['fuel'] == 'LPG').sum())}). "
         f"Transmission: Manual ({int((clean['transmission'] == 'Manual').sum()):,}), Automatic ({int((clean['transmission'] == 'Automatic').sum()):,}). "
         f"Mileage is in km/kg for {kmkg} cars, all of them {' or '.join(kmkg_fuels)} cars ({', '.join(f'{k} {v}' for k, v in kmkg_fuels.items())}), and in kmpl for the rest."),
       picture(FIGDIR / "eda_target_distribution.png", f"Figure 10.1: the price is very skewed (skewness {price.skew():.2f}); the log scale is much more symmetric (skewness {np.log1p(price).skew():.2f})", max_h=6 * cm),
       picture(FIGDIR / "eda_price_by_category.png", "Figure 10.2: price by fuel, transmission, owner and seller type (log scale)", max_h=9 * cm),
       P(f"The median price is {med_by['transmission']['Automatic']:,.0f} for automatic and {med_by['transmission']['Manual']:,.0f} for manual cars, {med_by['fuel']['Diesel']:,.0f} for diesel and "
         f"{med_by['fuel']['Petrol']:,.0f} for petrol, and {med_by['owner']['First Owner']:,.0f} for first owners against {med_by['owner']['Fourth & Above Owner']:,.0f} for fourth owners and above. "
         f"Rank correlations with log price: car age {corr_price['car_age']:.2f}, maximum power {corr_price['max_power']:.2f}, engine {corr_price['engine']:.2f}, seats {corr_price['seats']:.2f}, "
         f"kilometres {corr_price['km_driven']:.2f}, mileage {corr_price['mileage']:.2f}. These are associations, not causes, and several features move together (for example power and engine size)."),
       P("**Suspicious values.** Two cars show more than one million kilometres (the highest is 2,360,457 km on a 2007 car), which looks like a typing error. They were kept: there are only two, "
         "and the log transform limits their influence. The highest price (10,000,000 for a Volvo XC90 T8) is plausible for a luxury car and was not treated as an error.")]

# ------------------------------------------------------------------ 11 features
check(ablation.loc["1", "feature_removed"] == "car_age", "car_age is no longer the most important feature in the drop-one test")
small_ab = {r["feature_removed"]: r["change_vs_full"] for _, r in ablation.iterrows()}
check(small_ab["seats"] < 0.0006 and small_ab["seller_type"] < 0.0006, "seats / seller_type no longer have a negligible effect")
check(fe_tab.loc["B. A + brand", "RMSLE"] < fe_tab.loc["A. parsed numbers, no brand", "RMSLE"] - 0.005, "adding brand no longer clearly helps")
check(abs(fe_tab.loc["D. C + km_per_year + bhp_per_litre", "RMSLE"] - fe_tab.loc["C. B, log(km) instead of km", "RMSLE"]) < 0.003, "km_per_year / bhp_per_litre now change the error")
rows = [["Experiment (Random Forest, 5-fold CV)", "RMSLE", "R2", "MAE"]] + [[k, f4(r["RMSLE"]), f4(r["R2"]), f"{r['MAE']:,.0f}"] for k, r in fe_tab.iterrows()]
rows_ab = [["Feature removed", "RMSLE", "Increase vs all 11 features"]] + [[r["feature_removed"], f4(r["RMSLE"]), f"{r['change_vs_full']:+.4f}"] for _, r in ablation.iterrows()]
rows_j = [["Model", "RMSLE all 11", "RMSLE reduced to 9", "Difference", "Fold noise (std)"]] + [[r["model"], f4(r["RMSLE_all_11"]), f4(r["RMSLE_reduced_9"]), f"{r['difference']:+.4f}", f4(r["fold_noise_(std)"])] for _, r in joint.iterrows()]
st += [P("11. Feature Engineering and Feature Selection", "h1"),
       P("**Feature engineering.** `car_age` is 2020 minus the year; `brand` is the first word of the name ('Land' is corrected to 'Land Rover'); `log_km_driven` is the logarithm of the kilometres "
         f"(skewness {skew['km_driven']:.2f} before, {skew['log_km_driven']:.2f} after). Two more were tried: `km_per_year` and `bhp_per_litre`. Every choice was tested with 5-fold cross-validation on the training set; "
         "the score is RMSLE (error on the log-price scale, roughly a typical percentage error; lower is better) and the fold-to-fold noise is about 0.01."),
       table(rows, [8.2 * cm, 2.6 * cm, 2.6 * cm, 3.6 * cm]), Spacer(1, 4),
       *bullets([f"**Brand helps:** the error falls from {f4(fe_tab.loc['A. parsed numbers, no brand', 'RMSLE'])} to {f4(fe_tab.loc['B. A + brand', 'RMSLE'])} (about one fold-noise standard deviation, and consistent with the drop-one test below).",
                 "**Log kilometres and the two extra features made no measurable difference for a Random Forest** (trees ignore monotonic transforms). `log_km_driven` is kept because the skew is a known problem for SVR; `km_per_year` and `bhp_per_litre` are **not used** because they did not help.",
                 f"**Log target:** RMSLE {f4(tgt_tab.loc['log(price) target', 'RMSLE'])} with log(price) against {f4(tgt_tab.loc['raw price target', 'RMSLE'])} without, and R2 {f4(tgt_tab.loc['log(price) target', 'R2'])} against {f4(tgt_tab.loc['raw price target', 'R2'])}. The effect is small for a forest; the log target is kept because it matches the relative-error metric and suits SVR."]),
       P("**Feature selection.** Three checks on training data. (1) Correlation: only two pairs of numeric features reach 0.6 "
         f"(engine and max_power {float(corr_pairs.iloc[0]['corr']):.2f}; engine and seats {float(corr_pairs.iloc[1]['corr']):.2f}), so no feature is removed for redundancy alone. "
         "(2) Mutual information (a screening score for each feature alone) was computed, but it measures each feature in isolation, so it ranks car age lower than the better test below. "
         "(3) Drop-one-feature test: remove one feature at a time and measure how much the error rises."),
       table(rows_ab, [5 * cm, 3.5 * cm, 6 * cm]), Spacer(1, 4),
       picture(FIGDIR / "fs_ablation.png", "Figure 11.1: increase in cross-validated error when each feature is removed (bigger bar = more useful feature)", max_w=12 * cm, max_h=7.5 * cm),
       P(f"`seats` ({small_ab['seats']:+.4f}) and `seller_type` ({small_ab['seller_type']:+.4f}) have almost no effect. Removing both together was checked on three model types:"),
       table(rows_j, [3.8 * cm, 3.1 * cm, 3.6 * cm, 3 * cm, 3.5 * cm]), Spacer(1, 4),
       P("All differences are far smaller than the fold noise, so both features were dropped. `fuel`, `transmission` and `engine` have small effects too, but they were kept: the removal threshold was strict, and they have an obvious link to price. "
         "**Final features (9):** `car_age`, `log_km_driven`, `mileage`, `engine`, `max_power` (numeric) and `brand`, `fuel`, `transmission`, `owner` (categorical). `torque` was not used because it is free text with mixed units.")]

# ------------------------------------------------------------------ 12 models
base_rows = [["Model", "CV RMSLE", "CV std", "CV R2", "CV MAE", "Fit time (s)"]]
for m, r in base.iterrows():
    base_rows.append([m, f4(r["RMSLE"]), f4(r["RMSLE_std"]), f4(r["R2"]), f"{r['MAE']:,.0f}", f3(r["fit_seconds"])])
check(base.loc["Baseline (always median price)", "R2"] < 0.01, "the median baseline R2 is not about 0")
st += [P("12. Part 2 - Model Development", "h1"),
       P("Every model is placed in the **same pipeline** (feature engineering, imputation, scaling, encoding, log target) and trained on the same training set, so the comparison is fair. "
         "The first five rows of the next table are the required models plus the optional XGBoost; Ridge regression and an 'always predict the median price' model are added as reference points."),
       table([["Model", "Role", "Needs scaling?", "Default settings used"],
              ["Decision Tree", "Required", "No", "defaults (no depth limit)"],
              ["Random Forest", "Required", "No", "200 trees"],
              ["SVR (Support Vector Machine)", "Required", "Yes", "defaults (RBF kernel)"],
              ["Gradient Boosting", "Required", "No", "defaults"],
              ["XGBoost", "Optional in the task", "No", "defaults (`hist` method)"],
              ["Ridge regression", "Reference", "Yes", "`alpha=1`"],
              ["Always predict the median", "Reference floor", "-", "-"]], [5 * cm, 3.6 * cm, 3 * cm, 5.4 * cm]), Spacer(1, 6),
       P("**Metrics.** RMSLE is the selection metric: prices run from about 30 thousand to 10 million, and RMSLE measures relative error, so a 10% miss on a cheap and on an expensive car count the same. "
         "R2 is the share of price variation explained; MAE and RMSE are in rupees (RMSE punishes large misses and is dominated by expensive cars). MedAPE is the median absolute percentage error."),
       P("**Cross-validation results with default settings (training set, 5 folds):**"),
       table(base_rows, [5.6 * cm, 2.2 * cm, 2 * cm, 2.2 * cm, 2.6 * cm, 2.4 * cm]), Spacer(1, 6),
       picture(FIGDIR / "model_baseline_cv.png", "Figure 12.1: default-setting models, 5-fold cross-validated RMSLE (bars show the spread over folds)", max_w=13 * cm, max_h=7.5 * cm),
       *bullets([f"The always-median model has RMSLE {f4(base.loc['Baseline (always median price)', 'RMSLE'])} and R2 {f4(base.loc['Baseline (always median price)', 'R2'])}, so the metrics are wired correctly (R2 near 0 is expected).",
                 f"All tree-based models and SVR beat Ridge regression ({f4(base.loc['Ridge regression (reference)', 'RMSLE'])}); the best default models (XGBoost, SVR, Random Forest) are within about 0.002 of each other, which is far less than the fold noise.",
                 f"The unpruned Decision Tree is the weakest real model ({f4(base.loc['Decision Tree', 'RMSLE'])}); it overfits when it has no depth limit."])]

# ------------------------------------------------------------------ 13 tuning
check(comp.loc["SVR (Support Vector Machine)", "improvement"] <= 0.001, "tuning now helps SVR")
check(all(comp.loc[m, "improvement"] > 0.005 for m in MODELS if not m.startswith("SVR")), "tuning no longer helps every tree-based model")
check(tuned["RMSLE"].min() < base.loc["Ridge regression (reference)", "RMSLE"] - 0.03, "tuned models no longer clearly beat Ridge")
check(re.search(r"'C': 1(?![\d.])", best_params["SVR (Support Vector Machine)"]) is not None, "SVR best C is no longer at the edge of its grid (1)")
grids = [["Model", "Parameters searched (15 random combinations each)"],
         ["Decision Tree", "`max_depth` {4, 6, 8, 10, 12, 15, None}; `min_samples_leaf` {1, 2, 5, 10, 20}; `min_samples_split` {2, 5, 10, 20}"],
         ["Random Forest", "`n_estimators` {100, 200, 300}; `max_depth` {None, 10, 15, 20}; `min_samples_leaf` {1, 2, 4}; `max_features` {sqrt, 0.5, 1.0}"],
         ["Gradient Boosting", "`n_estimators` {100, 200, 300, 400}; `learning_rate` {0.03, 0.05, 0.1, 0.2}; `max_depth` {2, 3, 4, 5}; `subsample` {0.7, 0.85, 1.0}; `min_samples_leaf` {1, 5, 10}"],
         ["SVR (SVM)", "`C` {1, 3, 10, 30, 100}; `epsilon` {0.01, 0.05, 0.1, 0.2}; `gamma` {scale, 0.01, 0.03, 0.1}"],
         ["XGBoost", "`n_estimators` {200, 400, 600}; `learning_rate` {0.03, 0.05, 0.1}; `max_depth` {3, 4, 5, 6}; `subsample` {0.7, 0.85, 1.0}; `colsample_bytree` {0.6, 0.8, 1.0}; `min_child_weight` {1, 3, 5}"]]
rows = [["Model", "Best parameters found", "CV RMSLE", "CV std", "CV R2", "CV MAE", "Search (s)"]]
for m in MODELS:
    r = tuned.loc[m]
    rows.append([nice(m), best_params[m].replace("'", ""), f4(r["RMSLE"]), f4(r["RMSLE_std"]), f4(r["R2"]), f"{r['MAE']:,.0f}", str(search_secs[m])])
rows_c = [["Model", "Default CV RMSLE", "Tuned CV RMSLE", "Improvement"]] + [[nice(m), f4(comp.loc[m, "CV RMSLE default"]), f4(comp.loc[m, "CV RMSLE tuned"]), f"{comp.loc[m, 'improvement']:+.4f}"] for m in MODELS]
st += [P("13. Part 3 - Model Optimization (Hyperparameter Tuning and Cross-Validation)", "h1"),
       P("**Design.** `RandomizedSearchCV` tries 15 random combinations of settings per model. Every combination is scored with **5-fold cross-validation on the training set only**: the training data is cut into five parts, "
         "the model is trained on four and scored on the fifth, five times, and the scores are averaged. The score is RMSLE. Random search was used instead of a full grid because it covers wide ranges in much less time. "
         "After the search each model is refitted on the whole training set with its best settings. The test set plays no part in tuning."),
       KeepTogether([P("13.1 Settings searched", "h2"), table(grids, [3.2 * cm, 13.8 * cm], small=True, keep=False)]),
       KeepTogether([P("13.2 Results", "h2"), table(rows, [2.6 * cm, 5.4 * cm, 1.6 * cm, 1.4 * cm, 1.5 * cm, 1.7 * cm, 1.8 * cm], small=True, keep=False)]), Spacer(1, 6),
       table(rows_c, [5 * cm, 4 * cm, 4 * cm, 4 * cm]), Spacer(1, 6),
       picture(BA_PNG, "Figure 13.1: cross-validated RMSLE before and after tuning (re-drawn from the notebook's numbers)", max_h=8 * cm),
       *bullets([f"**Tuning helped every tree-based model** (improvements {comp.loc['Random Forest', 'improvement']:.4f} to {comp.loc['Decision Tree', 'improvement']:.4f}); the Decision Tree gained most because its depth limit was tuned.",
                 f"**Tuning did not help SVR** ({comp.loc['SVR (Support Vector Machine)', 'improvement']:+.4f}). Its best `C` is 1, the smallest value offered, so a wider range (smaller `C`) might do better.",
                 f"All tuned tree-based models ({f4(tuned['RMSLE'].min())} to {f4(tuned.loc['Random Forest', 'RMSLE'])}) beat Ridge regression ({f4(base.loc['Ridge regression (reference)', 'RMSLE'])}), so the extra complexity is justified.",
                 "15 random combinations is a modest search, and several winning values lie inside large grids, so a longer search might change small differences between models."])]

# ------------------------------------------------------------------ 14 selection and test
check(selected == tuned["RMSLE"].idxmin(), "selected model is not the lowest tuned CV RMSLE")
check(gap < sel_noise, "the gap between the top two models is no longer smaller than the fold noise")
check(test.loc[runner_up, "R2"] > test.loc[selected, "R2"], "the runner-up no longer has the higher test R2 (the 'tie' sentence)")
check(test.loc[selected, "RMSLE"] <= test["RMSLE"].iloc[:5].min() + 1e-9, "the selected model no longer has the best test RMSLE")
check(test.loc[selected, "R2"] > tuned.loc[selected, "R2"] + 0.02, "test R2 is no longer clearly higher than cross-validation R2")
check(big_test < big_expected * 0.75, "the test set no longer has fewer very expensive cars than a fair share")
rows = [["Model", "R2", "MAE (rupees)", "RMSE (rupees)", "RMSLE", "Median % error"]]
sel_row = 1
for k, (m, r) in enumerate(test.iterrows(), start=1):
    name = "Always predict the median" if m.startswith("Baseline") else nice(m)
    if m == selected:
        name, sel_row = name + " (selected)", k
    rows.append([name, f4(r["R2"]), f"{r['MAE']:,.0f}", f"{r['RMSE']:,.0f}", f4(r["RMSLE"]), pct(r["MedAPE_%"])])
sel = test.loc[selected]
st += [P("14. Model Selection and Final Test Results", "h1"),
       P(f"**Selection rule, fixed before the test set was used:** the model with the lowest tuned cross-validated RMSLE wins. The code applies it. **{nice(selected).replace(' (optional)', '')}** was selected "
         f"(CV RMSLE {f4(tuned.loc[selected, 'RMSLE'])}, fold spread {sel_noise:.4f}). Its settings: {best_params[selected].replace(chr(39), '')}."),
       P(f"**It is effectively a tie.** The runner-up, {nice(runner_up)}, is only {gap:.4f} behind, much less than the fold-to-fold noise ({sel_noise:.4f}). Changing the choice after seeing test scores would be selecting a model with the test set, so the rule was kept."),
       P("The test set (1,386 cars) was then used once. All tuned models are shown for transparency; the choice above was made from cross-validation alone."),
       table(rows, [4.8 * cm, 1.9 * cm, 2.6 * cm, 2.8 * cm, 2.0 * cm, 2.9 * cm], hilite=sel_row), Spacer(1, 6),
       *bullets([f"Selected model on the test set: median error **{pct(sel['MedAPE_%'])}** of the price, MAE **{sel['MAE']:,.0f}**, RMSLE **{f4(sel['RMSLE'])}** (always guessing the median: {f4(test.loc['Baseline (always median price)', 'RMSLE'])}), R2 {f4(sel['R2'])}.",
                 f"{nice(runner_up)} scores slightly higher on test R2 ({f4(test.loc[runner_up, 'R2'])}) and MAE ({test.loc[runner_up, 'MAE']:,.0f}), the selected model on RMSLE: consistent with a tie.",
                 f"**Why test R2 ({f4(sel['R2'])}) is higher than cross-validation R2 ({f4(tuned.loc[selected, 'R2'])}):** R2 is dominated by a few very expensive cars, and the test set happens to contain only {big_test} cars above 30 lakh, while the training set has {big_train} (a fair share would be about {big_expected:.0f}). "
                 f"RMSLE hardly moves ({f4(tuned.loc[selected, 'RMSLE'])} in cross-validation, {f4(sel['RMSLE'])} on test), so **RMSLE and the median percentage error are the more trustworthy summaries**."])]

# ------------------------------------------------------------------ 15 error analysis
check(bands["median_abs_pct_error"].astype(float).idxmax() == "< 2 lakh", "the cheapest band is no longer the least accurate")
check(float(bands.loc["< 2 lakh", "median_signed_pct_error"]) > 0 and float(bands.loc["> 10 lakh", "median_signed_pct_error"]) < 0, "the over/under-pricing pattern changed")
check(abs(coverage - 80) < 3, "the indicative price range no longer covers about 80% of test cars")
rows = [["Actual price band", "Cars", "Median % error", "Median signed % error", "Mean abs. error (rupees)"]]
for b_, r in bands.iterrows():
    rows.append([b_, f"{int(r['cars'])}", f"{r['median_abs_pct_error']:.1f}%", f"{r['median_signed_pct_error']:+.1f}%", f"{r['mean_abs_error_rupees']:,.0f}"])
w1, w_last = worst.iloc[0], worst.iloc[-1]
n_over = int((worst["pct_error"].astype(float) > 0).sum())
st += [P("15. Error Analysis of the Selected Model", "h1"),
       P("A single score can hide where a model fails, so the errors were examined by price band and one by one."),
       picture(FIGDIR / "model_error_analysis.png", "Figure 15.1: left: predicted against actual price on the test set (log scales, red line = perfect); right: distribution of the errors, centred near zero", max_h=7.5 * cm),
       table(rows, [3.4 * cm, 1.8 * cm, 3.2 * cm, 4.2 * cm, 4.4 * cm]), Spacer(1, 4),
       P("1 lakh = 100,000 rupees. A positive signed error means the model predicted too high."),
       *bullets([f"The model is least accurate for **cars under 2 lakh** (median error {bands.loc['< 2 lakh', 'median_abs_pct_error']:.1f}%) and tends to over-price them; it **under-prices cars above 10 lakh** (median {float(bands.loc['> 10 lakh', 'median_signed_pct_error']):+.1f}%), which is typical of a model that pulls extreme values toward the average.",
                 f"Of the {len(worst)} largest percentage errors, {n_over} are over-estimates of cheap, old cars (for example {w1['name']}: actual {int(w1['actual']):,}, predicted about {float(w1['predicted']):,.0f}); the other is {w_last['name']} (actual {int(w_last['actual']):,}, predicted about {float(w_last['predicted']):,.0f}).",
                 f"An indicative price range of the predicted price x {q_low:.2f} to x {q_high:.2f} (the middle 80% of out-of-fold errors on training data) covered **{coverage:.1f}%** of the untouched test cars."])]

# ------------------------------------------------------------------ 16 importance
check(list(perm.index[:2]) == ["year", "max_power"], "the two most important permutation features changed")
check(builtin[0][0] != perm.index[0].replace("year", "car_age"), "built-in and permutation importance now agree on the top feature (re-read the paragraph)")
rows = [["Input column", "Importance (RMSLE increase)", "Spread (std)"]] + [[k, f4(r["importance_(RMSLE_increase)"]), f4(r["std"])] for k, r in perm.iterrows()]
st += [P("16. Feature Importance", "h1"),
       P("**Permutation importance (main result).** One input column of the test set is shuffled at a time, and the rise in error is measured (10 repeats). It works for any model and measures real predictive value on unseen data."),
       picture(FIGDIR / "feature_importance_permutation.png", "Figure 16.1: permutation importance; `name` carries the brand and `year` carries the car age", max_w=12 * cm, max_h=7.5 * cm),
       table(rows, [5 * cm, 6 * cm, 4 * cm]), Spacer(1, 6),
       P(f"**Car age (`year`, {perm.iloc[0]['importance_(RMSLE_increase)']:.3f}) and maximum power ({perm.iloc[1]['importance_(RMSLE_increase)']:.3f}) dominate**, followed by brand ({perm.loc['name', 'importance_(RMSLE_increase)']:.3f}), fuel ({perm.loc['fuel', 'importance_(RMSLE_increase)']:.3f}) and engine size ({perm.loc['engine', 'importance_(RMSLE_increase)']:.3f}). "
         "This agrees with the drop-one test in section 11."),
       P(f"**Built-in importance** of the XGBoost model gives a different order ({', '.join(f'{k} {v:.2f}' for k, v in builtin[:4])}, ...): it is summed over one-hot columns and favours features with many splits, "
         "so the permutation result is treated as the main one. Importance shows what the model **used**, not what **causes** a price; correlated features (power and engine) share credit.")]

# ------------------------------------------------------------------ 17 saving
st += [P("17. Part 4 - Save the Best Model (Joblib and Pickle)", "h1"),
       P("The **whole pipeline** is saved (feature engineering, imputation, scaling, encoding, model and the log-price conversion), so the saved file accepts a raw car description and returns a price."),
       table([["File", "Size", "Purpose"],
              ["`models/best_model.joblib`", f"{size['best_model.joblib']:.2f} MB", "Joblib format (compressed)"],
              ["`models/best_model.pkl`", f"{size['best_model.pkl']:.2f} MB", "Pickle format"],
              ["`models/model_metadata.json`", "small", "Model name, settings, metrics, input ranges, library versions"]], [6 * cm, 2.6 * cm, 8.4 * cm]),
       Spacer(1, 4),
       P("**Verification (never trust a save that was not reloaded):**"),
       *bullets(["Both files reload and give predictions identical to the in-memory model on the whole test set.",
                 "A brand-new Python process loads both files and produces identical predictions.",
                 "The model handles missing specifications and a brand it has never seen (a Tesla, absent from training) and still returns a price.",
                 f"Saved with scikit-learn {ver['scikit-learn']} and XGBoost {ver.get('xgboost', '-')}; `requirements.txt` pins the versions because a saved model can only be loaded safely with the same versions. Pickle files must only be loaded from trusted sources.",
                 "**Recovery path (found when the model was first run on another computer):** with different library versions, loading failed with `XGBoostError: input stream corrupted`. `python -m src.retrain` now rebuilds the model in seconds from the saved best settings, using the libraries installed on that computer, and prints its test scores next to the original ones (they matched exactly when tested). The app also explains the problem and both fixes on screen instead of crashing."])]

# ------------------------------------------------------------------ 18 bonus
st += [P("18. Bonus - Streamlit Prediction Interface", "h1"),
       P("`app.py` loads the saved Joblib pipeline and lets a user enter a car's brand, year, kilometres, fuel, transmission, owner, mileage, engine size and power. It builds one raw-format row and calls `predict`. "
         f"It shows the estimated price and a likely range (predicted price x {q_low:.2f} to x {q_high:.2f}), warns when an input is outside the range seen in training (for example very high kilometres), "
         "and explains its limits in an 'About this model' box. Run it with `streamlit run app.py`."),
       P(f"The app is covered by automated tests (Streamlit's headless test runner starts the app, clicks the button and checks the result and the warnings) and was started on a real server to check that it loads. "
         "No screenshot is included in this report.")]

# ------------------------------------------------------------------ 19 conclusion
st += [P("19. Conclusion and Recommendation", "h1"),
       P(f"I built a complete, leak-free pipeline on {N:,} used-car listings: removed {n_dup:,} duplicate rows, split the data before learning anything from it, parsed and engineered features, chose features by experiment, "
         f"compared four required models plus XGBoost with cross-validation, tuned them, selected one by a rule fixed in advance, evaluated it once on the test set, and saved it in two formats."),
       P(f"**Recommendation: use the tuned {nice(selected).replace(' (optional)', '')} model** (or, equivalently, {nice(runner_up)}, which is statistically tied). On the untouched test set it has a median error of {pct(sel['MedAPE_%'])} of the price "
         f"and an RMSLE of {f4(sel['RMSLE'])}, against {f4(test.loc['Baseline (always median price)', 'RMSLE'])} for always guessing the median. It is most reliable for mid-priced cars (2 to 10 lakh) and should be used with the indicative price range, "
         "not as a point valuation."),
       P("From this project I learned that the split must come before any fitting, that one pipeline prevents leakage and makes saving simple, that features should be kept or dropped on evidence, that R2 can mislead when a few expensive items dominate, "
         "that two models can be tied and the honest answer is to say so, and that a saved model must be reloaded and tested.")]

# ------------------------------------------------------------------ 20-21 limitations / future
st += [P("20. Limitations", "h1"),
       *bullets([f"**Old market data:** the newest car is from {int(clean['year'].max())} and the prices are listing prices from that period, so the model is **not valid for current prices**.",
                 f"**Uneven accuracy:** weaker for very cheap old cars and for luxury cars (only {int(bands.loc['> 10 lakh', 'cars'])} test cars are above 10 lakh).",
                 f"**Rare categories** (Test Drive Car owners, CNG and LPG fuel, {n_rare} rare brands) have too few rows to learn from and are grouped as 'infrequent'.",
                 "**Duplicate removal is an assumption**, and near-duplicate listings (same model, similar details) may remain on both sides of the split, which could make scores slightly optimistic.",
                 "**Random split, not a time split:** it does not test performance on future years.",
                 "**Unused columns:** `torque`, `seats` and `seller_type` were not used.",
                 "**Modest tuning:** 15 random combinations per model; SVR's best value was at the edge of its range. A single train/test split also leaves some noise in the test scores.",
                 "**Currency and licence:** the currency is assumed to be rupees, and the dataset's licence was not verified."]),
       P("21. Future Improvements", "h1"),
       *bullets(["Parse `torque` into Nm and test it; extract the car **model** (not only the brand) as a feature.",
                 "Use newer data and a time-aware split to measure performance on future years.",
                 "Group near-identical listings together during cross-validation to reduce leakage.",
                 "Run a wider or Bayesian hyperparameter search; try quantile models for a principled price range.",
                 "Deploy the app (for example on Streamlit Community Cloud) and monitor incoming inputs against the training ranges."]),
       P("22. Major Findings", "h1"),
       *bullets([f"After removing {n_dup:,} duplicates the dataset has {N:,} cars; the price is very skewed (skewness {price.skew():.2f}) and about 3% of rows lack mileage, engine, power and seats together.",
                 f"Car age is by far the most useful feature (error rises {small_ab['car_age']:.4f} without it), then maximum power and brand; `seats` and `seller_type` add nothing measurable.",
                 f"Tuning improved every tree-based model, but not SVR. {nice(selected)} and {nice(runner_up)} are tied ({gap:.4f} apart in cross-validation, noise {sel_noise:.4f}).",
                 f"Test result of the selected model: median error {pct(sel['MedAPE_%'])}, MAE {sel['MAE']:,.0f}, RMSLE {f4(sel['RMSLE'])}.",
                 "Errors are largest for cheap old cars (over-priced) and for luxury cars (under-priced).",
                 "The saved Joblib and Pickle models reload to identical predictions, and the Streamlit app works on top of them."])]

# ------------------------------------------------------------------ 23 appendix
mod_src = (ROOT / "src" / "modeling.py").read_text(encoding="utf-8")
pipe_src = mod_src[mod_src.index("def build_preprocessor"): mod_src.index("def regression_metrics")]
st += [P("23. Appendix - Key Code", "h1"),
       P("The code below is copied from the project files (text, not screenshots). The notebooks in the repository contain all cells with their saved outputs."),
       *code_listing((ROOT / "src" / "car_features.py").read_text(encoding="utf-8"), "A.1 Feature-engineering transformer (src/car_features.py)", "Listing A.1: parsing and feature engineering, shared by training and the app"),
       *code_listing(pipe_src, "A.2 Preprocessing and pipeline (src/modeling.py)", "Listing A.2: the common pipeline used for every model"),
       *code_listing(src(C3, 3), "A.3 Hyperparameter tuning (notebook 03)", "Listing A.3: RandomizedSearchCV for every model"),
       *code_listing(src(C3, 12), "A.4 Saving the model (notebook 03)", "Listing A.4: metadata, Joblib and Pickle")]

# ------------------------------------------------------------------ 24
st += [P("24. GitHub Repository Link", "h1"), P(f"{GITHUB_URL} (folder: `{GITHUB_FOLDER}`)"),
       P("The task folder contains the three executed notebooks, `src/` (reusable code), the data, the saved models in `models/`, the figures and result tables in `reports/`, the tests, `app.py`, "
         "this report and its builder script in `report/`, `requirements.txt` and the README.")]


# ------------------------------------------------------------------ build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "Devixo Solutions - AI/ML Internship Program - Task 03")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                        title="Task 03 - End-to-End Machine Learning Project (Used-Car Price Prediction)", author=STUDENT_NAME)
doc.build(st, onFirstPage=footer, onLaterPages=footer)
BA_PNG.unlink(missing_ok=True)

print(f"Report written: {OUT}")
if WARN:
    print(f"\nWARNING: {len(WARN)} statement(s) in the written text may no longer match the notebook results. Re-read these parts:")
    for w in WARN:
        print("  -", w)
else:
    print("All consistency checks between the written text and the notebook results passed.")
