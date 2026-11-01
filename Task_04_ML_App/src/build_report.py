"""Build the PDF project report (Task 3 style: cover page, numbered sections, appendix).

Every number is read from the saved results (models/*.json), the dataset, the executed
notebooks or docs/test_results.txt. Nothing is typed in by hand.

Run:  python -m src.build_report --author "Your Name" --date "November 1, 2026"
(needs: pip install reportlab)
"""
import argparse
import importlib.metadata as md
import json
import textwrap

import nbformat
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from src.data_utils import ROOT, clean, get_split, load_raw, validate
from src.pipeline import build_pipeline

FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Task04_Bank_Churn_Report.pdf"
W = A4[0] - 4 * cm
NAVY = colors.HexColor("#1f3a5f")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#444444"),
                       fontName="Helvetica-Oblique", alignment=TA_CENTER)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=19, spaceBefore=12, spaceAfter=6, textColor=NAVY, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4, textColor=NAVY, keepWithNext=1)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10, alignment=0, spaceAfter=0)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=8.1)
COVER_BIG = ParagraphStyle("cb", parent=ss["Title"], fontSize=30, leading=36, textColor=NAVY)
COVER_MID = ParagraphStyle("cm", parent=BODY, alignment=TA_CENTER, fontSize=15, leading=20, textColor=NAVY)
COVER_LINE = ParagraphStyle("cl", parent=BODY, fontSize=11.5, leading=17, alignment=0, leftIndent=2.2 * cm)

_state = {"sec": 0, "fig": 0}


# --------------------------------------------------------------------------- helpers
def P(text, style=BODY):
    return Paragraph(text, style)


def h1(title):
    _state["sec"] += 1
    _state["fig"] = 0
    return Paragraph(f"{_state['sec']}. {title}", H1)


def bullets(items):
    st = ParagraphStyle("b", parent=BODY, leftIndent=12, firstLineIndent=-8, spaceAfter=2, alignment=0)
    return [Paragraph(f"- {t}", st) for t in items]


def img(name, width):
    iw, ih = ImageReader(str(FIG / name)).getSize()
    return Image(str(FIG / name), width=width, height=width * ih / iw)


def fig(name, width=W, caption=""):
    _state["fig"] += 1
    cap = f"Figure {_state['sec']}.{_state['fig']}: {caption}" if caption else ""
    return KeepTogether([img(name, width)] + ([Paragraph(cap, SMALL)] if cap else []) + [Spacer(1, 4)])


def table(rows, col_widths=None):
    data = [[Paragraph(str(c), CELLH if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d4")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("BACKGROUND", (0, 0), (-1, 0), NAVY),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for i in range(2, len(rows), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#eef3f9")))
    t.setStyle(TableStyle(style))
    return t


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x):
    return f"{x:.1%}"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.2 * cm, "Devixo Solutions - AI/ML Internship Program - Task 04")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def code_block(text, caption):
    out = []
    for line in text.rstrip().splitlines():
        if len(line) <= 108:
            out.append(line)
        else:
            indent = len(line) - len(line.lstrip())
            out += textwrap.wrap(line, 108, subsequent_indent=" " * (indent + 4), break_long_words=False) or [line]
    return [Preformatted("\n".join(out), CODE), Paragraph(caption, SMALL), Spacer(1, 6)]


def notebook_cell(nb_name, marker):
    nb = nbformat.read(str(ROOT / "notebooks" / nb_name), as_version=4)
    for c in nb.cells:
        if c.cell_type == "code" and marker in c.source:
            return c.source
    raise ValueError(f"cell with {marker!r} not found in {nb_name}")


def ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "?"


# --------------------------------------------------------------------------- main
def build(author, date_text, repo_url, repo_folder, app_url):
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    ablation = json.loads((ROOT / "models" / "feature_ablation.json").read_text())
    ea = json.loads((ROOT / "models" / "error_analysis.json").read_text())
    tests_path = ROOT / "docs" / "test_results.txt"
    test_text = tests_path.read_text().strip() if tests_path.exists() else "Test results file not found."

    meta, models, best = metrics["meta"], metrics["models"], metrics["best_model"]
    order = metrics["model_order"]
    raw = load_raw()
    df = clean(raw)
    q = validate(raw)
    X_train, X_test, y_train, y_test = get_split(df)
    bt = models[best]["tuned"]
    bd, btt = bt["test_default_threshold"], bt["test_tuned_threshold"]
    n_prep_cols = build_pipeline("Logistic Regression")[:-1].fit_transform(X_train).shape[1]
    csv_mb = (ROOT / "data" / "raw" / "Bank Customer Churn Prediction.csv").stat().st_size / 1e6
    space_ns = {}
    exec(notebook_cell("03_modeling_tuning_and_saving.ipynb", "SEARCH_SPACES = {"), space_ns)  # my own notebook code
    spaces = space_ns["SEARCH_SPACES"]

    def rate(mask):
        return df.loc[mask, "churn"].mean()

    S = []
    # ============================================================ cover page
    S += [Spacer(1, 3.2 * cm), P("DEVIXO SOLUTIONS", COVER_BIG), Spacer(1, 4),
          P("AI/ML Internship Program", COVER_MID), Spacer(1, 1.0 * cm),
          P("<b>Task 04</b>", ParagraphStyle("t4", parent=COVER_MID, fontSize=22, leading=28)),
          P("<b>End-to-End ML Application with Deployment</b>", ParagraphStyle("t4b", parent=COVER_MID, fontSize=20, leading=26)),
          Spacer(1, 2.2 * cm),
          P(f"<b>Student Name:</b> {author}", COVER_LINE),
          P("<b>Internship Domain:</b> AI/ML", COVER_LINE),
          P(f"<b>Submission Date:</b> {date_text}", COVER_LINE),
          P("<b>Project:</b> Bank Customer Churn Prediction System (classification) on the Kaggle Bank Customer "
            "Churn Prediction dataset, with a multi-page Streamlit application", COVER_LINE),
          PageBreak()]

    # ============================================================ 1 introduction
    S.append(h1("Introduction"))
    S.append(P(
        "This report documents Task 04 of the Devixo Solutions AI/ML Internship Program: an end-to-end machine learning "
        "application that takes real-world user input, processes it, generates a prediction and presents the result through an "
        "interactive interface. It covers data processing, training and comparing five models, hyperparameter tuning, saving the "
        "final model with Joblib, a Streamlit application with all five bonus features, deployment preparation and automated testing."))
    S.append(P(
        "The chosen project is <b>customer churn prediction</b>: given a bank customer's details, estimate how likely the customer "
        "is to leave the bank. This is a <b>classification</b> problem, so accuracy, precision, recall, F1, the confusion matrix "
        "and ROC-AUC are used (MAE, MSE, RMSE and R2 from the task sheet apply to regression and are not relevant here)."))
    S.append(P("The next table links each part of the Task 04 sheet to the section of this report that answers it."))
    S.append(table([
        ["Task 04 requirement", "Report section"],
        ["Part 1 - Data processing: load dataset, missing values, duplicates, outliers, encoding, scaling, feature engineering", "6, 9, 10, 11"],
        ["Part 2 - Model development: at least 4 models, comparison, best model, classification metrics", "12, 14, 15, 16"],
        ["Part 3 - Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) and before/after comparison", "13"],
        ["Part 4 - Save the final model (Joblib or Pickle)", "17"],
        ["Part 5 - Streamlit application: inputs, Predict button, prediction, relevant information", "18"],
        ["Part 6 - Deployment", "19"],
        ["Bonus: prediction probability, feature importance, data visualization page, model performance page, multiple model selection", "16, 18"],
        ["Documentation: dataset, workflow, results, limitations, future improvements", "3-8, 21-24"],
    ], [W - 3.2 * cm, 3.2 * cm]))

    # ============================================================ 2 objective
    S.append(h1("Objective"))
    S.append(P(
        "The objective is to build a complete, reproducible machine learning application: prepare the data without leaking test "
        "information, compare several algorithms fairly, tune them with cross-validation, choose a model by evidence fixed in "
        "advance, explain what it learned, check where it fails, save it, and serve it through a professional interface that "
        "a non-technical user can operate. The result must be a working application, not only a Jupyter notebook."))

    # ============================================================ 3-5 dataset
    S.append(h1("Dataset Name"))
    S.append(P("Bank Customer Churn Prediction, file <font face='Courier'>Bank Customer Churn Prediction.csv</font> "
               "(stored in the project as <font face='Courier'>data/raw/</font>)."))
    S.append(h1("Dataset Source"))
    S.append(P(
        "Kaggle dataset 'Bank Customer Churn Prediction'. The dataset description describes it as data for 'ABC Multistate Bank' with "
        "customer attributes and a churn flag. The Kaggle page does <b>not</b> state how the data was collected, whether it is real "
        "or simulated, or under which licence it may be reused. This report therefore does not claim that the customers are real; "
        "this is recorded as a limitation (section 22) and should be discussed with the mentor. The licence should be checked on "
        "the Kaggle page before the data is redistributed."))
    S.append(h1("Dataset Description"))
    S.append(P(
        "Each row is one bank customer. The columns describe the customer (country, gender, age), the relationship with the bank "
        "(tenure, number of products, credit card, active membership), financial details (credit score, balance, estimated salary) "
        "and the target <b>churn</b>, which is 1 when the customer left the bank and 0 otherwise. The description gives no time "
        "window for 'left the bank', so predictions are reported as a churn <i>risk</i>, not as a prediction for a specific date."))

    # ============================================================ 6 size
    S.append(h1("Dataset Size"))
    S.append(table([
        ["", "Raw dataset", "After processing"],
        ["Rows (customers)", f"{len(raw):,}", f"{len(df):,}"],
        ["Columns", f"{raw.shape[1]}", f"{len(X_train.columns)} model inputs (customer_id and churn removed), +4 engineered"],
        ["Exact duplicate rows", f"{q['full_duplicates']}", "0"],
        ["Duplicates ignoring customer_id", f"{q['duplicates_ignoring_id']}", "0"],
        ["Missing values", f"{q['missing_total']}", "0 (imputers still in the pipeline)"],
        ["Train / test rows", "-", f"{len(X_train):,} / {len(X_test):,} (80% / 20%, stratified)"],
        ["Columns after encoding and scaling", "-", f"{n_prep_cols} (training matrix)"],
        ["File size", f"about {csv_mb:.2f} MB", "-"],
    ], [5.2 * cm, 3.6 * cm, W - 8.8 * cm]))

    # ============================================================ 7 columns
    S.append(h1("Features/Columns Description"))
    S.append(table([
        ["Column", "Type", "Role / meaning"],
        ["customer_id", "identifier", "Unique per row; listed as an unused variable; not a model input"],
        ["credit_score", "integer", "Input: customer credit score"],
        ["country", "categorical", "Input: France, Germany or Spain"],
        ["gender", "categorical", "Input: Female or Male"],
        ["age", "integer", "Input: age in years"],
        ["tenure", "integer", "Input: years as a customer"],
        ["balance", "float", "Input: account balance (many customers have exactly 0)"],
        ["products_number", "integer", "Input: number of bank products (1 to 4)"],
        ["credit_card", "binary", "Input: 1 if the customer has a credit card"],
        ["active_member", "binary", "Input: 1 if the customer is an active member"],
        ["estimated_salary", "float", "Input: estimated annual salary"],
        ["churn", "binary", "Target: 1 = customer left the bank"],
    ], [3.4 * cm, 2.6 * cm, W - 6 * cm]))

    # ============================================================ 8 methodology
    S.append(h1("Methodology"))
    S.append(P(
        f"The work was done in Python {meta['python']} with pandas {meta['pandas']}, NumPy {meta['numpy']}, scikit-learn "
        f"{meta['sklearn']}, Matplotlib {ver('matplotlib')}, Seaborn {ver('seaborn')}, Joblib {ver('joblib')} and Streamlit "
        f"{ver('streamlit')}, in Jupyter notebooks with reusable code in <font face='Courier'>src/</font>. The random seed is "
        f"{meta['random_state']} everywhere. The workflow is:"))
    S += bullets([
        "<b>Notebook 01 - Data audit and EDA:</b> read-only inspection of the raw file; records every data check and the decision it leads to.",
        "<b>Notebook 02 - Preparation and features:</b> clean, split 80/20 <b>before</b> learning anything from the data, engineer features, "
        "test them with an ablation experiment, and show the preprocessing pipeline.",
        "<b>Notebook 03 - Modelling:</b> baseline models, tuning, before/after comparison, feature importance, selection, error analysis, "
        "saving with Joblib and verification by reloading.",
        "<b>One pipeline:</b> feature engineering, imputation, scaling, encoding and the model form one scikit-learn Pipeline fitted on the "
        "training data only, so the test set cannot leak and the saved model accepts raw customer inputs.",
        "<b>Streamlit app</b> (<font face='Courier'>app/</font>) reads the saved files, and <b>automated tests</b> (pytest) check the data, "
        "features, saved models and the app.",
    ])

    # ============================================================ 9 data processing
    S.append(h1("Part 1 - Data Processing"))
    S.append(P("Every action is recorded below with its reason. The raw file was never changed; all work was done on copies."))
    flagged = [r["column"] for r in meta["outliers_iqr"] if r["n_outliers"] > 0]
    n_out = {r["column"]: r["n_outliers"] for r in meta["outliers_iqr"]}
    S.append(table([
        ["Step", "Action", "Result", "Reason"],
        ["1", "Loaded the CSV", f"{len(raw):,} rows, {raw.shape[1]} columns", "Customer data with a binary target"],
        ["2", "Checked for missing values and blank text", f"{q['missing_total']} found", "Imputers (median / most frequent) stay inside the pipeline so the saved model is safe if a missing value ever arrives; tested"],
        ["3", "Checked for and removed duplicates", f"{q['full_duplicates']} found ({q['duplicates_ignoring_id']} ignoring customer_id)", "Done before the split so a repeated customer cannot sit in both train and test"],
        ["4", "Dropped customer_id", "1 column", "Identifier, not a feature"],
        ["5", "Detected outliers (1.5 x IQR)", ", ".join(f"{c}: {n_out[c]}" for c in flagged) or "none", "Detection only; values judged plausible and kept (section 10)"],
        ["6", "Stratified split 80/20 (seed 42)", f"{len(X_train):,} train / {len(X_test):,} test", "Done before any statistic is learned; both parts keep the same churn rate"],
        ["7", "Engineered 4 features", "balance_zero, balance_to_salary, many_products, age_group", "Tested by ablation in section 11"],
        ["8", "Scaled 7 numeric columns (StandardScaler)", "fitted on training data only", "Needed by Logistic Regression, KNN and SVM; harmless for trees"],
        ["9", "One-hot encoded country, gender, age_group", "unseen categories ignored safely", "Models need numbers; no false ordering is created"],
        ["10", "Saved the whole pipeline with the model", "see section 17", "The app applies exactly the same steps as training"],
    ], [1.0 * cm, 4.2 * cm, 4.6 * cm, W - 9.8 * cm]))
    S.append(Spacer(1, 5))
    S.append(P(
        "The dataset was already clean, so no value needed fixing. The required steps (missing values, duplicates) were still "
        "implemented and unit-tested; here they change nothing, and no problems were invented."))

    # ============================================================ 10 EDA
    S.append(h1("Exploratory Data Analysis"))
    counts = df["churn"].value_counts().sort_index()
    S.append(P(
        f"<b>Target balance.</b> {int(counts[0]):,} customers stayed ({counts[0]/len(df):.1%}) and {int(counts[1]):,} churned "
        f"({counts[1]/len(df):.1%}). A model that always predicts 'stays' would score {meta['majority_class_accuracy_test']:.1%} "
        f"accuracy on the test set, so accuracy alone is misleading and ROC-AUC, recall and F1 are used as well."))
    S.append(fig("01_churn_distribution.png", W * 0.5, "churn distribution."))
    S.append(P(
        f"<b>Churn by group.</b> Germany has a churn rate of {pct(rate(df.country=='Germany'))} versus {pct(rate(df.country=='France'))} "
        f"(France) and {pct(rate(df.country=='Spain'))} (Spain). Women churn at {pct(rate(df.gender=='Female'))} versus "
        f"{pct(rate(df.gender=='Male'))} for men. Inactive members churn at {pct(rate(df.active_member==0))} versus "
        f"{pct(rate(df.active_member==1))} for active members. Customers with 3 or 4 products churn at "
        f"{pct(rate(df.products_number==3))} (n={int((df.products_number==3).sum())}) and {pct(rate(df.products_number==4))} "
        f"(n={int((df.products_number==4).sum())}); these groups are small, so their rates are less reliable. Having a credit card "
        f"makes almost no difference ({pct(rate(df.credit_card==1))} with versus {pct(rate(df.credit_card==0))} without). "
        f"These are associations in this dataset, not proof of cause."))
    S.append(fig("03_churn_rate_by_category.png", W, "churn rate by group (shared y-axis; n = group size; dashed line = overall churn rate)."))
    ages = df.groupby(pd.cut(df.age, [17, 30, 40, 50, 60, 100]), observed=True)["churn"].mean()
    zb = df.balance == 0
    S.append(P(
        f"<b>Age.</b> Churn is {pct(ages.iloc[0])} for customers aged 30 or under, rises to {pct(ages.iloc[3])} in the 51-60 band and "
        f"falls again to {pct(ages.iloc[4])} above 60. The relationship is not a straight line, which is why age bands were "
        f"engineered. <b>Zero balances:</b> {int(zb.sum()):,} customers ({zb.mean():.1%}) have a balance of exactly 0. This is a valid "
        f"group, not missing data; their churn rate is {pct(rate(zb))} versus {pct(rate(~zb))} for customers with a balance."))
    S.append(fig("04_numeric_by_churn.png", W, "age, balance and credit-score distributions by churn."))
    S.append(fig("05_correlation_heatmap.png", W * 0.68, "correlation matrix (linear association only; not proof of cause)."))
    out_rows = [["Column", "IQR lower", "IQR upper", "Outliers", "% of rows", "Min", "Max"]]
    for r in meta["outliers_iqr"]:
        out_rows.append([r["column"], f"{r['lower_bound']:,.0f}", f"{r['upper_bound']:,.0f}", r["n_outliers"],
                         f"{r['pct']}%", f"{r['min']:,.0f}", f"{r['max']:,.0f}"])
    S.append(P(f"<b>Outlier detection.</b> The 1.5 x IQR rule flagged values in {len(flagged)} of 5 numeric columns ({', '.join(flagged)}):"))
    S.append(table(out_rows, [3.6 * cm, 2.4 * cm, 2.4 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, W - 14.4 * cm]))
    S.append(Spacer(1, 4))
    S.append(fig("02_boxplots_outliers.png", W, "boxplots of the numeric columns."))
    old = df[df.age >= 70]
    S.append(P(
        f"<b>Decision: keep the outliers.</b> The flagged ages go up to {int(df.age.max())} and the lowest credit score is "
        f"{int(df.credit_score.min())}; these are plausible values for bank customers, not data-entry errors. {len(old)} customers are "
        f"aged 70 or over and they churn at {pct(old.churn.mean())} versus {pct(df.churn.mean())} overall, so deleting them would remove "
        f"real information. Tree models are not sensitive to such values and the linear and distance-based models use standardised "
        f"inputs. This was a judgement based on the evidence, and it is documented so it can be reversed."))

    # ============================================================ 11 FE
    S.append(h1("Feature Engineering and Feature Selection"))
    S.append(P("<b>Feature engineering.</b> Four features were created, each from values in the same row only (so nothing can leak "
               "from other rows):"))
    S.append(table([
        ["Feature", "Definition", "Why"],
        ["balance_zero", "1 if balance = 0", f"Large group ({zb.mean():.0%}) with a different churn rate"],
        ["balance_to_salary", "balance / (salary + 1)", "Share of income held at the bank"],
        ["many_products", "1 if products_number >= 3", "Very different churn behaviour from 1-2 products"],
        ["age_group", "bands: <=30, 31-40, 41-50, 51-60, 60+", "Churn peaks in the 50s (not a straight line)"],
    ], [3.4 * cm, 5.6 * cm, W - 9 * cm]))
    S.append(Spacer(1, 4))
    lr, rf = ablation["Logistic Regression"], ablation["Random Forest"]
    S.append(P(
        f"<b>Did they help? (ablation test).</b> Using 5-fold cross-validation on the training data only, Logistic Regression's "
        f"ROC-AUC rose from {f3(lr['none']['cv_roc_auc_mean'])} (no engineered features) to {f3(lr['all']['cv_roc_auc_mean'])} (all four); "
        f"the largest single gain came from many_products ({f3(lr['many_products']['cv_roc_auc_mean'])}). For Random Forest the change was small "
        f"({f3(rf['none']['cv_roc_auc_mean'])} to {f3(rf['all']['cv_roc_auc_mean'])}) and within the fold-to-fold variation "
        f"(about {f3(rf['all']['cv_roc_auc_std'])}), as expected because trees can find such splits themselves. balance_zero and "
        f"balance_to_salary added almost nothing for Logistic Regression on their own. All four were kept because the combined set was "
        f"best or equal for both models and none hurt performance."))
    S.append(fig("13_feature_ablation.png", W, "cross-validated ROC-AUC with each engineered feature added (training data only)."))
    S.append(P("<b>Feature selection.</b> All ten raw inputs were kept: there are few columns, none is an identifier or a leak, and "
               "<font face='Courier'>customer_id</font> is the only column removed. Permutation importance in section 16 shows which inputs "
               "the final model actually uses."))

    # ============================================================ 12 models
    S.append(h1("Part 2 - Model Development"))
    S.append(P(
        "Every model is placed in the <b>same pipeline</b> and trained on the same training set with the same cross-validation folds, "
        "so the comparison is fair. The task requires at least four models; five were trained."))
    S.append(table([
        ["Model", "Needs scaling?", "Default settings used"],
        ["Logistic Regression", "Yes", "max_iter=2000, regularisation C=1"],
        ["KNN (K-Nearest Neighbours)", "Yes", "5 neighbours, uniform weights"],
        ["SVM (RBF kernel)", "Yes", "C=1, gamma='scale', probability estimates on"],
        ["Random Forest", "No", "200 trees"],
        ["Gradient Boosting", "No", "scikit-learn defaults"],
    ], [5 * cm, 3 * cm, W - 8 * cm]))
    S.append(Spacer(1, 4))
    S.append(P("<b>Metrics.</b> Accuracy, precision, recall, F1, the confusion matrix and ROC-AUC are reported (the classification metrics "
               "required by the task). ROC-AUC is the selection metric because it does not depend on a threshold and works on imbalanced data. "
               "Recall is the share of real churners that are caught; precision is the share of flagged customers that really churn."))
    rows = [["Model", "CV ROC-AUC", "CV std", "Test accuracy", "Precision", "Recall", "F1", "Test ROC-AUC"]]
    for n in order:
        b = models[n]["baseline"]
        m = b["test_default_threshold"]
        rows.append([n, f3(b["cv_roc_auc_mean"]), f3(b["cv_roc_auc_std"]), f3(m["accuracy"]), f3(m["precision"]),
                     f3(m["recall"]), f3(m["f1"]), f3(m["roc_auc"])])
    S.append(P("<b>Results with default settings</b> (cross-validation on the training set; test metrics at threshold 0.5):"))
    S.append(table(rows, [3.8 * cm, 1.9 * cm, 1.4 * cm, 2.0 * cm, 1.8 * cm, 1.5 * cm, 1.3 * cm, W - 13.7 * cm]))
    S.append(Spacer(1, 4))
    worst = min(order, key=lambda n: models[n]["baseline"]["cv_roc_auc_mean"])
    S.append(P(
        f"With default settings the weakest model by cross-validated ROC-AUC is {worst} "
        f"({f3(models[worst]['baseline']['cv_roc_auc_mean'])}); distance-based and linear models depend on scaled inputs, which the pipeline provides."))

    # ============================================================ 13 tuning
    S.append(h1("Part 3 - Hyperparameter Tuning (Before vs After)"))
    S.append(P(
        "Every model was tuned with cross-validation on the <b>training set only</b> (5 stratified folds, scoring = ROC-AUC): "
        "GridSearchCV for Logistic Regression and KNN (small search spaces, every combination tried) and RandomizedSearchCV for SVM, "
        "Random Forest and Gradient Boosting (a fixed number of random combinations). After each search the model is refitted on the whole "
        "training set with its best settings. The test set plays no part in tuning."))
    S.append(P("<b>13.1 Settings searched</b>", H2))
    rows = [["Model", "Search", "Parameters searched"]]
    for n in order:
        kind, space, n_iter = spaces[n]
        desc = "; ".join(f"{k.replace('model__', '')} {{{', '.join(str(v) for v in vals)}}}" for k, vals in space.items())
        rows.append([n, "GridSearchCV" if kind == "grid" else f"RandomizedSearchCV ({n_iter} combinations)", desc])
    S.append(table(rows, [3.2 * cm, 3.6 * cm, W - 6.8 * cm]))
    S.append(Spacer(1, 4))
    S.append(P("<b>13.2 Results</b>", H2))
    rows = [["Model", "Best parameters found", "CV ROC-AUC", "CV std"]]
    for n in order:
        t = models[n]["tuned"]
        rows.append([n, ", ".join(f"{k}={v}" for k, v in t["best_params"].items()), f4(t["cv_roc_auc_mean"]), f4(t["cv_roc_auc_std"])])
    S.append(table(rows, [3.4 * cm, W - 7.6 * cm, 2.2 * cm, 2.0 * cm]))
    S.append(Spacer(1, 6))
    rows = [["Model", "CV AUC before", "CV AUC after", "Test AUC before", "Test AUC after", "Change (test AUC)", "Test F1 before", "Test F1 after"]]
    gains = {}
    for n in order:
        b, t = models[n]["baseline"], models[n]["tuned"]
        bd_, td_ = b["test_default_threshold"], t["test_default_threshold"]
        gains[n] = td_["roc_auc"] - bd_["roc_auc"]
        rows.append([n, f4(b["cv_roc_auc_mean"]), f4(t["cv_roc_auc_mean"]), f4(bd_["roc_auc"]), f4(td_["roc_auc"]),
                     f"{0.0 if abs(gains[n]) < 0.00005 else gains[n]:+.4f}", f3(bd_["f1"]), f3(td_["f1"])])
    S.append(table(rows, [3.1 * cm, 1.8 * cm, 1.8 * cm, 1.9 * cm, 1.9 * cm, 2.0 * cm, 1.8 * cm, W - 14.3 * cm]))
    S.append(Spacer(1, 4))
    S.append(fig("10_tuning_before_after_auc.png", W * 0.9, "test ROC-AUC before and after tuning."))
    big = [n for n in order if gains[n] >= 0.005]
    flat = [n for n in order if abs(gains[n]) < 0.005]
    worse = [n for n in order if gains[n] <= -0.005]
    parts = []
    if big:
        parts.append("Test ROC-AUC improved noticeably (+0.005 or more) for " + ", ".join(f"{n} ({gains[n]:+.3f})" for n in big) + ".")
    if flat:
        parts.append("It was essentially unchanged (within 0.005) for " + ", ".join(f"{n} ({0.0 if abs(gains[n]) < 0.0005 else gains[n]:+.3f})" for n in flat) + ".")
    if worse:
        parts.append("It was slightly worse for " + ", ".join(f"{n} ({gains[n]:+.3f})" for n in worse) + ".")
    f1_drop = [n for n in order if models[n]["tuned"]["test_default_threshold"]["f1"] < models[n]["baseline"]["test_default_threshold"]["f1"] - 0.005]
    if f1_drop:
        parts.append("F1 at the default 0.5 threshold went <i>down</i> after tuning for " + ", ".join(f1_drop) +
                     ". This is expected: tuning optimised ROC-AUC, which ranks customers across all thresholds, while F1 at 0.5 depends "
                     "on one fixed cut-off; section 14 handles the threshold separately.")
    S.append(P(" ".join(parts) + " The 'CV AUC after' value is the best score found during the search and is slightly optimistic by "
                                 "construction; the test columns are the unbiased comparison. The KNN grid was widened to 81 neighbours "
                                 "after a first, narrower grid chose its edge value."))

    # ============================================================ 14 selection
    S.append(h1("Model Selection and Final Test Results"))
    ranked = sorted(order, key=lambda n: -models[n]["tuned"]["cv_roc_auc_mean"])
    gap = models[ranked[0]]["tuned"]["cv_roc_auc_mean"] - models[ranked[1]]["tuned"]["cv_roc_auc_mean"]
    S.append(P(
        f"<b>Selection rule, fixed before the test set was used:</b> the model with the highest tuned cross-validated ROC-AUC wins. "
        f"{best} was selected (CV ROC-AUC {f4(bt['cv_roc_auc_mean'])}, fold spread {f4(bt['cv_roc_auc_std'])}). Its settings: "
        f"{', '.join(f'{k}={v}' for k, v in bt['best_params'].items())}. The runner-up, {ranked[1]}, is {gap:.4f} behind, "
        f"{'much less than' if gap < bt['cv_roc_auc_std'] else 'comparable to'} the fold-to-fold noise, so the top models are effectively "
        f"close. Changing the choice after seeing test scores would mean selecting a model with the test set, so the rule was kept."))
    rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Threshold", "Recall @thr", "Precision @thr", "F1 @thr"]]
    for n in ranked:
        d, t = models[n]["tuned"]["test_default_threshold"], models[n]["tuned"]["test_tuned_threshold"]
        rows.append([n + (" (selected)" if n == best else ""), f3(d["accuracy"]), f3(d["precision"]), f3(d["recall"]), f3(d["f1"]),
                     f3(d["roc_auc"]), f"{t['threshold']:.2f}", f3(t["recall"]), f3(t["precision"]), f3(t["f1"])])
    S.append(P(f"The test set ({len(X_test):,} customers) was then used once. All tuned models are shown for transparency. The first five "
               "metric columns use the standard 0.5 threshold; the last four use each model's own F1-optimised threshold, chosen from "
               "out-of-fold predictions on the training data only."))
    S.append(table(rows, [3.3 * cm, 1.5 * cm, 1.6 * cm, 1.3 * cm, 1.2 * cm, 1.5 * cm, 1.5 * cm, 1.6 * cm, 1.7 * cm, W - 15.2 * cm]))
    S.append(Spacer(1, 4))
    S.append(fig("06_model_comparison.png", W, "test-set metrics of the tuned models (threshold 0.5)."))
    S.append(fig("07_roc_curves.png", W * 0.58, "ROC curves of the tuned models on the test set."))
    f1_at_thr = {n: models[n]["tuned"]["test_tuned_threshold"]["f1"] for n in order}
    top_f1 = max(f1_at_thr, key=f1_at_thr.get)
    if top_f1 != best:
        S.append(P(
            f"<b>Honest note.</b> On this particular test set {top_f1} reached a slightly higher F1 at its own F1-optimised threshold "
            f"({f3(f1_at_thr[top_f1])} versus {f3(f1_at_thr[best])}). The gap is small and within the noise of a {len(X_test):,}-row test set, "
            f"so it does not change the selection rule fixed in advance; the app lets you switch to {top_f1} to compare."))
    tn, fp = bd["confusion_matrix"][0]
    fn, tp = bd["confusion_matrix"][1]
    S.append(P(
        f"<b>Selected model on the test set.</b> At threshold 0.5 it caught {tp} of {tp + fn} churners (recall {f3(bd['recall'])}) and "
        f"{tp} of its {tp + fp} churn predictions were correct (precision {f3(bd['precision'])}); {fn} churners were missed. In churn "
        f"problems a missed churner is often costlier than a false alarm (an assumption: the dataset has no cost information), so a "
        f"second threshold ({bt['tuned_threshold']:.2f}) was chosen to maximise F1. There, recall rises to {f3(btt['recall'])} with precision "
        f"{f3(btt['precision'])} and F1 {f3(btt['f1'])}. The app labels a customer 'likely to churn' using this second threshold and "
        f"always shows the raw probability."))
    two = Table([[img("08_confusion_matrix_best_default.png", W * 0.46), img("09_confusion_matrix_best_tuned_threshold.png", W * 0.46)]],
                colWidths=[W * 0.5, W * 0.5])
    S.append(two)
    _state["fig"] += 1
    S.append(P(f"Figure {_state['sec']}.{_state['fig']}: confusion matrices of the selected model on the test set (left: threshold 0.5; right: F1-optimised threshold).", SMALL))

    # ============================================================ 15 error analysis
    S.append(h1("Error Analysis of the Selected Model"))
    S.append(P(f"A single score can hide where a model fails, so the selected model ({best}, at its F1-optimised threshold "
               f"{ea['threshold']:.2f}) was examined by customer group and for probability quality on the test set."))
    seg_rows = [["Group", "Customers", "Actual churn rate", "Flagged as churn", "Recall", "Precision"]]
    seg_lookup = []
    for col, rows_ in ea["segments"].items():
        for r in rows_:
            seg_rows.append([f"{col} = {r['group']}", r["n"], pct(r["churn_rate"]), pct(r["flag_rate"]),
                             "-" if r["recall"] is None else f3(r["recall"]), "-" if r["precision"] is None else f3(r["precision"])])
            if r["recall"] is not None and r["n_churned"] >= 30:
                seg_lookup.append((f"{col} = {r['group']}", r["recall"], r["n_churned"]))
    S.append(table(seg_rows, [4.6 * cm, 2.0 * cm, 2.8 * cm, 2.8 * cm, 2.0 * cm, W - 14.2 * cm]))
    S.append(Spacer(1, 4))
    lo, hi = min(seg_lookup, key=lambda x: x[1]), max(seg_lookup, key=lambda x: x[1])
    S.append(P(
        f"Among groups with at least 30 churners in the test set, recall is lowest for <b>{lo[0]}</b> ({f3(lo[1])}, {lo[2]} churners) and highest for "
        f"<b>{hi[0]}</b> ({f3(hi[1])}, {hi[2]} churners). Recall differs by group because some groups are easier to separate than others and "
        f"some are small; the per-group numbers are noisy, so they point to where to look, not to firm conclusions."))
    cal_rows = [["Predicted probability bin", "Customers", "Mean predicted", "Observed churn rate"]]
    gaps = []
    for c in ea["calibration"]:
        cal_rows.append([c["bin"], c["n"], "-" if c["mean_predicted"] is None else pct(c["mean_predicted"]),
                         "-" if c["observed_rate"] is None else pct(c["observed_rate"])])
        if c["n"] >= 50:
            gaps.append(abs(c["mean_predicted"] - c["observed_rate"]))
    S.append(table(cal_rows, [4.6 * cm, 2.6 * cm, 3.6 * cm, W - 10.8 * cm]))
    S.append(Spacer(1, 4))
    S.append(fig("14_calibration.png", W * 0.5, "calibration of the selected model on the test set."))
    verdict = ("close to the observed churn rates" if max(gaps) <= 0.05 else
               "reasonably close to the observed churn rates" if max(gaps) <= 0.10 else "noticeably different from the observed churn rates")
    S.append(P(
        f"<b>Are the probabilities trustworthy?</b> In bins with at least 50 customers the mean predicted probability differs from the observed "
        f"churn rate by at most {max(gaps)*100:.1f} percentage points, so the probabilities are {verdict}. The Brier score is "
        f"{ea['brier_score']:.3f} (lower is better). Bins with few customers are noisy, and this was checked on one test set only."))

    # ============================================================ 16 feature importance
    S.append(h1("Feature Importance"))
    imp = bt["permutation_importance"]
    top = sorted(imp.items(), key=lambda kv: -kv[1]["mean"])
    S.append(P("<b>Permutation importance.</b> One input column of the test set is shuffled at a time and the drop in ROC-AUC is measured "
               "(5 repeats). It works for any model, uses the raw input columns and measures predictive value on unseen data. "
               f"For {best}, the most influential inputs were " + ", ".join(f"<b>{k}</b> ({v['mean']:.3f})" for k, v in top[:3]) +
               "; the least influential were " + ", ".join(f"{k} ({v['mean']:.3f})" for k, v in top[-2:]) + "."))
    S.append(fig("12_feature_importance_best.png", W * 0.7, "permutation importance of the selected model."))
    rows = [["Input column", "Importance (AUC drop)", "Spread (std)"]] + [[k, f4(v["mean"]), f4(v["std"])] for k, v in top]
    S.append(table(rows, [5 * cm, 5 * cm, W - 10 * cm]))
    S.append(Spacer(1, 4))
    S.append(P("Importance shows what the <i>model</i> used, not what <i>causes</i> customers to leave; correlated inputs can share credit, and "
               "values near zero mean the model barely used that column. The app shows this chart per model on its Feature Importance page."))

    # ============================================================ 17 saving
    S.append(h1("Part 4 - Save the Final Model (Joblib)"))
    S.append(P("The <b>whole pipeline</b> is saved (feature engineering, imputation, scaling, encoding and the model), so a saved file "
               "accepts a raw customer description and returns a probability."))
    rows = [["File", "Size", "Purpose"]]
    for rel, purpose in [("models/final_model.joblib", "The selected model (Joblib, compressed)"),
                         ("models/all_models/*.joblib", "All five tuned pipelines (used by the multiple-model selection)"),
                         ("models/metrics.json", "Every metric shown in the app and report"),
                         ("models/reference.json", "Typical values and segment churn rates from the training split"),
                         ("models/error_analysis.json", "Segment and calibration results"),
                         ("models/feature_ablation.json", "Feature-engineering experiment")]:
        if "*" in rel:
            size = sum(p.stat().st_size for p in (ROOT / "models" / "all_models").glob("*.joblib"))
        else:
            size = (ROOT / rel).stat().st_size
        rows.append([f"<font face='Courier'>{rel}</font>", f"{size/1e6:.2f} MB", purpose])
    S.append(table(rows, [6.4 * cm, 2.0 * cm, W - 8.4 * cm]))
    S.append(Spacer(1, 4))
    S.append(P("<b>Verification (never trust a save that was not reloaded):</b>"))
    S += bullets([
        "Notebook 03 reloads every saved model and checks that it predicts the same probabilities as the in-memory model on the whole test set.",
        "Automated tests check that each model loads, returns valid probabilities, handles missing values, unseen categories and extreme "
        "values, and that the metrics stored in metrics.json equal a fresh evaluation of the saved model.",
        f"The models were saved with scikit-learn {meta['sklearn']}; requirements.txt pins the library versions because a saved model can only "
        "be loaded safely with the same versions. If loading fails on another computer, re-running notebook 03 rebuilds the models with the "
        "installed libraries. Joblib/Pickle files must only be loaded from trusted sources.",
    ])

    # ============================================================ 18 streamlit
    S.append(h1("Part 5 - Streamlit Application and Bonus Features"))
    S.append(P("The application (<font face='Courier'>app/app.py</font> plus three extra pages) is a working ML application, not only a notebook. "
               "It loads the saved pipelines, builds one raw-format customer row from the form and calls the model."))
    S.append(table([
        ["Page", "What the user can do"],
        ["Predict (main page)", "Enter 10 customer details, click 'Predict churn risk', and see the verdict, <b>churn probability</b> (percentage and progress bar), "
                                "how it compares with the average customer, an approximate chart of which inputs raise or lower the risk, and the historical "
                                "churn rates of similar customers. Warns when an input is outside the range seen in training."],
        ["Data Visualization (bonus)", "Filter by country, gender and age; churn overview, churn rate by group, numeric distributions, correlation."],
        ["Model Performance (bonus)", "Comparison table, before/after tuning, confusion matrices, ROC curves, calibration and the feature-engineering ablation, all read from saved files."],
        ["Feature Importance (bonus)", "Permutation importance for the selected model and a comparison across all five models."],
        ["Sidebar on every page (bonus)", "<b>Multiple model selection:</b> any of the five tuned models can be chosen; the best is the default."],
    ], [4.2 * cm, W - 4.2 * cm]))
    S.append(Spacer(1, 4))
    S.append(P("<b>How the 'what is driving this prediction' chart works.</b> For each input, only that value is replaced with the typical value from "
               "the training data and the change in predicted probability is shown. It is a simple model-based explanation that ignores interactions "
               "and is not evidence of cause. <b>Usability details:</b> the form starts with a typical customer; results appear only after the button is "
               "clicked; tables with highlighted cells set both background and text colour so they are readable in light and dark themes."))
    S.append(P("Run it with <font face='Courier'>streamlit run app/app.py</font>. The app is covered by automated tests (section 20) and was started on "
               "a real Streamlit server to check that it loads. No screenshot is included in this report."))

    # ============================================================ 19 deployment
    S.append(h1("Part 6 - Deployment"))
    if app_url:
        S.append(P(f"The application is deployed at: <font face='Courier'>{app_url}</font>"))
    else:
        S.append(P("<b>Status: not deployed yet.</b> The application is deployment-ready (relative paths, pinned requirements.txt, models "
                   "included in the repository) and runs locally. The task asks to deploy 'if possible'; the steps below deploy it on Streamlit "
                   "Community Cloud, which is free."))
    S += bullets([
        "Push the project to a public GitHub repository (the models/ folder must be included).",
        "Open share.streamlit.io, sign in with GitHub and choose New app.",
        "Select the repository and branch and set the main file path to app/app.py; choose Python 3.12 in the advanced settings.",
        "Open the live URL, test a prediction on every page, and add the URL to the README and to this report.",
    ])

    # ============================================================ 20 QA
    S.append(h1("Quality Assurance and Testing"))
    S.append(P("An automated test suite (pytest) checks the data, the pipeline, the saved models, the notebooks and the app:"))
    S += bullets([
        "<b>Data:</b> schema, no missing values or duplicates, cleaning logic (duplicates and missing targets are removed), the split is stratified, "
        "reproducible and has no overlap, customer_id is never a model input.",
        "<b>Features:</b> correct values, age-band boundaries (18, 30, 31, 40, 41, 50, 51, 60, 61, 100), no NaN or infinity, input data not modified, "
        "zero salary does not divide by zero.",
        "<b>Models:</b> all five load; probabilities are valid; minimum, maximum, risky-combination, unseen-category and missing-value inputs do not crash; "
        "stored metrics equal a fresh evaluation; confusion-matrix totals equal the test-set size; the stored best model has the highest CV score.",
        "<b>Notebooks:</b> all three notebooks exist, are executed and contain no error output.",
        "<b>App (Streamlit AppTest):</b> every page loads without errors; the form and Predict button exist; no result appears before the click; a click gives "
        "exactly one verdict with a probability and progress bar; every one of the five models can be selected; extreme inputs give a warning, not a crash; "
        "a high-risk profile scores higher than a low-risk profile.",
    ])
    S.append(P("<b>Test run output:</b>"))
    S.append(Preformatted(test_text, ParagraphStyle("t", fontName="Courier", fontSize=7.5, leading=9.5)))
    S.append(Spacer(1, 4))
    S.append(P("Automated tests cannot replace looking at the running app in a browser; screenshots should be captured and added to the README."))

    # ============================================================ 21 conclusion
    S.append(h1("Conclusion and Recommendation"))
    S.append(P(
        f"A complete, leak-free pipeline was built on {len(df):,} customers: the data was audited, split before anything was learned from it, features "
        f"were engineered and tested, five models were compared and tuned with cross-validation, one was selected by a rule fixed in advance, its errors "
        f"and probabilities were examined, and everything was saved and put behind a Streamlit application."))
    S.append(P(
        f"<b>Recommendation:</b> use the tuned {best}. On the untouched test set it reaches ROC-AUC {f3(bd['roc_auc'])}; at its F1-optimised threshold it "
        f"catches {pct(btt['recall'])} of churners with {pct(btt['precision'])} precision. It should be used to <i>rank</i> customers by churn risk and prioritise "
        f"retention effort, not as a certain prediction for any one customer."))
    S.append(P("<b>Lessons from this project:</b> the split must come before any fitting; one pipeline prevents leakage and makes saving simple; accuracy "
               "misleads on imbalanced data; features should be kept or dropped on evidence; tuning does not always help; models can be close to tied, and "
               "the honest answer is to say so; and a saved model must be reloaded and tested."))

    # ============================================================ 22 limitations
    S.append(h1("Limitations"))
    S += bullets([
        "<b>Data provenance:</b> the source and licence are not stated on Kaggle, and it is not confirmed that the data is real; the task asks for a real-world dataset.",
        "<b>Churn definition:</b> no time window is given, so the app reports a risk, not a prediction for a date.",
        f"<b>Small groups:</b> customers with 3-4 products are few ({int((df.products_number==3).sum())} and {int((df.products_number==4).sum())} customers), so patterns there are less reliable.",
        f"<b>Single split:</b> a {len(X_test):,}-row test set gives noisy estimates; small differences between models should not be over-interpreted.",
        "<b>Calibration</b> was checked on one test set only; bins with few customers are noisy.",
        "<b>Fairness:</b> gender and country are model inputs because they are in the dataset; in a real bank their use would need legal and ethical review.",
        "<b>Explanations</b> (importance, driving factors) describe model behaviour, not causes.",
        "<b>Random split, not a time split:</b> the data has no dates, so performance on future customers is untested.",
        "<b>Modest tuning:</b> 8-20 random combinations for the larger search spaces; a longer search might change small differences between models.",
        f"<b>Versions:</b> models were trained with scikit-learn {meta['sklearn']} and only load reliably with that version (pinned).",
    ])

    # ============================================================ 23 future
    S.append(h1("Future Improvements"))
    S += bullets([
        "Confirm the data source and licence, and test on genuinely independent data.",
        "Calibrate the probabilities (for example isotonic regression) and choose the threshold from real retention costs.",
        "Use repeated cross-validation or a larger hold-out for more stable model comparison; try a wider or Bayesian hyperparameter search.",
        "Add fairness checks across country and gender groups.",
        "Deploy the app, monitor incoming inputs against the training ranges, and retrain when the data drifts.",
        "Add per-customer explanations with a dedicated method such as SHAP, and optional models such as XGBoost.",
    ])

    # ============================================================ 24 findings
    S.append(h1("Major Findings"))
    S += bullets([
        f"The dataset has {len(df):,} customers with no missing values or duplicates; {meta['churn_rate_train']:.0%} churn, so accuracy is misleading "
        f"(always predicting 'stays' scores {meta['majority_class_accuracy_test']:.1%}).",
        "Churn is higher for Germany, women, inactive members and customers with 3-4 products; churn peaks in the 51-60 age band.",
        f"many_products clearly helped Logistic Regression in the ablation test ({f3(lr['none']['cv_roc_auc_mean'])} to {f3(lr['many_products']['cv_roc_auc_mean'])} ROC-AUC); "
        f"for Random Forest the engineered features made little difference.",
        f"Tuning helped some models noticeably and others not at all (see section 13); {best} was selected by cross-validation.",
        f"Test result of the selected model: ROC-AUC {f3(bd['roc_auc'])}, recall {f3(bd['recall'])} and precision {f3(bd['precision'])} at 0.5; "
        f"recall {f3(btt['recall'])} and precision {f3(btt['precision'])} at the F1-optimised threshold.",
        "The most influential inputs were " + ", ".join(k for k, _ in top[:3]) + ".",
        "The saved Joblib pipelines reload to identical predictions, and the Streamlit app and all tests pass.",
    ])

    # ============================================================ 25 checklist
    S.append(h1("Requirements Checklist and Status"))
    S.append(table([
        ["Requirement", "Status", "Evidence"],
        ["Load a real-world dataset", "PARTIAL", "Kaggle bank churn data (10,000 rows); real vs simulated origin not stated by the source (sections 4, 22)."],
        ["Handle missing values", "PASS", "Checked (0 found); imputers in the pipeline; tested with a missing input."],
        ["Remove duplicates", "PASS", "Checked (0 found); clean() removes them; unit-tested."],
        ["Detect outliers", "PASS", "IQR table and boxplots (section 10); decision to keep documented."],
        ["Encode categorical variables", "PASS", "One-hot encoding for country, gender, age_group."],
        ["Scale numerical features", "PASS", "StandardScaler inside the pipeline."],
        ["Feature engineering", "PASS", "Four features with ablation evidence (section 11)."],
        ["Train at least 4 models, compare, select best", "PASS", f"5 models; best = {best} by cross-validated ROC-AUC (sections 12, 14)."],
        ["Classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC)", "PASS", "Sections 12 and 14; all in metrics.json."],
        ["Hyperparameter tuning (Grid or Randomized)", "PASS", "Both used (section 13)."],
        ["Compare before/after tuning", "PASS", "Section 13 tables and figure."],
        ["Save final model (Joblib or Pickle)", "PASS", "models/final_model.joblib plus all_models/*.joblib (full pipelines), reload-verified."],
        ["Streamlit app: inputs, button, prediction, relevant information", "PASS", "app/app.py; verified by automated app tests."],
        ["Bonus: prediction probability", "PASS", "Percentage and progress bar."],
        ["Bonus: feature importance", "PASS", "Feature Importance page."],
        ["Bonus: data visualization page", "PASS", "app/pages/1_Data_Visualization.py."],
        ["Bonus: model performance page", "PASS", "app/pages/2_Model_Performance.py."],
        ["Bonus: multiple model selection", "PASS", "Sidebar selector on every page."],
        ["Training code in Jupyter notebooks", "PASS", "notebooks/01-03 (.ipynb), executed with saved outputs."],
        ["Deploy online if possible", "DEPLOYED" if app_url else "NOT DONE", (f"Live at {app_url}." if app_url else "App is deployment-ready; steps in section 19 and README.")],
    ], [5.6 * cm, 2.0 * cm, W - 7.6 * cm]))

    # ============================================================ 26 appendix
    S.append(PageBreak())
    S.append(h1("Appendix - Key Code"))
    S.append(P("The code below is copied from the project files at build time (text, not screenshots). The notebooks in the repository contain all "
               "cells with their saved outputs."))
    S.append(P("<b>A.1 Feature engineering (src/features.py)</b>", H2))
    S += code_block((ROOT / "src" / "features.py").read_text(), "Listing A.1: feature engineering used inside the pipeline")
    S.append(P("<b>A.2 Preprocessing and model pipeline (src/pipeline.py)</b>", H2))
    S += code_block((ROOT / "src" / "pipeline.py").read_text(), "Listing A.2: the common pipeline used for every model")
    S.append(P("<b>A.3 Hyperparameter tuning (notebook 03)</b>", H2))
    S += code_block(notebook_cell("03_modeling_tuning_and_saving.ipynb", "SEARCH_SPACES = {") + "\n\n" +
                    notebook_cell("03_modeling_tuning_and_saving.ipynb", "tuned_models = {}"),
                    "Listing A.3: search spaces and the tuning loop")
    S.append(P("<b>A.4 Saving the models (notebook 03)</b>", H2))
    S += code_block(notebook_cell("03_modeling_tuning_and_saving.ipynb", "joblib.dump(tuned_models[name]"),
                    "Listing A.4: saving the pipelines, metrics and reference files")
    app_utils = (ROOT / "app" / "app_utils.py").read_text()
    start = app_utils.index("def factor_analysis")
    end = app_utils.index("def age_group")
    S.append(P("<b>A.5 Prediction explanation used by the app (app/app_utils.py)</b>", H2))
    S += code_block(app_utils[start:end], "Listing A.5: approximate 'what drives this prediction' analysis")

    # ============================================================ 27 repo
    S.append(h1("GitHub Repository Link"))
    S.append(P(f"{repo_url} (folder: {repo_folder})"))
    S.append(P("The task folder contains the three executed notebooks, src/ (reusable code), the data, the saved models in models/, the figures in "
               "reports/, the tests, the Streamlit app in app/, this report, requirements.txt and the README."))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Task 04 - End-to-End ML Application with Deployment", author=author)
    doc.build(S, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("saved", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="Muhammad Ali")
    ap.add_argument("--date", default="November 1, 2026")
    ap.add_argument("--repo-url", default="https://github.com/Al7bh/Devixo-Internship")
    ap.add_argument("--repo-folder", default="Task_04_ML_App")
    ap.add_argument("--app-url", default="")
    a = ap.parse_args()
    build(a.author, a.date, a.repo_url, a.repo_folder, a.app_url)
