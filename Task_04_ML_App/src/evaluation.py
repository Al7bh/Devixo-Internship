"""Small reusable helpers for evaluating models (used by notebook 03).

The training and tuning code itself lives in notebooks/03_modeling_tuning_and_saving.ipynb,
so you can read it step by step. These helpers only hold the pieces that are
repeated for every model.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_predict

from src.features import add_features


def slug(name: str) -> str:
    """'Random Forest' -> 'random_forest' (used in file names)."""
    return name.lower().replace(" ", "_")


def metrics_at(y_true, proba, threshold=0.5) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC and confusion matrix at a threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def best_f1_threshold(y_true, proba) -> float:
    """Decision threshold that maximises F1 (use OUT-OF-FOLD predictions on training data)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def roc_points(y_true, proba, n=101) -> dict:
    """ROC curve resampled to n points (keeps metrics.json small)."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    grid = np.linspace(0, 1, n)
    return {"fpr": grid.tolist(), "tpr": np.interp(grid, fpr, tpr).tolist()}


def evaluate_pipeline(pipe, X_train, y_train, X_test, y_test, cv) -> dict:
    """Metrics for an already-fitted pipeline.

    The tuned threshold comes from out-of-fold predictions on the TRAINING data
    only, so the test set is never used to choose it.
    """
    proba_test = pipe.predict_proba(X_test)[:, 1]
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thr = best_f1_threshold(y_train, oof)
    return {
        "test_default_threshold": metrics_at(y_test, proba_test, 0.5),
        "test_tuned_threshold": metrics_at(y_test, proba_test, thr),
        "tuned_threshold": thr,
        "roc_curve": roc_points(y_test, proba_test),
    }


def reference_profile(X_train, y_train) -> dict:
    """Typical values and segment churn rates, from the TRAINING split only.
    The Streamlit app uses this for defaults, range warnings and explanations."""
    Xf = add_features(X_train)
    data = Xf.assign(churn=y_train.values)
    numeric = ["credit_score", "age", "tenure", "balance", "estimated_salary"]
    categorical = ["country", "gender", "products_number", "credit_card",
                   "active_member", "age_group"]
    seg = {}
    for c in categorical:
        g = data.groupby(c)["churn"].agg(["mean", "count"])
        seg[c] = {str(k): {"churn_rate": float(r["mean"]), "count": int(r["count"])}
                  for k, r in g.iterrows()}

    def py(v):  # numpy scalar -> plain Python value (JSON friendly)
        return v.item() if hasattr(v, "item") else v

    return {
        "n_train": int(len(X_train)),
        "overall_churn_rate_train": float(y_train.mean()),
        "numeric_median": {c: float(X_train[c].median()) for c in numeric},
        "numeric_min": {c: float(X_train[c].min()) for c in numeric},
        "numeric_max": {c: float(X_train[c].max()) for c in numeric},
        "mode": {c: py(X_train[c].mode().iloc[0])
                 for c in ["country", "gender", "products_number", "credit_card", "active_member"]},
        "segment_churn": seg,
    }


def error_analysis(X_test, y_test, proba, threshold) -> dict:
    """Where does the final model do well or badly? (test set)

    - segments: recall / precision / flag rate for each customer group
    - calibration: do predicted probabilities match real churn frequencies?
    """
    Xf = add_features(X_test).assign(actual=y_test.values, proba=proba)
    Xf["flagged"] = (Xf["proba"] >= threshold).astype(int)

    segments = {}
    for col in ["country", "gender", "active_member", "products_number", "age_group"]:
        rows = []
        for group, g in Xf.groupby(col):
            tp = int(((g["flagged"] == 1) & (g["actual"] == 1)).sum())
            fn = int(((g["flagged"] == 0) & (g["actual"] == 1)).sum())
            fp = int(((g["flagged"] == 1) & (g["actual"] == 0)).sum())
            rows.append({
                "group": str(group), "n": int(len(g)), "n_churned": int(g["actual"].sum()),
                "churn_rate": float(g["actual"].mean()),
                "flag_rate": float(g["flagged"].mean()),
                "recall": float(tp / (tp + fn)) if (tp + fn) else None,
                "precision": float(tp / (tp + fp)) if (tp + fp) else None,
            })
        segments[col] = rows

    edges = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001]
    labels = ["0-10%", "10-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    Xf["bin"] = np.digitize(Xf["proba"], edges[1:-1])
    calibration = []
    for i, lab in enumerate(labels):
        g = Xf[Xf["bin"] == i]
        calibration.append({
            "bin": lab, "n": int(len(g)),
            "mean_predicted": float(g["proba"].mean()) if len(g) else None,
            "observed_rate": float(g["actual"].mean()) if len(g) else None,
        })
    brier = float(np.mean((Xf["proba"] - Xf["actual"]) ** 2))
    return {"threshold": float(threshold), "segments": segments,
            "calibration": calibration, "brier_score": brier}
