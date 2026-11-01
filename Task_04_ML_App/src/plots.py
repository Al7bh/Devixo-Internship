"""Plot functions used by BOTH the Streamlit app and the report.

Every function takes the loaded metrics dict (models/metrics.json) and returns a
matplotlib Figure, so charts always show the real saved results - nothing is
hard-coded.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def model_comparison(metrics: dict, stage: str = "tuned"):
    """Grouped bars of the main metrics for every model (at threshold 0.5)."""
    _style()
    names = metrics["model_order"]
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [metrics["models"][n][stage]["test_default_threshold"][k] for k in keys]
        ax.bar(np.arange(len(keys)) + i * width, vals, width, label=n, color=PALETTE[i % 5])
    ax.axhline(metrics["meta"]["majority_class_accuracy_test"], color="grey", ls="--", lw=1)
    ax.text(len(keys) - 0.45, metrics["meta"]["majority_class_accuracy_test"] + 0.01,
            "accuracy of always predicting 'stays'", ha="right", fontsize=8, color="grey")
    ax.set_xticks(np.arange(len(keys)) + width * (len(names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Test-set metrics at threshold 0.5 ({stage} models)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def roc_curves(metrics: dict, stage: str = "tuned", highlight: str = None):
    _style()
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, n in enumerate(metrics["model_order"]):
        m = metrics["models"][n][stage]
        auc = m["test_default_threshold"]["roc_auc"]
        lw = 2.8 if n == highlight else 1.5
        ax.plot(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"], lw=lw, color=PALETTE[i % 5],
                label=f"{n} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(f"ROC curves on the test set ({stage} models)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def confusion_matrix_plot(metrics: dict, model: str, threshold: str = "default", stage: str = "tuned"):
    _style()
    key = "test_default_threshold" if threshold == "default" else "test_tuned_threshold"
    m = metrics["models"][model][stage][key]
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=16,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stays", "Churns"]); ax.set_yticklabels(["Stays", "Churns"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{model}\nthreshold = {m['threshold']:.2f}")
    fig.tight_layout()
    return fig


def tuning_before_after(metrics: dict, metric: str = "roc_auc"):
    """Before vs after tuning for each model (test set, threshold 0.5)."""
    _style()
    names = metrics["model_order"]
    before = [metrics["models"][n]["baseline"]["test_default_threshold"][metric] for n in names]
    after = [metrics["models"][n]["tuned"]["test_default_threshold"][metric] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, before, 0.4, label="Before tuning", color="#9ecae1")
    ax.bar(x + 0.2, after, 0.4, label="After tuning", color="#08519c")
    for xi, b, a in zip(x, before, after):
        ax.text(xi - 0.2, b + 0.005, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{metric.upper().replace('_', '-')} on the test set: before vs after tuning")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    return fig


def feature_importance(metrics: dict, model: str, top_n: int = 10):
    """Permutation importance (drop in ROC-AUC when a column is shuffled)."""
    _style()
    imp = metrics["models"][model]["tuned"]["permutation_importance"]
    items = sorted(imp.items(), key=lambda kv: kv[1]["mean"])[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([k for k, _ in items], [v["mean"] for _, v in items],
            xerr=[v["std"] for _, v in items], color="#2ca02c", ecolor="#555", capsize=2)
    ax.set_xlabel("Drop in test ROC-AUC when the column is shuffled")
    ax.set_title(f"Permutation feature importance - {model}")
    fig.tight_layout()
    return fig


def feature_ablation_plot(ablation: dict):
    _style()
    fig, axes = plt.subplots(1, len(ablation), figsize=(11, 3.8), sharey=False)
    for ax, (model, cfgs) in zip(np.atleast_1d(axes), ablation.items()):
        names = list(cfgs)
        means = [cfgs[n]["cv_roc_auc_mean"] for n in names]
        stds = [cfgs[n]["cv_roc_auc_std"] for n in names]
        ax.barh(names, means, xerr=stds, color="#6baed6", ecolor="#444", capsize=2)
        lo = min(means) - 0.03
        ax.set_xlim(lo, max(means) + 0.03)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("5-fold CV ROC-AUC (training data)")
        ax.invert_yaxis()
    fig.suptitle("Feature engineering ablation: engineered features added one at a time", fontsize=11)
    fig.tight_layout()
    return fig


def calibration_plot(error_analysis: dict):
    """Reliability diagram: predicted probability vs how often customers really churned."""
    _style()
    pts = [c for c in error_analysis["calibration"] if c["n"] > 0]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.plot([c["mean_predicted"] for c in pts], [c["observed_rate"] for c in pts],
            "o-", color="#d62728", label="Model (test set)")
    for c in pts:
        ax.annotate(f"n={c['n']}", (c["mean_predicted"], c["observed_rate"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=7)
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Calibration: do probabilities match reality?")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig
