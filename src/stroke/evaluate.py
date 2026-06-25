"""Diagnostic plots and SHAP explanations.

Generates publication-style figures: ROC + PR curves, calibration
(reliability) curve, confusion matrix, and SHAP global importance. All
figures are written to ``artifacts/figures``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from . import config


def plot_roc_pr(y_true, y_prob, out_dir: Path = config.FIGURES_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0], name="model")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC Curve")
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[1], name="model")
    base = float(np.mean(y_true))
    axes[1].axhline(base, ls="--", color="grey", alpha=0.6, label=f"baseline={base:.3f}")
    axes[1].legend()
    axes[1].set_title("Precision-Recall Curve")
    fig.tight_layout()
    path = out_dir / "roc_pr_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"roc_pr_curves": str(path)}


def plot_calibration(y_true, y_prob, out_dir: Path = config.FIGURES_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed stroke frequency")
    ax.set_title("Calibration (Reliability) Curve")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "calibration_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"calibration_curve": str(path)}


def plot_confusion(y_true, y_pred, out_dir: Path = config.FIGURES_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["no stroke", "stroke"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix @ tuned threshold")
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"confusion_matrix": str(path)}


def shap_summary(
    fitted_pipeline,
    X_sample,
    out_dir: Path = config.FIGURES_DIR,
    max_display: int = 20,
) -> dict:
    """Global SHAP importance.

    Works on either a plain imblearn/sklearn pipeline (with ``features``
    and ``model`` steps) or a CalibratedClassifierCV wrapping one.
    """
    try:
        import shap
    except ImportError:  # pragma: no cover
        return {}

    pipe = _unwrap_pipeline(fitted_pipeline)
    if pipe is None:
        return {}

    engineer = pipe.named_steps["engineer"]
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    X_trans = preprocess.transform(engineer.transform(X_sample))
    feat_names = list(preprocess.get_feature_names_out())

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):  # binary -> take positive class
            shap_values = shap_values[1]
    except Exception:
        explainer = shap.Explainer(model.predict_proba, X_trans)
        shap_values = explainer(X_trans).values[..., 1]

    out_dir.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feat_names,
        max_display=max_display,
        show=False,
    )
    path = out_dir / "shap_summary.png"
    plt.gcf().tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return {"shap_summary": str(path)}


def _unwrap_pipeline(fitted):
    """Return a *fitted* inner imblearn/sklearn pipeline if present.

    For ``CalibratedClassifierCV`` fitted with cross-validation, the
    ``.estimator`` attribute is the unfitted template; the fitted clones
    live in ``calibrated_classifiers_``. We prefer a fitted clone so
    downstream SHAP/feature-name calls work.
    """
    from sklearn.calibration import CalibratedClassifierCV

    if isinstance(fitted, CalibratedClassifierCV):
        if getattr(fitted, "calibrated_classifiers_", None):
            inner = fitted.calibrated_classifiers_[0].estimator
            if hasattr(inner, "named_steps"):
                return inner
        est = getattr(fitted, "estimator", None)
        if est is not None and hasattr(est, "named_steps"):
            return est
        return None

    if hasattr(fitted, "named_steps"):
        return fitted
    return None
