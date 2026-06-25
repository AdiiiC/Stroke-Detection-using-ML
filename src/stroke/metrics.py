"""Evaluation metrics tuned for a rare, high-cost positive class.

Accuracy is deliberately demoted: at ~4% prevalence a constant
"no-stroke" predictor scores ~96%. The headline metrics here are
ROC-AUC, PR-AUC (average precision), recall, and a recall-weighted
F-beta at a tuned decision threshold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


@dataclass
class ClassificationReport:
    roc_auc: float
    pr_auc: float
    recall: float
    precision: float
    f1: float
    fbeta: float
    brier: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict:
        return asdict(self)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = config.TARGET_RECALL,
    beta: float = config.F_BETA,
) -> float:
    """Pick an operating point.

    Strategy: among thresholds achieving at least ``target_recall``,
    choose the one maximising F-beta (beta>1 favours recall). If no
    threshold reaches the target recall, fall back to the global F-beta
    maximiser.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_score = 0.5, -1.0
    fallback_t, fallback_score = 0.5, -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        rec = recall_score(y_true, y_pred, zero_division=0)
        fb = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

        if fb > fallback_score:
            fallback_score, fallback_t = fb, t
        if rec >= target_recall and fb > best_score:
            best_score, best_t = fb, t

    return best_t if best_score >= 0 else fallback_t


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    beta: float = config.F_BETA,
) -> ClassificationReport:
    """Compute the full metric suite at a (possibly tuned) threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if threshold is None:
        threshold = find_best_threshold(y_true, y_prob, beta=beta)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return ClassificationReport(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        fbeta=float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        brier=float(brier_score_loss(y_true, y_prob)),
        threshold=float(threshold),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
