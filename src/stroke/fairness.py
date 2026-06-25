"""Subgroup fairness audit.

Reports performance disparities across sensitive attributes (gender,
age band, residence, work type). Uses Fairlearn when available and
falls back to a pandas implementation otherwise so the audit always
runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from . import config, data


def _safe_auc(y_true, y_prob) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def audit(
    X_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    sensitive_features: list[str] | None = None,
) -> pd.DataFrame:
    """Per-subgroup metrics for each sensitive attribute.

    Returns a tidy DataFrame: one row per (attribute, group) with
    selection rate, recall (TPR), and AUC. Disparities are the spread
    across groups within an attribute.
    """
    sensitive_features = sensitive_features or config.SENSITIVE_FEATURES
    df = X_raw.copy()
    if "age_group" in sensitive_features and "age_group" not in df.columns:
        df = data.add_age_group(df)

    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for attr in sensitive_features:
        if attr not in df.columns:
            continue
        for group, idx in df.groupby(attr, observed=True).groups.items():
            mask = df.index.isin(idx)
            yt = np.asarray(y_true)[mask]
            yp = y_pred[mask]
            pr = y_prob[mask]
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "attribute": attr,
                    "group": str(group),
                    "n": int(mask.sum()),
                    "positive_rate": float(np.mean(yt)),
                    "selection_rate": float(np.mean(yp)),
                    "recall": float(recall_score(yt, yp, zero_division=0)),
                    "auc": _safe_auc(yt, pr),
                }
            )
    return pd.DataFrame(rows)


def disparities(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise max-min gaps per attribute (equal-opportunity style)."""
    out = []
    for attr, grp in audit_df.groupby("attribute"):
        out.append(
            {
                "attribute": attr,
                "recall_gap": float(grp["recall"].max() - grp["recall"].min()),
                "selection_rate_gap": float(
                    grp["selection_rate"].max() - grp["selection_rate"].min()
                ),
                "auc_gap": float(grp["auc"].max() - grp["auc"].min()),
                "worst_recall_group": str(grp.loc[grp["recall"].idxmin(), "group"]),
            }
        )
    return pd.DataFrame(out)
