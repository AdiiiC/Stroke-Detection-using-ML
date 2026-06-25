"""Inference helpers: load the persisted pipeline and score patients."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config


def load_model(path: Path = config.DEFAULT_MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No model at {path}. Train one first: python -m stroke.run train"
        )
    return joblib.load(path)


def load_threshold(path: Path = config.DEFAULT_THRESHOLD_PATH) -> float:
    if Path(path).exists():
        with open(path) as fh:
            return float(json.load(fh)["threshold"])
    return 0.5


def _coerce_frame(records: dict | list[dict] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, dict):
        df = pd.DataFrame([records])
    else:
        df = pd.DataFrame(records)
    # Ensure all expected raw columns exist; fill missing with NaN.
    for col in config.ALL_RAW_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df[config.ALL_RAW_FEATURES]


def predict_proba(model, records) -> np.ndarray:
    df = _coerce_frame(records)
    return model.predict_proba(df)[:, 1]


def predict(
    model,
    records,
    threshold: float | None = None,
) -> list[dict]:
    """Return risk score, label, and a coarse risk band per patient."""
    if threshold is None:
        threshold = load_threshold()
    probs = predict_proba(model, records)
    out = []
    for p in probs:
        out.append(
            {
                "stroke_probability": float(p),
                "predicted_stroke": int(p >= threshold),
                "risk_band": _band(p),
                "threshold": float(threshold),
            }
        )
    return out


def _band(p: float) -> str:
    if p >= 0.5:
        return "high"
    if p >= 0.2:
        return "elevated"
    if p >= 0.08:
        return "moderate"
    return "low"
