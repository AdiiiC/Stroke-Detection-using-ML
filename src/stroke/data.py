"""Data loading, validation, and leakage-safe splitting.

All splitting happens *before* any fitting (encoders, imputers,
resamplers). Resampling such as SMOTE is applied only inside the
training fold via the imblearn pipeline, never here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


@dataclass
class Dataset:
    """A train/validation split plus the held-out competition test frame."""

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    X_test: pd.DataFrame | None = None
    test_ids: pd.Series | None = None


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light, leakage-free cleaning that is identical for train/test rows.

    - Normalises the obviously-invalid ``gender == "Other"`` to the mode
      is intentionally NOT done here (would leak the train mode); instead
      the categorical encoder handles unseen/rare values robustly.
    - Coerces numeric columns and leaves missing values for the imputer.
    """
    df = df.copy()
    for col in config.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.BINARY_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add a categorical age band used only for the fairness audit/EDA."""
    df = df.copy()
    bins = [0, 18, 35, 50, 65, 80, 200]
    labels = ["<18", "18-34", "35-49", "50-64", "65-79", "80+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    return df


def load_raw_train() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_TRAIN_CSV)
    return _basic_clean(df)


def load_raw_test() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_TEST_CSV)
    return _basic_clean(df)


def load_dataset(
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    include_competition_test: bool = True,
) -> Dataset:
    """Load and stratify-split the training data.

    The split is stratified on the target to preserve the ~4% positive
    rate in both folds.
    """
    train = load_raw_train()
    y = train[config.TARGET].astype(int)
    X = train[config.ALL_RAW_FEATURES].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_test = None
    test_ids = None
    if include_competition_test and config.RAW_TEST_CSV.exists():
        test = load_raw_test()
        test_ids = test[config.ID_COLUMN]
        X_test = test[config.ALL_RAW_FEATURES].copy()

    return Dataset(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True) if X_test is not None else None,
        test_ids=test_ids.reset_index(drop=True) if test_ids is not None else None,
    )


def class_balance(y: pd.Series) -> dict:
    """Return counts and positive rate for logging."""
    counts = y.value_counts().to_dict()
    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))
    total = pos + neg
    return {
        "n_total": total,
        "n_positive": pos,
        "n_negative": neg,
        "positive_rate": pos / total if total else float("nan"),
        "imbalance_ratio": (neg / pos) if pos else float("inf"),
    }


def scale_pos_weight(y: pd.Series) -> float:
    """negative/positive ratio for gradient-boosting ``scale_pos_weight``."""
    bal = class_balance(y)
    return bal["imbalance_ratio"] if np.isfinite(bal["imbalance_ratio"]) else 1.0
