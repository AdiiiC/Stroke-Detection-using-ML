"""Model factory + leakage-safe training with CV, Optuna, calibration.

Key design points
-----------------
* Resampling (SMOTE / SMOTETomek) is wrapped in an ``imblearn.Pipeline``
  so it runs **inside** each CV training fold only -- synthetic points
  never leak into validation.
* Hyperparameters are tuned with Optuna using stratified K-fold, scoring
  on PR-AUC (average precision), the right metric for rare positives.
* The final estimator is probability-calibrated so a predicted "30%
  risk" is trustworthy.
"""
from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config, data, metrics
from .features import build_feature_steps

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    _HAS_OPTUNA = False

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False


# --------------------------------------------------------------------------- #
# Estimator factory
# --------------------------------------------------------------------------- #
def make_estimator(
    model: str,
    params: dict[str, Any] | None = None,
    *,
    spw: float = 1.0,
    use_class_weight: bool = True,
    random_state: int = config.RANDOM_STATE,
):
    """Instantiate a base classifier with imbalance handling baked in."""
    params = dict(params or {})

    if model == "lightgbm":
        if not _HAS_LGBM:
            raise ImportError("lightgbm is not installed")
        if use_class_weight:
            params.setdefault("class_weight", "balanced")
        else:
            params.setdefault("scale_pos_weight", spw)
        return LGBMClassifier(
            random_state=random_state, n_jobs=-1, verbose=-1, **params
        )

    if model == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed")
        if not use_class_weight:
            params.setdefault("scale_pos_weight", spw)
        return XGBClassifier(
            random_state=random_state,
            n_jobs=-1,
            eval_metric="aucpr",
            tree_method="hist",
            **params,
        )

    if model == "random_forest":
        if use_class_weight:
            params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(
            random_state=random_state, n_jobs=-1, **params
        )

    if model == "hist_gb":
        if use_class_weight:
            params.setdefault("class_weight", "balanced")
        return HistGradientBoostingClassifier(random_state=random_state, **params)

    if model == "logreg":
        if use_class_weight:
            params.setdefault("class_weight", "balanced")
        params.setdefault("max_iter", 1000)
        return LogisticRegression(random_state=random_state, **params)

    raise ValueError(f"Unknown model '{model}'")


def _resampler(kind: str, random_state: int):
    if kind == "smote":
        return ("resample", SMOTE(random_state=random_state))
    if kind == "smotetomek":
        return ("resample", SMOTETomek(random_state=random_state))
    return None


def build_pipeline(cfg: config.TrainConfig, params: dict | None = None, *, spw: float = 1.0):
    """Assemble preprocessing (+ optional resampling) + estimator."""
    use_class_weight = cfg.resampling in ("class_weight", "none")
    estimator = make_estimator(
        cfg.model,
        params,
        spw=spw,
        use_class_weight=use_class_weight,
        random_state=cfg.random_state,
    )

    steps = list(build_feature_steps())  # flat: ("engineer", ...), ("preprocess", ...)
    res = _resampler(cfg.resampling, cfg.random_state)
    if res is not None:
        steps.append(res)
    steps.append(("model", estimator))
    return ImbPipeline(steps=steps)


# --------------------------------------------------------------------------- #
# Hyperparameter search space
# --------------------------------------------------------------------------- #
def _suggest_params(trial, model: str) -> dict:
    if model == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
    if model == "hist_gb":
        return {
            "max_iter": trial.suggest_int("max_iter", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        }
    if model == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
        }
    return {}


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: config.TrainConfig,
) -> dict:
    """Optuna search maximising mean CV PR-AUC. Returns best params."""
    if not _HAS_OPTUNA or cfg.n_trials <= 0:
        return {}

    spw = data.scale_pos_weight(y)
    cv = StratifiedKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state
    )

    def objective(trial):
        params = _suggest_params(trial, cfg.model)
        pipe = build_pipeline(cfg, params, spw=spw)
        scores = cross_val_score(
            pipe, X, y, scoring="average_precision", cv=cv, n_jobs=-1
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
    )
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
    return study.best_params


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: config.TrainConfig,
    params: dict | None = None,
) -> dict:
    """Stratified CV reporting mean +/- std for the key metrics."""
    spw = data.scale_pos_weight(y)
    cv = StratifiedKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state
    )
    roc, pr = [], []
    for train_idx, val_idx in cv.split(X, y):
        pipe = build_pipeline(cfg, params, spw=spw)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        rep = metrics.evaluate(y.iloc[val_idx].to_numpy(), prob)
        roc.append(rep.roc_auc)
        pr.append(rep.pr_auc)
    return {
        "cv_roc_auc_mean": float(np.mean(roc)),
        "cv_roc_auc_std": float(np.std(roc)),
        "cv_pr_auc_mean": float(np.mean(pr)),
        "cv_pr_auc_std": float(np.std(pr)),
    }


def fit_final(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: config.TrainConfig,
    params: dict | None = None,
):
    """Fit the final (optionally calibrated) pipeline on all of X, y."""
    spw = data.scale_pos_weight(y)
    pipe = build_pipeline(cfg, params, spw=spw)

    if cfg.calibrate:
        # Calibrate the whole imbalanced pipeline with stratified CV.
        calibrated = CalibratedClassifierCV(
            pipe,
            method=cfg.calibration_method,
            cv=StratifiedKFold(
                n_splits=3, shuffle=True, random_state=cfg.random_state
            ),
        )
        calibrated.fit(X, y)
        return calibrated

    pipe.fit(X, y)
    return pipe
