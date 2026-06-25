"""SHAP-based per-patient explanation for the served model.

Returns the signed contribution of each input feature toward the
predicted stroke risk, so a clinician sees *why* a patient was flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stroke.evaluate import _unwrap_pipeline


class PatientExplainer:
    def __init__(self, model):
        self._pipe = _unwrap_pipeline(model)
        self._explainer = None
        self._feat_names = None
        if self._pipe is not None:
            self._init_explainer()

    def _init_explainer(self):
        try:
            import shap
        except ImportError:
            return
        preprocess = self._pipe.named_steps["preprocess"]
        self._engineer = self._pipe.named_steps["engineer"]
        self._preprocess = preprocess
        self._model = self._pipe.named_steps["model"]
        self._feat_names = list(preprocess.get_feature_names_out())
        try:
            self._explainer = shap.TreeExplainer(self._model)
        except Exception:
            self._explainer = None

    def available(self) -> bool:
        return self._explainer is not None

    def explain(self, record: dict, top_k: int = 8) -> list[dict]:
        if self._explainer is None:
            return []
        df = pd.DataFrame([record])
        X = self._preprocess.transform(self._engineer.transform(df))
        vals = self._explainer.shap_values(X)
        if isinstance(vals, list):
            vals = vals[1]
        contrib = np.asarray(vals).reshape(-1)
        order = np.argsort(np.abs(contrib))[::-1][:top_k]
        return [
            {
                "feature": self._feat_names[i],
                "contribution": float(contrib[i]),
                "direction": "increases risk" if contrib[i] > 0 else "decreases risk",
            }
            for i in order
        ]

    def global_importance(self, n: int = 400, top_k: int = 12) -> list[dict]:
        """Mean |SHAP| across a sample of training data -> global drivers."""
        if self._explainer is None:
            return []
        try:
            from stroke import data as _data

            ds = _data.load_dataset()
            sample = ds.X_train.head(n)
        except Exception:
            return []
        X = self._preprocess.transform(self._engineer.transform(sample))
        vals = self._explainer.shap_values(X)
        if isinstance(vals, list):
            vals = vals[1]
        mean_abs = np.abs(np.asarray(vals)).mean(axis=0).reshape(-1)
        order = np.argsort(mean_abs)[::-1][:top_k]
        return [
            {"feature": self._feat_names[i], "importance": float(mean_abs[i])}
            for i in order
        ]

