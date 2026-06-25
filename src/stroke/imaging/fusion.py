"""Late fusion of the tabular risk model and the imaging classifier.

Two strategies:

* ``WeightedFusion`` -- convex combination ``w * p_tab + (1-w) * p_img``
  with ``w`` chosen to maximise validation PR-AUC. Zero training cost,
  fully interpretable.
* ``StackedFusion`` -- a logistic-regression meta-learner over the two
  base probabilities (and optionally their product), learning the
  optimal decision surface.

Either way the inputs are the *probabilities* from each modality, so the
branches stay decoupled and independently trainable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


@dataclass
class WeightedFusion:
    weight: float = 0.5  # weight on the tabular probability

    def predict_proba(self, p_tab: np.ndarray, p_img: np.ndarray) -> np.ndarray:
        return self.weight * np.asarray(p_tab) + (1 - self.weight) * np.asarray(p_img)

    @classmethod
    def fit(cls, p_tab, p_img, y) -> "WeightedFusion":
        best_w, best_score = 0.5, -1.0
        for w in np.linspace(0, 1, 51):
            score = average_precision_score(y, w * p_tab + (1 - w) * p_img)
            if score > best_score:
                best_score, best_w = score, w
        return cls(weight=float(best_w))


class StackedFusion:
    """Logistic meta-learner over [p_tab, p_img, p_tab*p_img]."""

    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)

    @staticmethod
    def _features(p_tab, p_img) -> np.ndarray:
        p_tab = np.asarray(p_tab).reshape(-1, 1)
        p_img = np.asarray(p_img).reshape(-1, 1)
        return np.hstack([p_tab, p_img, p_tab * p_img])

    def fit(self, p_tab, p_img, y) -> "StackedFusion":
        self.clf.fit(self._features(p_tab, p_img), y)
        return self

    def predict_proba(self, p_tab, p_img) -> np.ndarray:
        return self.clf.predict_proba(self._features(p_tab, p_img))[:, 1]
