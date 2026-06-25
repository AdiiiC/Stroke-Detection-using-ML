"""Demonstrate late fusion of the tabular and imaging branches.

Because the public tabular and CT datasets describe *different* patients
(no shared IDs), a true paired fusion needs a linked cohort. This demo
shows the fusion machinery working end-to-end by:

1. Scoring the held-out tabular validation set with the trained tabular
   model to obtain ``p_tab``.
2. Simulating an aligned imaging probability ``p_img`` that is correlated
   with the true label (stand-in for a CNN run on the same patients).
3. Fitting both WeightedFusion and StackedFusion and reporting PR-AUC for
   tabular-only, imaging-only, and fused predictions.

Replace step 2 with real CNN probabilities once a paired CT+tabular
cohort is available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke import data, predict  # noqa: E402
from stroke.imaging.fusion import StackedFusion, WeightedFusion  # noqa: E402


def main():
    ds = data.load_dataset()
    model = predict.load_model()
    p_tab = model.predict_proba(ds.X_valid)[:, 1]
    y = ds.y_valid.to_numpy()

    # Stand-in imaging signal: informative but noisier than tabular.
    rng = np.random.default_rng(0)
    p_img = np.clip(0.55 * y + rng.normal(0.15, 0.22, size=len(y)), 0, 1)

    weighted = WeightedFusion.fit(p_tab, p_img, y)
    stacked = StackedFusion().fit(p_tab, p_img, y)

    print(f"tabular-only   PR-AUC: {average_precision_score(y, p_tab):.4f}")
    print(f"imaging-only   PR-AUC: {average_precision_score(y, p_img):.4f}")
    print(
        f"weighted fusion PR-AUC: "
        f"{average_precision_score(y, weighted.predict_proba(p_tab, p_img)):.4f} "
        f"(w_tab={weighted.weight:.2f})"
    )
    print(
        f"stacked fusion  PR-AUC: "
        f"{average_precision_score(y, stacked.predict_proba(p_tab, p_img)):.4f}"
    )


if __name__ == "__main__":
    main()
