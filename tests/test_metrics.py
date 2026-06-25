import numpy as np

from stroke import metrics


def test_threshold_respects_target_recall():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(50), np.zeros(450)]
    # Well-separated scores.
    prob = np.r_[rng.uniform(0.5, 1.0, 50), rng.uniform(0.0, 0.5, 450)]
    t = metrics.find_best_threshold(y, prob, target_recall=0.8)
    y_pred = (prob >= t).astype(int)
    recall = y_pred[y == 1].mean()
    assert recall >= 0.8


def test_evaluate_returns_full_report():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=200) < 0.1).astype(int)
    prob = rng.uniform(size=200)
    rep = metrics.evaluate(y, prob)
    d = rep.to_dict()
    for k in ["roc_auc", "pr_auc", "recall", "precision", "f1", "brier", "threshold"]:
        assert k in d
    assert d["tn"] + d["fp"] + d["fn"] + d["tp"] == 200
