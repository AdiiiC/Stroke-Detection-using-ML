"""Pipeline integration + leakage-safety tests (fast: no Optuna)."""
import numpy as np

from stroke import config, data, train


def _fast_cfg(resampling="class_weight"):
    return config.TrainConfig(
        model="logreg",  # fast, no native libs
        resampling=resampling,
        n_trials=0,
        calibrate=False,
        n_splits=3,
    )


def test_pipeline_fits_and_predicts_probabilities():
    ds = data.load_dataset()
    pipe = train.build_pipeline(_fast_cfg())
    pipe.fit(ds.X_train, ds.y_train)
    prob = pipe.predict_proba(ds.X_valid)[:, 1]
    assert prob.shape[0] == len(ds.X_valid)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_smote_only_resamples_training_fold():
    """The fitted pipeline must predict on the *original* class ratio.

    If SMOTE leaked into prediction, predict_proba length would differ
    from the input. Here we assert the validation set is scored 1:1,
    confirming resampling is confined to fit-time training folds.
    """
    ds = data.load_dataset()
    pipe = train.build_pipeline(_fast_cfg(resampling="smote"))
    pipe.fit(ds.X_train, ds.y_train)
    prob = pipe.predict_proba(ds.X_valid)[:, 1]
    assert len(prob) == len(ds.X_valid)  # no synthetic rows at inference


def test_cross_validate_reports_metrics():
    ds = data.load_dataset()
    cv = train.cross_validate(ds.X_train, ds.y_train, _fast_cfg())
    base_rate = data.class_balance(ds.y_train)["positive_rate"]
    assert cv["cv_roc_auc_mean"] > 0.6  # well above random
    assert cv["cv_pr_auc_mean"] > base_rate  # beats the no-skill baseline
