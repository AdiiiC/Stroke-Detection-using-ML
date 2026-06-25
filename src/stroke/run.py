"""End-to-end training/evaluation orchestration with MLflow + a CLI.

Usage
-----
    python -m stroke.run train --model lightgbm --trials 40
    python -m stroke.run evaluate
    python -m stroke.run predict-test     # writes a Kaggle submission
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config, data, evaluate, fairness, metrics, train

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    _HAS_MLFLOW = False


def _start_mlflow(cfg: config.TrainConfig):
    if not _HAS_MLFLOW:
        return None
    # Newer MLflow deprecates the bare file store; use a local SQLite backend.
    mlflow.set_tracking_uri(f"sqlite:///{config.PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(cfg.experiment_name)
    return mlflow.start_run()


def run_train(cfg: config.TrainConfig) -> dict:
    ds = data.load_dataset()
    bal = data.class_balance(ds.y_train)
    print(f"[data] train balance: {bal}")

    run_ctx = _start_mlflow(cfg)
    try:
        print(f"[tune] Optuna ({cfg.n_trials} trials) on {cfg.model}...")
        best_params = train.tune(ds.X_train, ds.y_train, cfg)
        print(f"[tune] best params: {best_params}")

        print("[cv] cross-validating tuned config...")
        cv_metrics = train.cross_validate(ds.X_train, ds.y_train, cfg, best_params)
        print(f"[cv] {cv_metrics}")

        print("[fit] fitting final calibrated pipeline...")
        model = train.fit_final(ds.X_train, ds.y_train, cfg, best_params)

        # Validation-set evaluation + threshold tuning.
        val_prob = model.predict_proba(ds.X_valid)[:, 1]
        threshold = metrics.find_best_threshold(ds.y_valid.to_numpy(), val_prob)
        report = metrics.evaluate(ds.y_valid.to_numpy(), val_prob, threshold)
        print(f"[valid] {json.dumps(report.to_dict(), indent=2)}")

        # Persist model + threshold.
        joblib.dump(model, config.DEFAULT_MODEL_PATH)
        with open(config.DEFAULT_THRESHOLD_PATH, "w") as fh:
            json.dump({"threshold": threshold, "beta": config.F_BETA}, fh, indent=2)

        # Figures + explanations.
        figs = {}
        figs.update(evaluate.plot_roc_pr(ds.y_valid.to_numpy(), val_prob))
        figs.update(evaluate.plot_calibration(ds.y_valid.to_numpy(), val_prob))
        y_pred = (val_prob >= threshold).astype(int)
        figs.update(evaluate.plot_confusion(ds.y_valid.to_numpy(), y_pred))
        try:
            figs.update(evaluate.shap_summary(model, ds.X_valid.sample(
                min(500, len(ds.X_valid)), random_state=cfg.random_state)))
        except Exception as exc:  # pragma: no cover
            print(f"[shap] skipped: {exc}")

        # Fairness audit.
        audit_df = fairness.audit(ds.X_valid, ds.y_valid.to_numpy(), val_prob, threshold)
        disp_df = fairness.disparities(audit_df)
        audit_df.to_csv(config.REPORTS_DIR / "fairness_audit.csv", index=False)
        disp_df.to_csv(config.REPORTS_DIR / "fairness_disparities.csv", index=False)
        print(f"[fairness] disparities:\n{disp_df}")

        # Save a consolidated metrics report.
        report_payload = {
            "model": cfg.model,
            "resampling": cfg.resampling,
            "best_params": best_params,
            "class_balance": bal,
            **cv_metrics,
            "validation": report.to_dict(),
        }
        with open(config.REPORTS_DIR / "metrics.json", "w") as fh:
            json.dump(report_payload, fh, indent=2)

        if _HAS_MLFLOW and run_ctx is not None:
            mlflow.log_params({"model": cfg.model, "resampling": cfg.resampling, **best_params})
            mlflow.log_metrics(cv_metrics)
            mlflow.log_metrics({f"valid_{k}": v for k, v in report.to_dict().items()
                                if isinstance(v, (int, float))})
            for p in figs.values():
                if Path(p).exists():
                    mlflow.log_artifact(p, artifact_path="figures")
            mlflow.log_artifact(str(config.REPORTS_DIR / "metrics.json"))
            mlflow.log_artifact(str(config.REPORTS_DIR / "fairness_audit.csv"))
            try:
                mlflow.sklearn.log_model(model, name="model")
            except Exception:
                mlflow.log_artifact(str(config.DEFAULT_MODEL_PATH))

        return report_payload
    finally:
        if _HAS_MLFLOW and run_ctx is not None:
            mlflow.end_run()


def run_evaluate() -> dict:
    from .predict import load_model, load_threshold

    ds = data.load_dataset()
    model = load_model()
    threshold = load_threshold()
    prob = model.predict_proba(ds.X_valid)[:, 1]
    report = metrics.evaluate(ds.y_valid.to_numpy(), prob, threshold)
    print(json.dumps(report.to_dict(), indent=2))
    return report.to_dict()


def run_predict_test() -> Path:
    from .predict import load_model, load_threshold

    ds = data.load_dataset()
    if ds.X_test is None:
        raise FileNotFoundError("No test.csv available for submission.")
    model = load_model()
    prob = model.predict_proba(ds.X_test)[:, 1]
    sub = pd.DataFrame({config.ID_COLUMN: ds.test_ids, config.TARGET: prob})
    out = config.SUBMISSIONS_DIR / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"[submission] wrote {out} ({len(sub)} rows)")
    return out


def _build_cfg(args) -> config.TrainConfig:
    return config.TrainConfig(
        model=args.model,
        resampling=args.resampling,
        n_trials=args.trials,
        calibrate=not args.no_calibrate,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stroke risk pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train + evaluate + explain")
    p_train.add_argument("--model", default="lightgbm",
                         choices=["lightgbm", "xgboost", "random_forest", "hist_gb", "logreg"])
    p_train.add_argument("--resampling", default="class_weight",
                         choices=["class_weight", "smote", "smotetomek", "none"])
    p_train.add_argument("--trials", type=int, default=40)
    p_train.add_argument("--no-calibrate", action="store_true")

    sub.add_parser("evaluate", help="evaluate persisted model on validation split")
    sub.add_parser("predict-test", help="write Kaggle submission from test.csv")

    args = parser.parse_args(argv)
    if args.command == "train":
        run_train(_build_cfg(args))
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "predict-test":
        run_predict_test()


if __name__ == "__main__":
    main()
