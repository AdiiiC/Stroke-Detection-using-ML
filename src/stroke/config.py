"""Central configuration: paths, column groups, and constants.

Keeping every magic string and path in one place makes the rest of the
pipeline declarative and prevents the "encoding done differently in
train vs. inference" class of bugs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT
RAW_TRAIN_CSV = DATA_DIR / "train.csv"
RAW_TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"

MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"

# Imaging artefacts
IMAGING_DATA_DIR = PROJECT_ROOT / "data" / "imaging"
IMAGING_MODELS_DIR = MODELS_DIR / "imaging"

for _d in (
    ARTIFACTS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    SUBMISSIONS_DIR,
    IMAGING_MODELS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
TARGET = "stroke"
ID_COLUMN = "id"

# Raw columns as they appear in the CSV.
NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
# hypertension / heart_disease are already 0/1 integers -> treated as binary.
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

# Ordinal feature with a meaningful order (used by feature engineering).
SMOKING_ORDER = ["never smoked", "Unknown", "formerly smoked", "smokes"]

ALL_RAW_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Sensitive attributes for the fairness audit.
SENSITIVE_FEATURES = ["gender", "age_group", "Residence_type", "work_type"]

# --------------------------------------------------------------------------- #
# Reproducibility & CV
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.2

# Operating-point selection. Recall is prioritised: a missed stroke is far
# more costly than a false alarm. We target a minimum recall and pick the
# threshold that maximises F-beta (beta>1 weights recall).
TARGET_RECALL = 0.80
F_BETA = 2.0

# Default model registry filename.
DEFAULT_MODEL_PATH = MODELS_DIR / "stroke_pipeline.joblib"
DEFAULT_THRESHOLD_PATH = MODELS_DIR / "decision_threshold.json"


@dataclass
class TrainConfig:
    """Tunable knobs for a training run."""

    model: str = "lightgbm"  # one of: lightgbm, xgboost, random_forest, logreg, hist_gb
    resampling: str = "class_weight"  # class_weight | smote | smotetomek | none
    n_trials: int = 40  # Optuna trials (0 disables tuning)
    n_splits: int = N_SPLITS
    calibrate: bool = True
    calibration_method: str = "isotonic"  # isotonic | sigmoid
    random_state: int = RANDOM_STATE
    experiment_name: str = "stroke-tabular"
    tags: dict = field(default_factory=dict)
