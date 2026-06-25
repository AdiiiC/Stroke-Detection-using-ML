"""Feature engineering and the leakage-safe preprocessing pipeline.

Everything that *learns* from data (imputation statistics, one-hot
categories, scaler means) lives inside an sklearn ``Pipeline`` /
``ColumnTransformer`` so it is fit on training folds only and applied
identically at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, domain-driven feature creation.

    Stateless => safe to run before the split without leakage. Produces
    interaction and risk-bucket features grounded in stroke epidemiology
    (age, glucose/diabetes thresholds, BMI categories).
    """

    GLUCOSE_DIABETES_THRESHOLD = 125.0  # mg/dL fasting-equivalent proxy
    GLUCOSE_PREDIABETES_THRESHOLD = 100.0

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Age is the single strongest stroke risk factor -> non-linear terms.
        X["age_squared"] = X["age"] ** 2
        X["is_elderly"] = (X["age"] >= 65).astype(int)
        X["is_pediatric"] = (X["age"] < 18).astype(int)

        # Metabolic risk.
        X["glucose_category"] = np.select(
            [
                X["avg_glucose_level"] >= self.GLUCOSE_DIABETES_THRESHOLD,
                X["avg_glucose_level"] >= self.GLUCOSE_PREDIABETES_THRESHOLD,
            ],
            ["diabetic", "prediabetic"],
            default="normal",
        )
        X["bmi_category"] = np.select(
            [X["bmi"] < 18.5, X["bmi"] < 25, X["bmi"] < 30],
            ["underweight", "normal", "overweight"],
            default="obese",
        )

        # Cardiometabolic burden: count of co-morbidities.
        X["comorbidity_count"] = (
            X["hypertension"].fillna(0).astype(float)
            + X["heart_disease"].fillna(0).astype(float)
            + (X["avg_glucose_level"] >= self.GLUCOSE_DIABETES_THRESHOLD).astype(float)
        )

        # Interaction: elderly + metabolic stress compounds risk.
        X["age_glucose_interaction"] = X["age"] * X["avg_glucose_level"] / 100.0

        return X


# Columns produced/used downstream of feature engineering.
_ENG_NUMERIC = config.NUMERIC_FEATURES + [
    "age_squared",
    "comorbidity_count",
    "age_glucose_interaction",
]
_ENG_BINARY = config.BINARY_FEATURES + ["is_elderly", "is_pediatric"]
_ENG_CATEGORICAL = config.CATEGORICAL_FEATURES + ["glucose_category", "bmi_category"]


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer with per-type imputation + encoding/scaling."""
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary_pipe = Pipeline(
        steps=[("impute", SimpleImputer(strategy="most_frequent"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, _ENG_NUMERIC),
            ("bin", binary_pipe, _ENG_BINARY),
            ("cat", categorical_pipe, _ENG_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_steps() -> list[tuple[str, object]]:
    """Return feature engineering + preprocessing as *flat* pipeline steps.

    imblearn's ``Pipeline`` forbids a nested ``Pipeline`` as an
    intermediate step, so the feature stage is exposed as individual
    steps (``engineer`` then ``preprocess``) that the training module
    splices directly into the imblearn pipeline.
    """
    return [
        ("engineer", ClinicalFeatureEngineer()),
        ("preprocess", build_preprocessor()),
    ]


def build_feature_pipeline() -> Pipeline:
    """Standalone preprocessing pipeline (used for tests/standalone transforms)."""
    return Pipeline(steps=build_feature_steps())


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Recover output feature names from a fitted pipeline with a ``preprocess`` step."""
    pre: ColumnTransformer = fitted_pipeline.named_steps["preprocess"]
    return list(pre.get_feature_names_out())
