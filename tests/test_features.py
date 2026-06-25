import numpy as np
import pandas as pd

from stroke import config
from stroke.features import (
    ClinicalFeatureEngineer,
    build_feature_pipeline,
    get_feature_names,
)


def _sample_frame():
    return pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "age": [67.0, 30.0],
            "hypertension": [1, 0],
            "heart_disease": [1, 0],
            "ever_married": ["Yes", "No"],
            "work_type": ["Private", "children"],
            "Residence_type": ["Urban", "Rural"],
            "avg_glucose_level": [228.0, 90.0],
            "bmi": [36.6, 22.0],
            "smoking_status": ["formerly smoked", "never smoked"],
        }
    )


def test_feature_engineer_is_stateless_and_adds_columns():
    eng = ClinicalFeatureEngineer()
    out = eng.transform(_sample_frame())
    for col in [
        "age_squared",
        "is_elderly",
        "glucose_category",
        "bmi_category",
        "comorbidity_count",
        "age_glucose_interaction",
    ]:
        assert col in out.columns
    # Elderly diabetic with comorbidities -> high comorbidity count.
    assert out.loc[0, "comorbidity_count"] == 3
    assert out.loc[0, "glucose_category"] == "diabetic"


def test_preprocessor_handles_missing_bmi():
    df = _sample_frame()
    df.loc[0, "bmi"] = np.nan
    pipe = build_feature_pipeline()
    arr = pipe.fit_transform(df)
    assert not np.isnan(arr).any()  # imputed
    names = get_feature_names(pipe)
    assert len(names) == arr.shape[1]


def test_preprocessor_handles_unseen_category():
    pipe = build_feature_pipeline()
    pipe.fit(_sample_frame())
    novel = _sample_frame()
    novel.loc[0, "smoking_status"] = "vapes"  # unseen value
    arr = pipe.transform(novel)  # must not raise
    assert arr.shape[0] == 2
