"""API smoke tests using FastAPI's TestClient (in-process)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from stroke import config

client = TestClient(app)

_needs_model = pytest.mark.skipif(
    not config.DEFAULT_MODEL_PATH.exists(),
    reason="no trained model artifact; run `python -m stroke.run train` first",
)

HIGH_RISK = {
    "gender": "Male",
    "age": 79,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 250.0,
    "bmi": 35.0,
    "smoking_status": "smokes",
}
LOW_RISK = {
    "gender": "Female",
    "age": 22,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "No",
    "work_type": "Private",
    "Residence_type": "Rural",
    "avg_glucose_level": 85.0,
    "bmi": 21.0,
    "smoking_status": "never smoked",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@_needs_model
def test_predict_high_vs_low_risk_ordering():
    hi = client.post("/predict", json=HIGH_RISK).json()
    lo = client.post("/predict", json=LOW_RISK).json()
    assert hi["stroke_probability"] > lo["stroke_probability"]
    assert 0 <= lo["stroke_probability"] <= 1


def test_predict_validation_rejects_bad_age():
    bad = dict(HIGH_RISK, age=999)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


@_needs_model
def test_batch_predict():
    r = client.post("/predict/batch", json=[HIGH_RISK, LOW_RISK])
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2
