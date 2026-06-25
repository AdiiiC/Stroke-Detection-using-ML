"""FastAPI service exposing the stroke-risk model.

Endpoints
---------
GET  /health             -> liveness + model-loaded status
GET  /metadata           -> model card summary, threshold, metrics
GET  /metrics            -> full training/validation metrics
GET  /fairness           -> subgroup audit + disparity gaps
GET  /feature-importance -> global SHAP feature importance
GET  /samples            -> preset demo patients (low/moderate/high)
POST /predict            -> risk score + band + (optional) SHAP explanation
POST /predict/batch      -> vectorised scoring for many patients
POST /whatif             -> actionable counterfactual risk-reduction levers
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys

# Make ``src`` importable when launched as ``uvicorn api.main:app``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stroke import config, predict  # noqa: E402
from api.explainer import PatientExplainer  # noqa: E402

app = FastAPI(
    title="Stroke Risk API",
    version="2.0.0",
    description="Explainable, calibrated stroke-risk prediction. "
    "Research/educational use only — not a medical device.",
)

# Allow the React frontend (dev server + container) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://localhost:3000",
        "http://localhost",
    ],
    allow_origin_regex=r"http://localhost(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


class Patient(BaseModel):
    gender: Literal["Male", "Female", "Other"] = "Male"
    age: float = Field(..., ge=0, le=120)
    hypertension: int = Field(0, ge=0, le=1)
    heart_disease: int = Field(0, ge=0, le=1)
    ever_married: Literal["Yes", "No"] = "Yes"
    work_type: Literal[
        "Private", "Self-employed", "Govt_job", "children", "Never_worked"
    ] = "Private"
    Residence_type: Literal["Urban", "Rural"] = "Urban"
    avg_glucose_level: float = Field(..., ge=0, le=500)
    bmi: Optional[float] = Field(None, ge=0, le=100)
    smoking_status: Literal[
        "never smoked", "formerly smoked", "smokes", "Unknown"
    ] = "never smoked"

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Male",
                "age": 67,
                "hypertension": 1,
                "heart_disease": 1,
                "ever_married": "Yes",
                "work_type": "Private",
                "Residence_type": "Urban",
                "avg_glucose_level": 228.69,
                "bmi": 36.6,
                "smoking_status": "formerly smoked",
            }
        }
    }


class PredictionResponse(BaseModel):
    stroke_probability: float
    predicted_stroke: int
    risk_band: str
    threshold: float
    explanation: list[dict] = []


@lru_cache(maxsize=1)
def _get_model():
    return predict.load_model()


@lru_cache(maxsize=1)
def _get_threshold() -> float:
    return predict.load_threshold()


@lru_cache(maxsize=1)
def _get_explainer() -> PatientExplainer:
    return PatientExplainer(_get_model())


@app.get("/health")
def health():
    try:
        _get_model()
        loaded = True
    except Exception:
        loaded = False
    return {"status": "ok", "model_loaded": loaded}


@app.get("/metadata")
def metadata():
    metrics_path = config.REPORTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return {
        "model_loaded": config.DEFAULT_MODEL_PATH.exists(),
        "threshold": _get_threshold(),
        "metrics": metrics.get("validation", {}),
        "cv": {k: v for k, v in metrics.items() if k.startswith("cv_")},
        "disclaimer": "Research/educational use only. Not a medical device.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_one(patient: Patient, explain: bool = True):
    try:
        model = _get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    record = patient.model_dump()
    result = predict.predict(model, record, threshold=_get_threshold())[0]
    if explain:
        try:
            result["explanation"] = _get_explainer().explain(record)
        except Exception:
            result["explanation"] = []
    return result


@app.post("/predict/batch")
def predict_batch(patients: list[Patient]):
    try:
        model = _get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    records = [p.model_dump() for p in patients]
    return {"predictions": predict.predict(model, records, threshold=_get_threshold())}


@app.get("/metrics")
def metrics():
    path = config.REPORTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Train a model first.")
    return json.loads(path.read_text())


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key, value in row.items():
            if key in ("attribute", "group", "worst_recall_group"):
                continue
            try:
                row[key] = float(value) if value not in ("", None) else None
            except (TypeError, ValueError):
                row[key] = None
    return rows


@app.get("/fairness")
def fairness():
    return {
        "audit": _read_csv(config.REPORTS_DIR / "fairness_audit.csv"),
        "disparities": _read_csv(config.REPORTS_DIR / "fairness_disparities.csv"),
    }


@lru_cache(maxsize=1)
def _global_importance() -> list[dict]:
    try:
        return _get_explainer().global_importance()
    except Exception:
        return []


@app.get("/feature-importance")
def feature_importance():
    return {"features": _global_importance()}


_SAMPLE_PATIENTS = {
    "high": {
        "label": "High-risk profile",
        "patient": {
            "gender": "Male", "age": 79, "hypertension": 1, "heart_disease": 1,
            "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
            "avg_glucose_level": 232.4, "bmi": 35.1, "smoking_status": "smokes",
        },
    },
    "moderate": {
        "label": "Borderline profile",
        "patient": {
            "gender": "Female", "age": 61, "hypertension": 1, "heart_disease": 0,
            "ever_married": "Yes", "work_type": "Self-employed", "Residence_type": "Rural",
            "avg_glucose_level": 142.0, "bmi": 29.4, "smoking_status": "formerly smoked",
        },
    },
    "low": {
        "label": "Low-risk profile",
        "patient": {
            "gender": "Female", "age": 31, "hypertension": 0, "heart_disease": 0,
            "ever_married": "No", "work_type": "Private", "Residence_type": "Rural",
            "avg_glucose_level": 84.0, "bmi": 22.5, "smoking_status": "never smoked",
        },
    },
}


@app.get("/samples")
def samples():
    return _SAMPLE_PATIENTS


class WhatIfLever(BaseModel):
    label: str
    new_probability: float
    delta: float
    relative_reduction: float


@app.post("/whatif")
def whatif(patient: Patient):
    """Estimate how modifiable risk factors would change the risk score."""
    try:
        model = _get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    threshold = _get_threshold()
    base_record = patient.model_dump()
    base = predict.predict(model, base_record, threshold=threshold)[0]
    base_p = base["stroke_probability"]

    levers: dict[str, dict] = {}
    if patient.hypertension == 1:
        levers["Treat hypertension"] = {"hypertension": 0}
    if patient.heart_disease == 1:
        levers["Manage heart disease"] = {"heart_disease": 0}
    if patient.smoking_status in ("smokes", "formerly smoked"):
        levers["Quit smoking"] = {"smoking_status": "never smoked"}
    if patient.avg_glucose_level > 140:
        levers["Lower glucose to 110 mg/dL"] = {"avg_glucose_level": 110.0}
    if patient.bmi is not None and patient.bmi > 25:
        levers["Reach healthy BMI (24)"] = {"bmi": 24.0}

    results: list[dict] = []
    for label, change in levers.items():
        rec = {**base_record, **change}
        new_p = predict.predict(model, rec, threshold=threshold)[0]["stroke_probability"]
        delta = new_p - base_p
        results.append(
            {
                "label": label,
                "new_probability": new_p,
                "delta": delta,
                "relative_reduction": (-delta / base_p) if base_p > 0 else 0.0,
            }
        )
    results.sort(key=lambda r: r["delta"])
    return {
        "base_probability": base_p,
        "risk_band": base["risk_band"],
        "levers": results,
    }
