"""Streamlit clinician dashboard for stroke-risk prediction.

Run:
    streamlit run dashboard/app.py
The dashboard talks to the FastAPI service if STROKE_API_URL is set,
otherwise it loads the model in-process.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))  # so the `api` package is importable too

API_URL = os.environ.get("STROKE_API_URL", "").rstrip("/")

st.set_page_config(page_title="Stroke Risk Assistant", page_icon="🧠", layout="wide")


# --------------------------------------------------------------------------- #
# Prediction backends
# --------------------------------------------------------------------------- #
@st.cache_resource
def _local_model():
    from stroke import predict

    return predict.load_model(), predict.load_threshold()


@st.cache_resource
def _local_explainer():
    from api.explainer import PatientExplainer
    from stroke import predict

    return PatientExplainer(predict.load_model())


def predict_patient(record: dict) -> dict:
    if API_URL:
        import requests

        resp = requests.post(f"{API_URL}/predict", json=record, timeout=30)
        resp.raise_for_status()
        return resp.json()

    from stroke import predict

    model, threshold = _local_model()
    result = predict.predict(model, record, threshold=threshold)[0]
    try:
        result["explanation"] = _local_explainer().explain(record)
    except Exception:
        result["explanation"] = []
    return result


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def risk_gauge(prob: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Stroke risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f2937"},
                "steps": [
                    {"range": [0, 8], "color": "#86efac"},
                    {"range": [8, 20], "color": "#fde68a"},
                    {"range": [20, 50], "color": "#fdba74"},
                    {"range": [50, 100], "color": "#fca5a5"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def contribution_bar(explanation: list[dict]) -> go.Figure:
    explanation = list(reversed(explanation))
    feats = [e["feature"] for e in explanation]
    vals = [e["contribution"] for e in explanation]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h", marker_color=colors))
    fig.update_layout(
        title="Why this prediction? (SHAP contributions)",
        xaxis_title="← lowers risk      raises risk →",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
st.title("🧠 Stroke Risk Assistant")
st.caption(
    "Calibrated, explainable risk estimation from patient risk factors. "
    "**Research/educational use only — not a medical device.**"
)

with st.sidebar:
    st.header("Patient details")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 0, 100, 67)
    hypertension = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart disease")
    ever_married = st.selectbox("Ever married", ["Yes", "No"])
    work_type = st.selectbox(
        "Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    )
    residence = st.selectbox("Residence", ["Urban", "Rural"])
    glucose = st.number_input("Avg. glucose (mg/dL)", 40.0, 400.0, 170.0)
    bmi = st.number_input("BMI", 10.0, 70.0, 30.0)
    smoking = st.selectbox(
        "Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"]
    )
    submitted = st.button("Assess risk", type="primary", use_container_width=True)

record = {
    "gender": gender,
    "age": age,
    "hypertension": int(hypertension),
    "heart_disease": int(heart_disease),
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence,
    "avg_glucose_level": glucose,
    "bmi": bmi,
    "smoking_status": smoking,
}

if submitted:
    try:
        result = predict_patient(record)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.plotly_chart(
            risk_gauge(result["stroke_probability"], result["threshold"]),
            use_container_width=True,
        )
        band = result["risk_band"].upper()
        color = {"LOW": "green", "MODERATE": "orange", "ELEVATED": "orange", "HIGH": "red"}.get(band, "gray")
        st.markdown(f"### Risk band: :{color}[{band}]")
        st.metric(
            "Decision",
            "FLAG for review" if result["predicted_stroke"] else "No flag",
        )
    with col2:
        if result.get("explanation"):
            st.plotly_chart(contribution_bar(result["explanation"]), use_container_width=True)
        else:
            st.info("Explanations unavailable (install `shap`).")

    st.divider()
    st.caption(
        f"Operating threshold = {result['threshold']:.3f}, tuned to prioritise recall "
        "(catching strokes) over precision. A flag is a prompt for clinical review, "
        "not a diagnosis."
    )
else:
    st.info("Enter patient details in the sidebar and click **Assess risk**.")
