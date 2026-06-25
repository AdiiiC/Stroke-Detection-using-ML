# Stroke Risk Prediction — Explainable, Calibrated & Fair ML

An end-to-end, production-grade machine-learning system that estimates
stroke risk from tabular clinical data, with an optional deep-learning
imaging branch (brain CT classification + MRI lesion segmentation) and
late fusion. The project is built around the realities of clinical ML:
**severe class imbalance, the high cost of false negatives, the need for
calibrated probabilities, explainability, and fairness.**

> ⚠️ **Research / educational use only — not a medical device.** See
> [MODEL_CARD.md](MODEL_CARD.md).

---

## Why this is more than a notebook

| Concern | What this project does |
|---|---|
| Imbalance (~4% positive) | SMOTE / class-weighting **inside** CV folds (no leakage), PR-AUC as the headline metric |
| Cost of a missed stroke | Threshold tuned to hit a **recall target** while maximising F‑β (β=2) |
| Trustworthy probabilities | Isotonic **calibration** + Brier score + reliability curve |
| Explainability | **SHAP** global summary + per-patient contributions served via the API |
| Fairness | Subgroup **audit** (gender, age band, residence, work type) with disparity gaps |
| Reproducibility | **Optuna** tuning, **MLflow** tracking, pinned deps, deterministic seeds |
| Serving | **FastAPI** scoring service + a **high-end React (Vite + TS)** dashboard |
| Ops | **Docker** / docker-compose, **pytest** suite, **GitHub Actions** CI |
| Multimodality | **CNN** (EfficientNet/ResNet) + **Grad-CAM**, **U-Net** segmentation, **late fusion** |

---

## Architecture

```
                 ┌────────────────────────── Tabular branch ──────────────────────────┐
 train.csv ─► clean ─► feature engineering ─► ColumnTransformer ─► SMOTE* ─► GBT ─► calibrate ─► threshold
                                                                                    │
                                                  SHAP · fairness audit · MLflow ◄──┤
                                                                                    ▼
                                                                    FastAPI  +  Streamlit
                 ┌────────────────────────── Imaging branch ──────────────────────────┐
 CT images ─► CNN (transfer learning) ─► Grad-CAM saliency ─────────┐
 MRI images ─► U-Net ─► lesion mask + Dice ─────────────────────────┤
                                                                    ▼
                                         WeightedFusion / StackedFusion(p_tab, p_img)
```
`*` resampling only ever sees the training fold.

---

## Project layout

```
src/stroke/            core tabular package
  config.py            paths, schema, hyper-defaults, TrainConfig
  data.py              load/split, age bands, class-balance stats
  features.py          ClinicalFeatureEngineer + preprocessing (flat steps)
  train.py             estimators, resamplers, Optuna tune, CV, calibration
  metrics.py           PR-AUC-first evaluation + recall-target thresholding
  evaluate.py          ROC/PR, calibration, confusion, SHAP figures
  fairness.py          subgroup audit + disparity gaps
  predict.py           load model/threshold, risk bands
  run.py               CLI: train / evaluate / predict-test (+ MLflow)
  imaging/             dataset, CNN model, Grad-CAM, U-Net, fusion, train_cnn
api/                   FastAPI service + SHAP explainer
frontend/              React + TypeScript (Vite) dashboard — high-end UI
dashboard/             legacy Streamlit clinician UI (optional)
scripts/               Kaggle downloader, fusion demo
tests/                 pytest suite (tabular, API, imaging)
artifacts/             models, figures, reports, submissions (generated)
Dockerfile · docker-compose.yml · Makefile · .github/workflows/ci.yml
```

---

## Quickstart

```bash
make setup            # python3.12 venv + core deps
source .venv/bin/activate

make train            # Optuna + CV + calibration + threshold + SHAP + fairness
make evaluate         # re-evaluate persisted model on the validation split
make submit           # write Kaggle submission from test.csv

make api              # FastAPI on http://localhost:8000  (/docs for Swagger)
make frontend-install # once: install React deps
make frontend         # React UI on http://localhost:5173
make docker           # api + React frontend via docker compose
make test             # pytest
```

Equivalent raw commands (from the project root, venv active):

```bash
PYTHONPATH=src python -m stroke.run train --model lightgbm --trials 40
PYTHONPATH=src uvicorn api.main:app --port 8000
npm --prefix frontend install && npm --prefix frontend run dev
```

### Headline results (validation split, 40 Optuna trials, LightGBM)

| Metric | Value |
|---|---|
| ROC-AUC | ~0.90 |
| PR-AUC (avg precision) | ~0.31 |
| Recall @ tuned threshold | ~0.87 |
| Brier score | ~0.033 |

Accuracy is deliberately **not** headline — at a 4% base rate a "predict
all negative" model scores 96% accuracy yet catches zero strokes. Figures
and `artifacts/reports/metrics.json` are regenerated on every run.

---

## API

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "gender":"Male","age":67,"hypertension":1,"heart_disease":1,
  "ever_married":"Yes","work_type":"Private","Residence_type":"Urban",
  "avg_glucose_level":228.69,"bmi":36.6,"smoking_status":"formerly smoked"
}'
```

Returns the calibrated probability, the risk band, the binary decision at
the tuned threshold, and the top SHAP feature contributions. Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + model-loaded status |
| `GET /metadata` · `GET /metrics` | model card / full metrics |
| `GET /fairness` | subgroup audit + disparity gaps |
| `GET /feature-importance` | global SHAP drivers |
| `GET /samples` | preset demo patients |
| `POST /predict` · `POST /predict/batch` | score one / many patients |
| `POST /whatif` | actionable counterfactual risk-reduction levers |

CORS is enabled for the React dev server.

---

## Frontend (React + TypeScript + Vite)

A high-end single-page dashboard in [`frontend/`](frontend/):

```bash
make frontend-install   # once
make api                # terminal A — http://localhost:8000
make frontend           # terminal B — http://localhost:5173
```

Features: animated SVG **risk gauge**, per-patient **SHAP** attribution
bars, a **what-if simulator** of modifiable risk factors, a **model
intelligence** panel (performance KPIs, global drivers, fairness audit),
and **cohort CSV scoring** with risk triage — all in a glassmorphism dark
UI. The Vite dev server proxies `/api` to FastAPI; the production build is
served by nginx (`make docker`, frontend on :8080). A legacy Streamlit
dashboard remains available via `make dashboard`.

---

## Imaging branch (optional, deep learning)

```bash
make install-imaging                                   # torch, grad-cam, etc.

# 1) download a real brain-CT stroke dataset (needs kaggle.json)
python scripts/download_data.py --dataset ct

# 2) train the CNN classifier with Grad-CAM
PYTHONPATH=src python -m stroke.imaging.train_cnn \
    --data-dir data/imaging/ct --backbone efficientnet_b0 --epochs 25

# quick smoke test on procedurally-generated scans (no download)
make imaging-quick
```

- **`imaging/model.py`** — EfficientNet-B0 / ResNet transfer learning.
- **`imaging/gradcam.py`** — Grad-CAM saliency overlays (with a
  dependency-free fallback).
- **`imaging/unet.py`** — U-Net + Dice / BCE-Dice losses for MRI lesion
  segmentation.
- **`imaging/fusion.py`** — `WeightedFusion` and `StackedFusion` combine
  the tabular and imaging probabilities. Run `make fusion-demo` to see
  fusion lift PR-AUC above either branch alone.

> The public tabular and CT datasets describe different patients, so true
> paired fusion needs a linked cohort. The fusion module and demo show the
> machinery end-to-end; swap in real paired probabilities when available.

---

## Experiment tracking

MLflow logs params, metrics, and artifacts to a local SQLite backend:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

---

## Testing & CI

`make test` runs the pytest suite (data, features, metrics, pipeline,
API, imaging). GitHub Actions trains a fast model and runs the suite on
every push (`.github/workflows/ci.yml`).

---

## Responsible use

This model is trained on public competition data, captures correlation
rather than causation, and is **not** validated for clinical use. Read
[MODEL_CARD.md](MODEL_CARD.md) before reusing it.

## Author

Adithya C
