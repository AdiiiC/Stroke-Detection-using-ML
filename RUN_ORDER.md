# RUN ORDER — Stroke Risk Prediction

Follow these steps top-to-bottom. Run everything from the project root
(`Stroke-Detection-using-ML/`). Steps 1–2 are required once; the rest are
on demand.

---

## 0. Prerequisites
- Python 3.12 (`python3.12 --version`)
- (Optional) Docker, for the containerised run in step 8
- (Optional) `kaggle.json` in `~/.kaggle/` for the imaging dataset (step 9)

---

## 1. Create the environment + install core deps  ⟵ run once
```bash
make setup
source .venv/bin/activate
```
Raw equivalent:
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

## 2. Train the tabular model  ⟵ produces the model everything else uses
```bash
make train
# = PYTHONPATH=src python -m stroke.run train --model lightgbm --trials 40
```
Writes: `artifacts/models/stroke_pipeline.joblib`,
`artifacts/models/decision_threshold.json`,
`artifacts/figures/*` (ROC/PR, calibration, confusion, SHAP),
`artifacts/reports/*` (metrics.json, fairness CSVs), and an MLflow run.

## 3. Evaluate the persisted model (optional)
```bash
make evaluate
```

## 4. Run the test suite (optional, recommended)
```bash
make test
# = python -m pytest -q -p no:warnings
```

## 5. Start the API  (needs step 2)
```bash
make api          # http://localhost:8000  ·  Swagger at /docs
```

## 6. Start the React frontend  (needs the API from step 5)
```bash
make frontend-install   # once: install npm deps
make frontend           # http://localhost:5173  (Vite dev server)
```
The dev server proxies `/api` → the FastAPI service on :8000, so no CORS
setup is needed. A legacy Streamlit dashboard is still available via
`make dashboard` (http://localhost:8501) if you prefer it.

## 7. Write a Kaggle submission (optional)
```bash
make submit       # reads test.csv -> artifacts/submissions/
```

## 8. Run the whole stack in Docker (alternative to 5+6)
```bash
make docker       # API on :8000, React frontend on :8080
```

## 9. Imaging branch (optional, deep learning)
```bash
make install-imaging                       # torch, grad-cam, etc.

# quick smoke test on synthetic scans (no download):
make imaging-quick

# OR train on real brain-CT data (needs kaggle.json):
python scripts/download_data.py --dataset ct
PYTHONPATH=src python -m stroke.imaging.train_cnn \
    --data-dir data/imaging/ct --backbone efficientnet_b0 --epochs 25
```

## 10. Fusion demo (optional; needs steps 2 + a CNN/synthetic signal)
```bash
make fusion-demo
```

## 11. View experiment tracking (optional)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

---

### Typical first run (minimum to get predictions)
```bash
make setup && source .venv/bin/activate   # 1
make train                                # 2
make api                                  # 5  (terminal A)
make frontend-install && make frontend    # 6  (terminal B) -> http://localhost:5173
```
