.PHONY: help setup install install-imaging train evaluate submit test lint api frontend frontend-install frontend-build dashboard docker imaging-quick fusion-demo clean

help:
	@echo "Targets:"
	@echo "  setup            create venv + install core deps"
	@echo "  install          install core requirements into current env"
	@echo "  install-imaging  install deep-learning imaging deps"
	@echo "  train            train tabular model (Optuna, calibration, SHAP, fairness)"
	@echo "  evaluate         evaluate persisted model on validation split"
	@echo "  submit           write Kaggle submission from test.csv"
	@echo "  test             run pytest"
	@echo "  lint             ruff check"
	@echo "  api              run FastAPI (uvicorn) on :8000"
	@echo "  frontend         run React (Vite) dev server on :5173"
	@echo "  frontend-install install React frontend npm deps"
	@echo "  frontend-build   production build of the React frontend"
	@echo "  dashboard        (legacy) run Streamlit dashboard on :8501"
	@echo "  docker           build + run api and React frontend via docker compose"
	@echo "  imaging-quick    CNN smoke test on synthetic scans"
	@echo "  fusion-demo      demonstrate tabular+imaging late fusion"

VENV?=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

setup:
	python3.12 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install:
	pip install -r requirements.txt

install-imaging:
	pip install -r requirements-imaging.txt

train:
	PYTHONPATH=src $(PY) -m stroke.run train --model lightgbm --trials 40

evaluate:
	PYTHONPATH=src $(PY) -m stroke.run evaluate

submit:
	PYTHONPATH=src $(PY) -m stroke.run predict-test

test:
	$(PY) -m pytest -q -p no:warnings

lint:
	$(PY) -m ruff check src api dashboard tests

api:
	PYTHONPATH=src $(VENV)/bin/uvicorn api.main:app --reload --port 8000

frontend-install:
	npm --prefix frontend install

frontend:
	npm --prefix frontend run dev

frontend-build:
	npm --prefix frontend run build

dashboard:
	PYTHONPATH=src $(VENV)/bin/streamlit run dashboard/app.py

docker:
	docker compose up --build

imaging-quick:
	PYTHONPATH=src $(PY) -m stroke.imaging.train_cnn --quick

fusion-demo:
	PYTHONPATH=src $(PY) scripts/run_fusion_demo.py

clean:
	rm -rf artifacts/figures/* artifacts/reports/* artifacts/submissions/* mlruns
