# 🛡️ Fintech Fraud & Merchant Risk

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="ML" src="https://img.shields.io/badge/ML-Fraud%20Detection-6A1B9A" />
  <img alt="API" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" />
  <img alt="Dashboard" src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img alt="Deployment" src="https://img.shields.io/badge/Deployment-Docker%20%2B%20Kubernetes-0db7ed?logo=kubernetes&logoColor=white" />
</p>

A production-grade machine learning system for **real-time card transaction fraud detection** and **merchant risk scoring**.

---

## 🎯 Project Goal

Predict whether a transaction is fraudulent **before authorization completes** while balancing operational constraints:

- 💸 Prevent high-value fraud losses
- 🧾 Control false positives and investigation workload
- 🚦 Respect alert volume limits (Top-K triage)
- ⚡ Maintain low-latency inference for production systems

## 📦 Current Scope

The repository currently includes:

- Synthetic data generation and dataset building pipelines
- Feature preprocessing and training workflows
- Fraud probability modeling and merchant risk modeling
- Evaluation outputs (metrics, threshold reports, model cards)
- A working FastAPI inference service with run loading and prediction endpoints
- A working Streamlit dashboard for experiment monitoring and batch CSV scoring

## 🧠 Key Outputs

- Transaction-level fraud probability
- Decision outcome (flag / pass)
- Human-readable reason codes for flagged transactions
- Merchant-level risk score based on observed behavior

## 🗺️ Roadmap

Planned near-term additions:

- More modeling and feature engineering experiments
- Endpoint hardening (auth/rate limits/request validation policies)
- Dashboard enhancements for deeper investigation workflows
- Experiment and data versioning with DVC
- Containerized model deployment using Docker on Kubernetes
- Deployment to a cloud provider

## 🏗️ Repository Layout (high level)

- `src/data/` – data generation and dataset assembly
- `src/features/` – preprocessing pipeline components
- `src/modeling/` – training, evaluation, merchant risk, reason codes
- `src/api/` – API service entrypoints and serving logic
- `src/dashboard/` – dashboard application
- `configs/` – experiment/configuration settings
- `artifacts/` – run outputs, models, and metrics

---

## 🚀 Quickstart: End-to-End Execution Steps

## 🐳 Container Setup and Run with Docker

Use Docker when you want to run the API and dashboard without creating a local Python virtual environment.

### 1) Prerequisites

- Docker Engine + Docker Compose plugin installed

### 2) Build and start containers

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

This will build and start:

- `fintech-risk-api` on `http://localhost:8000`
- `fintech-risk-dashboard` on `http://localhost:8501`

### 3) Check container status and logs

```bash
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f api
docker compose -f docker/docker-compose.yml logs -f dashboard
```

### 4) Stop containers

```bash
docker compose -f docker/docker-compose.yml down
```

### 5) Optional: run data generation + training before using `/predict`

If you want prediction endpoints and dashboard run metrics to use fresh local artifacts:

```bash
python -m src.data.generate --config configs/config.yaml
python -m src.data.build_dataset --config configs/config.yaml
python -m src.modeling.train --config configs/config.yaml
```

Because `data/` and `artifacts/` are mounted into both containers, newly generated files are available immediately.

---

### 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 2) Generate synthetic raw data

```bash
python -m src.data.generate --config configs/config.yaml
```

This creates raw files under:

- `data/datasets/<dataset_id>/raw/customers.csv`
- `data/datasets/<dataset_id>/raw/merchants.csv`
- `data/datasets/<dataset_id>/raw/cards.csv`
- `data/datasets/<dataset_id>/raw/transactions.csv`

### 3) Build processed training dataset

```bash
python -m src.data.build_dataset --config configs/config.yaml
```

This writes:

- `data/datasets/<dataset_id>/processed/fraud_dataset.csv`

### 4) Train model + run evaluation metrics

```bash
python -m src.modeling.train --config configs/config.yaml
```

Training run artifacts are saved under:

- `artifacts/runs/<run_id>/models/`
- `artifacts/runs/<run_id>/metrics/`
- `artifacts/runs/<run_id>/reports/model_card.md`

### 5) Review metrics

Check these files from the latest run:

- `artifacts/runs/<run_id>/metrics/metrics.json`
- `artifacts/runs/<run_id>/metrics/threshold_report.json`

### 6) Start the API (FastAPI)

The API serves health checks, run management, and online single-transaction scoring.

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

- `GET /health` → confirms service status and reports latest run ID
- `GET /runs` → lists available run IDs from `artifacts/runs`
- `POST /load?run_id=<run_id>` → loads a specific run (or latest run when omitted)
- `POST /predict` → scores one transaction payload and returns fraud probability, threshold decision, and reason codes

Open interactive API docs at:

- `http://localhost:8000/docs`

Example request:

```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "data": {
      "amount_usd": 725.0,
      "channel": "online",
      "merchant_country": "NG",
      "mcc": "7995",
      "card_present": 0,
      "is_night": 1,
      "card_txn_count_15m": 3,
      "merchant_txn_count_1h": 24,
      "distance_from_home_km": 180.0
    }
  }'
```

### 7) Start the Dashboard (Streamlit)

The dashboard surfaces run KPIs, trend tracking, run artifact inspection, and batch scoring.

```bash
streamlit run src/dashboard/app.py
```

Then open:

- `http://localhost:8501`

What you can do in the dashboard:

- Select any trained `run_id` from the sidebar
- Review KPIs (threshold, PR-AUC, net cost, alert rate)
- Inspect trends from `artifacts/metrics/experiments.csv` and `artifacts/metrics/generation.csv`
- View full metrics JSON, threshold report JSON, and model card for the selected run
- Upload a CSV for batch scoring and download a scored output file

---

## 🧪 Typical Local Workflow

Run these commands in order for a full local cycle:

```bash
python -m src.data.generate --config configs/config.yaml
python -m src.data.build_dataset --config configs/config.yaml
python -m src.modeling.train --config configs/config.yaml
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
streamlit run src/dashboard/app.py
```
---

## Kubernetes Deployment (API + Dashboard)

Use the manifests in `k8s/` to deploy the Dockerized API and dashboard.

### 1) Build images

From repo root:

```bash
docker build -f docker/Dockerfile.api -t fintech-risk-api:latest .
docker build -f docker/Dockerfile.dashboard -t fintech-risk-dashboard:latest .
```

### 2) Push to a registry (if your cluster cannot use local images)

```bash
docker tag fintech-risk-api:latest <your-registry>/fintech-risk-api:latest
docker tag fintech-risk-dashboard:latest <your-registry>/fintech-risk-dashboard:latest
docker push <your-registry>/fintech-risk-api:latest
docker push <your-registry>/fintech-risk-dashboard:latest
```

Then update `image:` in:

- `k8s/api-deployment.yaml`
- `k8s/dashboard-deployment.yaml`

### 3) Apply manifests

```bash
kubectl apply -k k8s/
```

### 4) Verify resources

```bash
kubectl -n fintech-risk get pods,svc,pvc,ingress
```

### 5) Access services

Ingress (from `k8s/ingress.yaml`):

- `http://fintech-risk.local/api/health`
- `http://fintech-risk.local/`

Or port-forward:

```bash
kubectl -n fintech-risk port-forward svc/fraud-api 8000:8000
kubectl -n fintech-risk port-forward svc/fraud-dashboard 8501:8501
```

Then open:

- `http://localhost:8000/docs`
- `http://localhost:8501`

---

## ✅ Project Status

This project is actively evolving. Expect regular updates as new experiments, API capabilities, dashboard functionality, and DVC integrations are being added.
