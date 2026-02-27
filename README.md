# 🛡️ Fintech Fraud & Merchant Risk

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="ML" src="https://img.shields.io/badge/ML-Fraud%20Detection-6A1B9A" />
  <img alt="API" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" />
  <img alt="Dashboard" src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img alt="Deployment" src="https://img.shields.io/badge/Deployment-Docker%20%2B%20Kubernetes-0db7ed?logo=kubernetes&logoColor=white" />
</p>

A production-style machine learning project for **real-time card transaction fraud detection** and **merchant risk scoring**.

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
- Initial API and dashboard scaffolding

## 🧠 Key Outputs

- Transaction-level fraud probability
- Decision outcome (flag / pass)
- Human-readable reason codes for flagged transactions
- Merchant-level risk score based on observed behavior

## 🗺️ Roadmap

Planned near-term additions:

- More modeling and feature engineering experiments
- A fuller production API
- A richer dashboard for monitoring and investigation workflows
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

## 🚀 Quickstart: Execution Steps (through training + evaluation)

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

---

## ✅ Project Status

This project is actively evolving. Expect regular updates as new experiments, API capabilities, dashboard functionality, DVC integration, and cloud-ready deployment assets are added.
