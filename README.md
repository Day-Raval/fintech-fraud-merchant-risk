# Fintech Fraud & Merchant Risk

A production-style machine learning project for **real-time card transaction fraud detection** and **merchant risk scoring**.

## Project Goal

Build a fraud detection system that can score transactions before authorization completes, while balancing business constraints such as:

- Preventing high-value fraud losses
- Controlling false positives and investigation workload
- Operating within alert volume limits (Top-K triage)
- Maintaining low-latency inference for production use

## Current Scope

The repository currently includes:

- Synthetic data generation and dataset building pipelines
- Feature preprocessing and training workflows
- Fraud probability modeling and merchant risk modeling
- Evaluation outputs (metrics, threshold reports, model cards)
- Initial API and dashboard application scaffolding

## Key Outputs

- Transaction-level fraud probability
- Decision outcome (flag / pass)
- Human-readable reason codes for flagged transactions
- Merchant-level risk score based on observed behavior

## Roadmap

Planned near-term additions:

- More modeling and feature engineering experiments
- A fuller production API
- A richer dashboard experience for monitoring and investigation workflows
- Experiment and data versioning with DVC
- Containerized model deployment using Docker on Kubernetes
- Deployment to a cloud provider

## Repository Layout (high level)

- `src/data/` – data generation and dataset assembly
- `src/features/` – preprocessing pipeline components
- `src/modeling/` – training, evaluation, merchant risk, reason codes
- `src/api/` – API service entrypoints and serving logic
- `src/dashboard/` – dashboard application
- `configs/` – experiment/configuration settings
- `artifacts/` – run outputs, models, and metrics

## Status

This project is actively evolving. Expect regular updates as new experiments, API capabilities, dashboard functionality, and deployment assets are added.
