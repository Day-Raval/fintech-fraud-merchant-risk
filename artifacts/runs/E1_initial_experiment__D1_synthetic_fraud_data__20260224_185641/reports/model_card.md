# Model Card — Fraud Detection

## Run
- run_id: E1_initial_experiment__D1_synthetic_fraud_data__20260224_185641
- dataset_id: D1_synthetic_fraud_data
- dataset_fingerprint: 87a1e708d8
- sigmoid_shift: 4.8
- dataset_path: /mnt/c/Users/dayes/Production ML Systems/fintech-fraud-merchant-risk/data/datasets/D1_synthetic_fraud_data/processed/fraud_dataset.csv

## Decisioning Policy
- fp_investigation_cost: 7.5
- alert_topk_per_day: 200
- fn_loss_multiplier: 1.0
- best_threshold (valid cost): 0.0606

## Evaluation
- valid PR-AUC: 0.3103
- test PR-AUC:  0.2881
- valid recall@topK: 0.3921
- test  recall@topK: 0.3812
- valid alert rate: 36.7985%
- test  alert rate: 37.1023%
