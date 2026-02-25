# Model Card — Fraud Detection

## Run
- run_id: E2_experiment__D2_synthetic_fraud_data__20260224_191455
- dataset_id: D2_synthetic_fraud_data
- dataset_fingerprint: 6b7298e145
- sigmoid_shift: 5.5
- dataset_path: /mnt/c/Users/dayes/Production ML Systems/fintech-fraud-merchant-risk/data/datasets/D2_synthetic_fraud_data/processed/fraud_dataset.csv

## Decisioning Policy
- fp_investigation_cost: 7.5
- alert_topk_per_day: 200
- fn_loss_multiplier: 1.0
- best_threshold (valid cost): 0.0463

## Evaluation
- valid PR-AUC: 0.2989
- test PR-AUC:  0.2887
- valid recall@topK: 0.4557
- test  recall@topK: 0.4417
- valid alert rate: 27.1194%
- test  alert rate: 26.8270%
