# Model Card — Fraud Detection

## Run
- run_id: A2_experiment__D5_synthetic_fraud_data__20260224_193101
- dataset_id: D5_synthetic_fraud_data
- dataset_fingerprint: 8800acddc4
- sigmoid_shift: 4.8
- dataset_path: /mnt/c/Users/dayes/Production ML Systems/fintech-fraud-merchant-risk/data/datasets/D5_synthetic_fraud_data/processed/fraud_dataset.csv

## Decisioning Policy
- fp_investigation_cost: 7.5
- alert_topk_per_day: 200
- fn_loss_multiplier: 1.0
- best_threshold (valid cost): 0.0657

## Evaluation
- valid PR-AUC: 0.3139
- test PR-AUC:  0.2924
- valid recall@topK: 0.3999
- test  recall@topK: 0.3821
- valid alert rate: 33.8173%
- test  alert rate: 34.0328%
