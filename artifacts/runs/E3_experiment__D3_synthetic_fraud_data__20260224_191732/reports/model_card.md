# Model Card — Fraud Detection

## Run
- run_id: E3_experiment__D3_synthetic_fraud_data__20260224_191732
- dataset_id: D3_synthetic_fraud_data
- dataset_fingerprint: b5a550b6e7
- sigmoid_shift: 4.0
- dataset_path: /mnt/c/Users/dayes/Production ML Systems/fintech-fraud-merchant-risk/data/datasets/D3_synthetic_fraud_data/processed/fraud_dataset.csv

## Decisioning Policy
- fp_investigation_cost: 7.5
- alert_topk_per_day: 200
- fn_loss_multiplier: 1.0
- best_threshold (valid cost): 0.0710

## Evaluation
- valid PR-AUC: 0.3613
- test PR-AUC:  0.3568
- valid recall@topK: 0.3097
- test  recall@topK: 0.3130
- valid alert rate: 62.1607%
- test  alert rate: 62.4122%
