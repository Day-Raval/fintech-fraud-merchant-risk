# Model Card — Fraud Detection

## Run
- run_id: A1_experiment__D4_synthetic_fraud_data__20260224_192749
- dataset_id: D4_synthetic_fraud_data
- dataset_fingerprint: ae2dcda5ba
- sigmoid_shift: 4.8
- dataset_path: /mnt/c/Users/dayes/Production ML Systems/fintech-fraud-merchant-risk/data/datasets/D4_synthetic_fraud_data/processed/fraud_dataset.csv

## Decisioning Policy
- fp_investigation_cost: 7.5
- alert_topk_per_day: 200
- fn_loss_multiplier: 1.0
- best_threshold (valid cost): 0.0536

## Evaluation
- valid PR-AUC: 0.3155
- test PR-AUC:  0.2912
- valid recall@topK: 0.3984
- test  recall@topK: 0.3767
- valid alert rate: 38.9219%
- test  alert rate: 39.0986%
