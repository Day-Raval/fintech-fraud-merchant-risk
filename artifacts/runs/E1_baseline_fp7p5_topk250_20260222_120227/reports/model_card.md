# Model Card — Fraud Detection (Project 1)

## Overview
Binary classifier to predict transaction fraud probability for investigator prioritization.

## Data
Synthetic-but-realistic transactions with injected fraud patterns:
- foreign merchant / risky MCC
- velocity bursts
- high amount / amount-to-limit ratio
- new customer tenure
- late-night + e-commerce channel effects

## Training
- Time-based split (no leakage)
- Model: hgbt
- Calibration: isotonic regression (validation set)
- Threshold: cost-optimized on validation using:
  - FP cost = 7.5
  - FN loss multiplier = 1.0
  - Alert budget (TopK/day) = 200

## Metrics
- Valid PR-AUC: 0.3103
- Test PR-AUC: 0.2881
- Valid Recall@TopK: 0.3921
- Test Recall@TopK: 0.3812
- Best threshold (valid cost): 0.0606

## Operational Notes
- Reason codes: rule-based and stable for investigators
- Drift monitoring: PSI module supported separately
