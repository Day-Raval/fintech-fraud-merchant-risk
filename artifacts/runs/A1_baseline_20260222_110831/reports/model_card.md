# Model Card — Fraud Detection (Project 1)

## Overview
Binary classifier to predict transaction fraud probability for investigator prioritization.

## Data
Synthetic-but-realistic transactions generated with:
- customer/card/merchant/transaction schemas
- injected fraud patterns: foreign, risky MCC, velocity, high amount, new tenure, late-night

## Training
- Time-based split (no leakage)
- Model: HistGradientBoosting + isotonic calibration
- Threshold optimized using cost function (false positive investigation cost vs fraud loss)

## Metrics
- Valid PR-AUC: 0.3103
- Test PR-AUC: 0.2881
- Best threshold (valid cost): 0.0606

## Operational Notes
- Alert budget: Top-K per day supported via dashboard and threshold tuning
- Reason codes: rule-based, stable explanations for investigators
