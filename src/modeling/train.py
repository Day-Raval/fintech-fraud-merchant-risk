from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.common.logging import get_logger
from src.common.utils import load_yaml, save_json
from src.common.paths import RepoPaths
from src.data.schema import DATASET_FILE
from src.features.preprocessing import FeatureSpec, build_preprocessor
from src.modeling.evaluate import BizConfig, pr_auc, recall_at_topk, find_best_threshold
from src.modeling.merchant_risk import build_merchant_risk_table
from src.common.experiment_log import log_experiment

logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    train_days: int
    valid_days: int
    test_days: int


def time_split(df: pd.DataFrame, split: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict time split to mimic production: train on the past, validate/test on future.
    """
    df = df.sort_values(FeatureSpec().time_col).copy()
    t0 = pd.to_datetime(df[FeatureSpec().time_col].min())
    train_end = t0 + timedelta(days=split.train_days)
    valid_end = train_end + timedelta(days=split.valid_days)

    train = df[df[FeatureSpec().time_col] < train_end]
    valid = df[(df[FeatureSpec().time_col] >= train_end) & (df[FeatureSpec().time_col] < valid_end)]
    test = df[df[FeatureSpec().time_col] >= valid_end]
    return train, valid, test


def build_model_pipeline(preproc: Pipeline, model_type: str) -> Pipeline:
    """
    Two solid baselines:
    - logreg: strong, fast, interpretable
    - hgbt: gradient boosting baseline without external deps (often stronger)
    """
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=200, n_jobs=None, class_weight="balanced")
    elif model_type == "hgbt":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            max_iter=250,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type={model_type}")

    return Pipeline(
        steps=[
            ("prep", preproc.named_steps["preprocessor"]),
            ("clf", clf),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    repo = RepoPaths(Path(".").resolve())

    proc_dir = repo.processed_dir
    df = pd.read_csv(proc_dir / DATASET_FILE, parse_dates=[FeatureSpec().time_col])

    # Splits
    split = SplitConfig(
        train_days=int(cfg["model"]["time_split_days_train"]),
        valid_days=int(cfg["model"]["time_split_days_valid"]),
        test_days=int(cfg["model"]["time_split_days_test"]),
    )
    train_df, valid_df, test_df = time_split(df, split)

    logger.info(f"Split sizes: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    logger.info(f"Fraud rates: train={train_df.is_fraud.mean():.3%}, valid={valid_df.is_fraud.mean():.3%}, test={test_df.is_fraud.mean():.3%}")

    # Preprocessing
    preproc, num_cols, cat_cols = build_preprocessor(train_df)

    # Train strong model + calibrate
    base_model = build_model_pipeline(preproc, model_type="hgbt")

    X_train = train_df.drop(columns=[FeatureSpec().target])
    y_train = train_df[FeatureSpec().target].astype(int).values

    logger.info("Fitting base model (HistGradientBoosting)...")
    base_model.fit(X_train, y_train)

    # Calibration improves probability quality (important for thresholding/cost)
    logger.info("Calibrating probabilities with isotonic regression (on validation)...")
    X_valid = valid_df.drop(columns=[FeatureSpec().target])
    y_valid = valid_df[FeatureSpec().target].astype(int).values

    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    calibrated.fit(X_valid, y_valid)

    # Evaluate
    valid_prob = calibrated.predict_proba(X_valid)[:, 1]
    test_X = test_df.drop(columns=[FeatureSpec().target])
    test_y = test_df[FeatureSpec().target].astype(int).values
    test_prob = calibrated.predict_proba(test_X)[:, 1]

    biz = BizConfig(
        alert_topk_per_day=int(cfg["model"]["alert_topk_per_day"]),
        fp_investigation_cost=float(cfg["model"]["fp_investigation_cost"]),
        fn_fraud_loss_multiplier=float(cfg["model"]["fn_fraud_loss_multiplier"]),
    )

    # Best threshold via cost on validation
    best_t, threshold_report = find_best_threshold(
        y_true=y_valid,
        y_prob=valid_prob,
        biz=biz,
        amounts=valid_df["amount_usd"].values,
    )

    # Metrics
    metrics = {
        "valid_pr_auc": pr_auc(y_valid, valid_prob),
        "test_pr_auc": pr_auc(test_y, test_prob),
        "valid_recall_at_topk": recall_at_topk(y_valid, valid_prob, k=biz.alert_topk_per_day * max(1, valid_df["timestamp"].dt.date.nunique())),
        "test_recall_at_topk": recall_at_topk(test_y, test_prob, k=biz.alert_topk_per_day * max(1, test_df["timestamp"].dt.date.nunique())),
        "best_threshold_valid_cost": best_t,
        "threshold_report_valid": threshold_report,
    }

    log_experiment({
        "stage": "train",
        "sigmoid_shift": float(cfg["data"].get("sigmoid_shift", 3.1)),
        "train_fraud_rate": float(train_df.is_fraud.mean()),
        "valid_fraud_rate": float(valid_df.is_fraud.mean()),
        "test_fraud_rate": float(test_df.is_fraud.mean()),
        "valid_pr_auc": float(metrics["valid_pr_auc"]),
        "test_pr_auc": float(metrics["test_pr_auc"]),
        "best_threshold": float(best_t),
        "valid_recall_at_topk": float(metrics["valid_recall_at_topk"]),
        "test_recall_at_topk": float(metrics["test_recall_at_topk"]),
        "valid_net_cost": float(metrics["threshold_report_valid"]["net_cost_lower_is_better"]),
    })

    # Merchant risk table (built from TRAIN only to avoid leakage)
    merchant_risk = build_merchant_risk_table(train_df)
    repo.models_dir.mkdir(parents=True, exist_ok=True)
    repo.metrics_dir.mkdir(parents=True, exist_ok=True)
    repo.reports_dir.mkdir(parents=True, exist_ok=True)

    # Save bundle: model + threshold + column spec for reason codes
    bundle = {
        "model": calibrated,
        "threshold": best_t,
        "feature_spec": {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        },
    }

    model_path = Path(cfg["artifacts"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    merchant_risk_path = Path(cfg["artifacts"]["merchant_risk_path"])
    merchant_risk_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"merchant_risk": merchant_risk}, merchant_risk_path)

    # Write metrics
    save_json(metrics, cfg["artifacts"]["metrics_path"])
    save_json({"best_threshold": best_t, **threshold_report}, cfg["artifacts"]["threshold_report_path"])

    # Model card (simple, useful)
    model_card = Path(cfg["artifacts"]["model_card_path"])
    model_card.parent.mkdir(parents=True, exist_ok=True)
    model_card.write_text(
        f"""# Model Card — Fraud Detection (Project 1)

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
- Valid PR-AUC: {metrics["valid_pr_auc"]:.4f}
- Test PR-AUC: {metrics["test_pr_auc"]:.4f}
- Best threshold (valid cost): {best_t:.4f}

## Operational Notes
- Alert budget: Top-K per day supported via dashboard and threshold tuning
- Reason codes: rule-based, stable explanations for investigators
""",
        encoding="utf-8",
    )

    logger.info(f"Saved model: {model_path}")
    logger.info(f"Saved merchant risk table: {merchant_risk_path}")
    logger.info(f"Saved metrics: {cfg['artifacts']['metrics_path']}")
    logger.info(f"Best threshold: {best_t:.4f}")


if __name__ == "__main__":
    main()