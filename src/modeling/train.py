from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.common.experiment_log import log_experiment
from src.common.logging import get_logger
from src.common.paths import RepoPaths
from src.common.utils import load_yaml, save_json
from src.data.schema import DATASET_FILE
from src.features.preprocessing import FeatureSpec, build_preprocessor
from src.modeling.evaluate import BizConfig, pr_auc, recall_at_topk, find_best_threshold
from src.modeling.merchant_risk import build_merchant_risk_table

logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    train_days: int
    valid_days: int
    test_days: int


def _require_keys(cfg: Dict[str, Any], section: str, keys: list[str]) -> None:
    if section not in cfg:
        raise KeyError(f"Missing '{section}' section in config.yaml")
    missing = [k for k in keys if k not in cfg[section]]
    if missing:
        raise KeyError(f"Missing keys under '{section}' in config.yaml: {missing}")


def dataset_fingerprint(path: Path) -> str:
    """
    Fast, practical dataset id: filename + bytes + mtime.
    Helps readers know if trend comparisons are apples-to-apples.
    """
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:10]


def time_split(df: pd.DataFrame, split: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict time split to mimic production: train on past, validate/test on future."""
    fs = FeatureSpec()
    df = df.sort_values(fs.time_col).copy()

    t0 = pd.to_datetime(df[fs.time_col].min())
    train_end = t0 + timedelta(days=split.train_days)
    valid_end = train_end + timedelta(days=split.valid_days)

    train = df[df[fs.time_col] < train_end]
    valid = df[(df[fs.time_col] >= train_end) & (df[fs.time_col] < valid_end)]
    test = df[df[fs.time_col] >= valid_end]
    return train, valid, test


def build_model_pipeline(preproc: Pipeline, model_cfg: Dict[str, Any]) -> Pipeline:
    """
    Build the sklearn Pipeline: (preprocessor -> classifier)

    Supports:
    - logreg: fast baseline
    - hgbt: strong baseline without external dependencies
    """
    model_type = str(model_cfg.get("type", "hgbt")).lower()

    if model_type == "logreg":
        clf = LogisticRegression(
            max_iter=int(model_cfg.get("max_iter", 200)),
            n_jobs=None,
            class_weight="balanced",
        )
    elif model_type == "hgbt":
        clf = HistGradientBoostingClassifier(
            learning_rate=float(model_cfg.get("learning_rate", 0.08)),
            max_depth=int(model_cfg.get("max_depth", 6)),
            max_iter=int(model_cfg.get("max_iter", 250)),
            random_state=int(model_cfg.get("random_state", 42)),
        )
    else:
        raise ValueError(f"Unknown model_params.type={model_type}. Use 'logreg' or 'hgbt'.")

    # build_preprocessor returns a Pipeline with "preprocessor" step;
    # we extract the transformer to avoid nested Pipelines.
    return Pipeline(steps=[
        ("prep", preproc.named_steps["preprocessor"]),
        ("clf", clf),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    repo = RepoPaths(Path(".").resolve())
    fs = FeatureSpec()

    # Validate required config keys
    _require_keys(cfg, "model", ["time_split_days_train", "time_split_days_valid", "time_split_days_test"])
    _require_keys(cfg, "decisioning", ["fp_investigation_cost", "alert_topk_per_day", "fn_loss_multiplier"])

    # Compute run_id ONCE and reuse everywhere
    base_tag = cfg.get("project", {}).get("experiment_tag", "default_run")
    run_id = f"{base_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load processed dataset
    proc_path = repo.processed_dir / DATASET_FILE
    if not proc_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {proc_path}")

    df = pd.read_csv(proc_path, parse_dates=[fs.time_col])

    # Basic schema checks
    for col in [fs.time_col, fs.target, "amount_usd"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in {proc_path}")

    # Time split
    split = SplitConfig(
        train_days=int(cfg["model"]["time_split_days_train"]),
        valid_days=int(cfg["model"]["time_split_days_valid"]),
        test_days=int(cfg["model"]["time_split_days_test"]),
    )
    train_df, valid_df, test_df = time_split(df, split)

    logger.info(f"Split sizes: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    logger.info(
        f"Fraud rates: train={train_df[fs.target].mean():.3%}, "
        f"valid={valid_df[fs.target].mean():.3%}, "
        f"test={test_df[fs.target].mean():.3%}"
    )

    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        raise ValueError("One of the splits is empty. Adjust time_split_days_* in config.yaml.")

    # Preprocessing
    preproc, num_cols, cat_cols = build_preprocessor(train_df)

    # Model params (optional section)
    model_cfg = cfg.get("model_params", {"type": "hgbt"})
    base_model = build_model_pipeline(preproc, model_cfg)

    X_train = train_df.drop(columns=[fs.target])
    y_train = train_df[fs.target].astype(int).values

    logger.info(f"Fitting base model ({model_cfg.get('type', 'hgbt')})...")
    base_model.fit(X_train, y_train)

    # Calibration on validation
    X_valid = valid_df.drop(columns=[fs.target])
    y_valid = valid_df[fs.target].astype(int).values

    logger.info("Calibrating probabilities with isotonic regression (on validation)...")
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    calibrated.fit(X_valid, y_valid)

    # Predict
    valid_prob = calibrated.predict_proba(X_valid)[:, 1]
    X_test = test_df.drop(columns=[fs.target])
    y_test = test_df[fs.target].astype(int).values
    test_prob = calibrated.predict_proba(X_test)[:, 1]

    # Decisioning (ONLY from decisioning block)
    d = cfg["decisioning"]
    fp_cost = float(d["fp_investigation_cost"])
    topk_per_day = int(d["alert_topk_per_day"])
    fn_mult = float(d["fn_loss_multiplier"])

    biz = BizConfig(
        alert_topk_per_day=topk_per_day,
        fp_investigation_cost=fp_cost,
        fn_fraud_loss_multiplier=fn_mult,
    )

    logger.info(
        "Decisioning params: fp_cost=%.3f | topk_per_day=%d | fn_multiplier=%.3f | run_id=%s",
        fp_cost, topk_per_day, fn_mult, run_id
    )

    # Threshold optimization (must respond to fp_cost/fn_mult)
    best_t, threshold_report = find_best_threshold(
        y_true=y_valid,
        y_prob=valid_prob,
        biz=biz,
        amounts=valid_df["amount_usd"].values,
    )

    # Metrics
    valid_days = max(1, valid_df[fs.time_col].dt.date.nunique())
    test_days = max(1, test_df[fs.time_col].dt.date.nunique())

    valid_recall_topk = recall_at_topk(y_valid, valid_prob, k=topk_per_day * valid_days)
    test_recall_topk = recall_at_topk(y_test, test_prob, k=topk_per_day * test_days)

    # Operational "prediction rate": how many transactions get flagged at best threshold
    valid_alert_rate = float((valid_prob >= best_t).mean())
    test_alert_rate = float((test_prob >= best_t).mean())

    # FP/FN counts at chosen threshold (validation)
    pred_valid = (valid_prob >= best_t).astype(int)
    valid_fp = int(((pred_valid == 1) & (y_valid == 0)).sum())
    valid_fn = int(((pred_valid == 0) & (y_valid == 1)).sum())

    metrics = {
        "valid_pr_auc": float(pr_auc(y_valid, valid_prob)),
        "test_pr_auc": float(pr_auc(y_test, test_prob)),
        "valid_recall_at_topk": float(valid_recall_topk),
        "test_recall_at_topk": float(test_recall_topk),
        "best_threshold_valid_cost": float(best_t),
        "valid_alert_rate": float(valid_alert_rate),
        "test_alert_rate": float(test_alert_rate),
        "threshold_report_valid": threshold_report,
        "decisioning": {
            "fp_investigation_cost": fp_cost,
            "alert_topk_per_day": topk_per_day,
            "fn_loss_multiplier": fn_mult,
        },
        "model_params": model_cfg,
        "run_id": run_id,
        "dataset_path": str(proc_path),
        "dataset_id": dataset_fingerprint(proc_path),
    }

    # --- Save per-run artifacts ---
    run_root = Path("artifacts/runs") / run_id
    model_dir = run_root / "models"
    metrics_dir = run_root / "metrics"
    report_dir = run_root / "reports"

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Merchant risk table (TRAIN only)
    merchant_risk = build_merchant_risk_table(train_df)

    # Bundle for serving
    bundle = {
        "model": calibrated,
        "threshold": float(best_t),
        "feature_spec": {
            "num_cols": list(num_cols),
            "cat_cols": list(cat_cols),
        },
    }

    model_path = model_dir / "fraud_model.joblib"
    merchant_risk_path = model_dir / "merchant_risk.joblib"
    metrics_path = metrics_dir / "metrics.json"
    threshold_report_path = metrics_dir / "threshold_report.json"
    model_card_path = report_dir / "model_card.md"

    joblib.dump(bundle, model_path)
    joblib.dump({"merchant_risk": merchant_risk}, merchant_risk_path)

    save_json(metrics, str(metrics_path))
    save_json({"best_threshold": float(best_t), **threshold_report}, str(threshold_report_path))

    # Model card (short + useful)
    model_card_path.write_text(
        f"""# Model Card — Fraud Detection

## Run
- run_id: {run_id}
- dataset_id: {metrics["dataset_id"]}
- dataset_path: {metrics["dataset_path"]}

## Decisioning Policy
- fp_investigation_cost: {fp_cost}
- alert_topk_per_day: {topk_per_day}
- fn_loss_multiplier: {fn_mult}
- best_threshold (valid cost): {best_t:.4f}

## Evaluation
- valid PR-AUC: {metrics["valid_pr_auc"]:.4f}
- test PR-AUC:  {metrics["test_pr_auc"]:.4f}
- valid recall@topK: {metrics["valid_recall_at_topk"]:.4f}
- test  recall@topK: {metrics["test_recall_at_topk"]:.4f}
- valid alert rate: {metrics["valid_alert_rate"]:.4%}
- test  alert rate: {metrics["test_alert_rate"]:.4%}
""",
        encoding="utf-8",
    )

    # --- Log clean row to experiments.csv (trend-friendly) ---
    log_experiment({
        "run_id": run_id,
        "log_tag": base_tag,
        "dataset_id": metrics["dataset_id"],

        "fp_cost": fp_cost,
        "topk_per_day": topk_per_day,
        "fn_mult": fn_mult,

        "train_fraud_rate": float(train_df[fs.target].mean()),
        "valid_fraud_rate": float(valid_df[fs.target].mean()),
        "test_fraud_rate": float(test_df[fs.target].mean()),

        "valid_pr_auc": float(metrics["valid_pr_auc"]),
        "test_pr_auc": float(metrics["test_pr_auc"]),
        "best_threshold": float(best_t),
        "valid_recall_at_topk": float(valid_recall_topk),
        "test_recall_at_topk": float(test_recall_topk),

        "valid_alert_rate": float(valid_alert_rate),
        "test_alert_rate": float(test_alert_rate),

        "valid_net_cost": float(threshold_report.get("net_cost_lower_is_better", np.nan)),
        "valid_fp": valid_fp,
        "valid_fn": valid_fn,
    })

    logger.info(f"Saved model: {model_path}")
    logger.info(f"Saved merchant risk table: {merchant_risk_path}")
    logger.info(f"Saved metrics: {metrics_path}")
    logger.info(f"Saved threshold report: {threshold_report_path}")
    logger.info(f"Saved model card: {model_card_path}")
    logger.info(f"Saved run artifacts under: {run_root}")
    logger.info(f"Best threshold: {best_t:.4f}")


if __name__ == "__main__":
    main()