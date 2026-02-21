from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass(frozen=True)
class BizConfig:
    alert_topk_per_day: int
    fp_investigation_cost: float
    fn_fraud_loss_multiplier: float


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))


def recall_at_topk(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """
    Recall when only the Top-K highest-risk transactions are investigated.
    """
    k = min(k, len(y_true))
    idx = np.argsort(-y_prob)[:k]
    return float(y_true[idx].sum() / max(1, y_true.sum()))


def cost_metric(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, biz: BizConfig, amounts: np.ndarray) -> Dict[str, float]:
    """
    Simple cost model:
    - False Positive: investigation cost per alert
    - False Negative: fraud loss = amount * multiplier
    - True Positive: prevents fraud loss (counts as savings)
    """
    y_hat = (y_prob >= threshold).astype(int)

    fp = ((y_hat == 1) & (y_true == 0)).sum()
    fn = ((y_hat == 0) & (y_true == 1)).sum()
    tp = ((y_hat == 1) & (y_true == 1)).sum()

    fp_cost = fp * biz.fp_investigation_cost
    fn_cost = float((amounts[(y_hat == 0) & (y_true == 1)] * biz.fn_fraud_loss_multiplier).sum())
    prevented = float((amounts[(y_hat == 1) & (y_true == 1)] * biz.fn_fraud_loss_multiplier).sum())

    net_cost = fp_cost + fn_cost - prevented

    return {
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "fp_cost": float(fp_cost),
        "fn_fraud_loss": float(fn_cost),
        "fraud_prevented": float(prevented),
        "net_cost_lower_is_better": float(net_cost),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, biz: BizConfig, amounts: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Choose threshold minimizing net cost. Evaluates thresholds from PR curve points
    (efficient and relevant).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has length n-1
    candidates = np.unique(np.clip(thresholds, 0.01, 0.99))
    best_t = 0.5
    best_report = None
    best_cost = float("inf")

    for t in candidates:
        rep = cost_metric(y_true, y_prob, float(t), biz, amounts)
        if rep["net_cost_lower_is_better"] < best_cost:
            best_cost = rep["net_cost_lower_is_better"]
            best_t = float(t)
            best_report = rep

    return best_t, (best_report or {})