from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class MerchantRiskScore:
    merchant_id: int
    txn_count: int
    fraud_rate: float
    avg_amount: float
    chargeback_rate: float
    risk_score: float


def build_merchant_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merchant risk is computed as a stable aggregate:
    - observed fraud rate (smoothed)
    - chargeback rate prior
    - higher avg amount increases impact
    Produces a table usable by dashboard + ops.
    """
    g = df.groupby("merchant_id", as_index=False).agg(
        txn_count=("transaction_id", "count"),
        fraud_rate=("is_fraud", "mean"),
        avg_amount=("amount_usd", "mean"),
        chargeback_rate=("chargeback_rate", "mean"),
    )

    # Empirical Bayes smoothing: avoid tiny merchants looking extreme
    prior = df["is_fraud"].mean()
    k = 50  # smoothing strength
    g["fraud_rate_smoothed"] = (g["fraud_rate"] * g["txn_count"] + prior * k) / (g["txn_count"] + k)

    # Risk score: interpretable weighted combination
    g["risk_score"] = (
        0.65 * g["fraud_rate_smoothed"]
        + 0.30 * g["chargeback_rate"]
        + 0.05 * (g["avg_amount"] / (df["amount_usd"].mean() + 1e-6))
    )

    return g.sort_values("risk_score", ascending=False).reset_index(drop=True)