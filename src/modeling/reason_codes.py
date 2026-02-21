from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ReasonCode:
    code: str
    message: str


# Simple mapping for interpretability (extendable).
# These align with common fraud ops language.
REASON_LIBRARY: Dict[str, ReasonCode] = {
    "high_amount_to_limit": ReasonCode("RC_HIGH_AMOUNT", "Transaction amount is high relative to card limit."),
    "velocity_60m": ReasonCode("RC_VELOCITY", "Unusually high transaction velocity in last hour."),
    "foreign_merchant": ReasonCode("RC_FOREIGN", "Merchant is outside the US (higher risk)."),
    "risky_mcc": ReasonCode("RC_RISKY_MCC", "Merchant category is historically high-risk."),
    "late_night": ReasonCode("RC_LATE_NIGHT", "Transaction occurred during late-night hours."),
    "ecommerce": ReasonCode("RC_ECOMMERCE", "Card-not-present / e-commerce transaction."),
    "high_chargeback": ReasonCode("RC_MERCHANT_CB", "Merchant has elevated chargeback history."),
    "new_account": ReasonCode("RC_NEW_ACCOUNT", "New customer tenure (limited history)."),
}


def rule_based_reason_codes(row: Dict[str, float | int | str]) -> List[Dict[str, str]]:
    """
    Deterministic reason codes to guarantee stable explanations in production.
    (Model-driven SHAP is great, but ops teams usually need consistent codes.)
    """
    reasons: List[Tuple[str, float]] = []

    if float(row.get("amount_to_limit_ratio", 0.0)) >= 0.45:
        reasons.append(("high_amount_to_limit", float(row["amount_to_limit_ratio"])))
    if float(row.get("card_txn_count_60m", 0.0)) >= 4:
        reasons.append(("velocity_60m", float(row["card_txn_count_60m"])))
    if int(row.get("is_foreign_merchant", 0)) == 1:
        reasons.append(("foreign_merchant", 1.0))
    if int(row.get("risky_mcc", 0)) == 1:
        reasons.append(("risky_mcc", 1.0))
    if int(row.get("is_ecommerce", 0)) == 1:
        reasons.append(("ecommerce", 1.0))
    if int(row.get("hour", 12)) in {1, 2, 3, 4, 5}:
        reasons.append(("late_night", 1.0))
    if float(row.get("chargeback_rate", 0.0)) >= 0.02:
        reasons.append(("high_chargeback", float(row["chargeback_rate"])))
    if float(row.get("tenure_months", 999)) < 3:
        reasons.append(("new_account", float(row["tenure_months"])))

    # Sort by "strength" descending; keep top 3
    reasons.sort(key=lambda x: x[1], reverse=True)

    out = []
    for key, _score in reasons[:3]:
        rc = REASON_LIBRARY[key]
        out.append({"code": rc.code, "message": rc.message})

    # Always provide something
    if not out:
        out.append({"code": "RC_MODEL", "message": "Model flagged elevated risk based on combined signals."})
    return out