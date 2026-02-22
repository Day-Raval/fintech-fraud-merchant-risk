from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd

from src.common.run_registry import RunPaths, resolve_run, latest_run_id


@dataclass
class LoadedArtifacts:
    """In-memory loaded artifacts for scoring."""
    run: RunPaths
    model: Any                     # sklearn estimator (CalibratedClassifierCV)
    threshold: float
    num_cols: List[str]
    cat_cols: List[str]
    merchant_risk: Optional[pd.DataFrame]  # if available


class ArtifactService:
    """
    Loads a model run from artifacts/runs/<run_id>/... and serves predictions.

    Supports:
    - load specific run_id
    - load latest available run
    """

    def __init__(self, runs_root: Path = Path("artifacts/runs")) -> None:
        self.runs_root = runs_root
        self._loaded: Optional[LoadedArtifacts] = None

    def load(self, run_id: Optional[str] = None) -> LoadedArtifacts:
        """Load artifacts for a specific run_id (or latest if None)."""
        if run_id is None:
            run_id = latest_run_id(self.runs_root)
            if run_id is None:
                raise FileNotFoundError(f"No runs found under {self.runs_root}")

        run = resolve_run(run_id, self.runs_root)

        bundle = joblib.load(run.model_path)
        model = bundle["model"]
        threshold = float(bundle["threshold"])
        feature_spec = bundle.get("feature_spec", {})
        num_cols = list(feature_spec.get("num_cols", []))
        cat_cols = list(feature_spec.get("cat_cols", []))

        mr_obj = joblib.load(run.merchant_risk_path)
        merchant_risk = None
        if isinstance(mr_obj, dict) and "merchant_risk" in mr_obj:
            # Can be a DataFrame or dict-like
            if isinstance(mr_obj["merchant_risk"], pd.DataFrame):
                merchant_risk = mr_obj["merchant_risk"]
            else:
                # Try to coerce dict into DataFrame if possible
                try:
                    merchant_risk = pd.DataFrame(mr_obj["merchant_risk"])
                except Exception:
                    merchant_risk = None

        self._loaded = LoadedArtifacts(
            run=run,
            model=model,
            threshold=threshold,
            num_cols=num_cols,
            cat_cols=cat_cols,
            merchant_risk=merchant_risk,
        )
        return self._loaded

    @property
    def loaded(self) -> LoadedArtifacts:
        """Return loaded artifacts; load latest if not loaded yet."""
        if self._loaded is None:
            return self.load(None)
        return self._loaded

    def predict_one(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score one transaction.

        Input: dict of transaction fields (must include all model features used during training).
        Output: probability + decision + reason codes.
        """
        art = self.loaded
        df = pd.DataFrame([payload])

        # Ensure missing columns exist (set NaN). This avoids KeyError and lets the pipeline handle missing.
        for col in (art.num_cols + art.cat_cols):
            if col not in df.columns:
                df[col] = np.nan

        prob = float(art.model.predict_proba(df)[:, 1][0])
        decision = int(prob >= art.threshold)

        reasons = self._reason_codes(payload, prob, art.threshold, art.merchant_risk)

        return {
            "run_id": art.run.run_id,
            "fraud_probability": prob,
            "threshold": float(art.threshold),
            "is_fraud_alert": decision,
            "reasons": reasons,
        }

    def _reason_codes(
        self,
        x: Dict[str, Any],
        prob: float,
        threshold: float,
        merchant_risk: Optional[pd.DataFrame],
    ) -> List[str]:
        """
        Simple, stable investigator-facing reason codes.
        Uses only fields that commonly exist in your dataset.
        """
        reasons: List[str] = []

        amt = _safe_float(x.get("amount_usd"))
        is_foreign = _safe_int(x.get("is_foreign_merchant"))
        is_ecom = _safe_int(x.get("is_ecommerce"))
        is_night = _safe_int(x.get("is_night"))
        vel_15m = _safe_float(x.get("txn_count_15m"))
        vel_60m = _safe_float(x.get("txn_count_60m"))
        mcc_risky = _safe_int(x.get("is_risky_mcc"))
        tenure_days = _safe_float(x.get("customer_tenure_days"))
        merchant_id = x.get("merchant_id")

        if prob >= threshold and prob >= 0.7:
            reasons.append("High model risk score")

        if amt is not None and amt >= 500:
            reasons.append("High transaction amount")

        if is_foreign == 1:
            reasons.append("Foreign merchant")

        if is_ecom == 1:
            reasons.append("E-commerce transaction")

        if is_night == 1:
            reasons.append("Late-night transaction")

        if mcc_risky == 1:
            reasons.append("High-risk merchant category (MCC)")

        if vel_15m is not None and vel_15m >= 3:
            reasons.append("High short-window velocity (15m)")

        if vel_60m is not None and vel_60m >= 6:
            reasons.append("High medium-window velocity (60m)")

        if tenure_days is not None and tenure_days <= 30:
            reasons.append("New customer tenure")

        # Merchant risk table (if available)
        if merchant_risk is not None and merchant_id is not None:
            # Try common column names
            for mid_col in ["merchant_id", "merchant", "merchant_key"]:
                if mid_col in merchant_risk.columns:
                    row = merchant_risk[merchant_risk[mid_col] == merchant_id]
                    if len(row) > 0:
                        for risk_col in ["fraud_rate", "risk_score", "merchant_fraud_rate"]:
                            if risk_col in row.columns:
                                mr = float(row.iloc[0][risk_col])
                                if mr >= 0.05:
                                    reasons.append("Historically high-risk merchant")
                                break
                    break

        # Keep reasons deterministic and short
        if not reasons and prob >= threshold:
            reasons.append("Model risk above threshold")

        return reasons[:6]


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> int:
    try:
        return int(v) if v is not None else 0
    except Exception:
        return 0