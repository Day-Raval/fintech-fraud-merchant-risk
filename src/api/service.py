from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import joblib
import numpy as np
import pandas as pd

from src.common.run_registry import RunPaths, resolve_run, latest_run_id

@dataclass
class LoadedArtifacts:
    """In-memory loaded artifacts for scoring."""
    run: RunPaths
    model: Any
    threshold: float
    num_cols: List[str]
    cat_cols: List[str]
    merchant_risk: Optional[pd.DataFrame]


class ArtifactService:
    """
    Loads a model run from artifacts/runs/<run_id>/... and serves predictions.
    """

    def __init__(self, runs_root: Path = Path("artifacts/runs")) -> None:
        self.runs_root = runs_root
        self._loaded: Optional[LoadedArtifacts] = None

    def load(self, run_id: Optional[str] = None) -> LoadedArtifacts:
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

        merchant_risk = None
        try:
            mr_obj = joblib.load(run.merchant_risk_path)
            if isinstance(mr_obj, dict) and "merchant_risk" in mr_obj:
                if isinstance(mr_obj["merchant_risk"], pd.DataFrame):
                    merchant_risk = mr_obj["merchant_risk"]
                else:
                    try:
                        merchant_risk = pd.DataFrame(mr_obj["merchant_risk"])
                    except Exception:
                        merchant_risk = None
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
        if self._loaded is None:
            return self.load(None)
        return self._loaded

    def predict_one(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        art = self.loaded
        df = pd.DataFrame([payload])

        # Ensure missing columns exist (NaN). Avoids KeyError.
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
        Investigator-facing reason codes aligned with your CURRENT feature schema.

        Supports both:
        - raw payloads (timestamp/channel/mcc/merchant_country/etc.)
        - engineered-feature payloads (is_foreign_merchant, card_txn_count_15m, etc.)
        """
        reasons: List[str] = []

        amt = _safe_float(x.get("amount_usd"))
        merchant_country = (x.get("merchant_country") or "").strip()
        channel = (x.get("channel") or "").strip()
        mcc = str(x.get("mcc") or "").strip()

        # engineered fields (preferred if present)
        is_foreign = _safe_int(x.get("is_foreign_merchant"))
        is_ecom = _safe_int(x.get("is_ecommerce"))
        risky_mcc = _safe_int(x.get("risky_mcc"))
        hour = _safe_int(x.get("hour"))

        v15 = _safe_float(x.get("card_txn_count_15m"))
        v60 = _safe_float(x.get("card_txn_count_60m"))
        s60 = _safe_float(x.get("card_amt_sum_60m"))
        tenure_months = _safe_float(x.get("tenure_months"))
        ratio = _safe_float(x.get("amount_to_limit_ratio"))

        merchant_id = x.get("merchant_id")

        # --- fallbacks if engineered fields missing ---
        if is_foreign == 0 and merchant_country:
            is_foreign = 1 if merchant_country.upper() != "US" else 0

        if is_ecom == 0 and channel:
            is_ecom = 1 if channel.lower() == "ecommerce" else 0

        if risky_mcc == 0 and mcc:
            risky_mcc = 1 if mcc in {"7995", "5967", "4829", "6011", "5944"} else 0

        # derive hour from timestamp if needed
        if hour == 0 and x.get("timestamp") is not None:
            try:
                ts = pd.to_datetime(x.get("timestamp"), errors="coerce")
                if pd.notna(ts):
                    hour = int(ts.hour)
            except Exception:
                pass

        is_night = 1 if hour in {1, 2, 3, 4, 5} else 0

        # --- reason rules ---
        if prob >= threshold and prob >= 0.70:
            reasons.append("High model risk score")

        if amt is not None and amt >= 500:
            reasons.append("High transaction amount")

        if ratio is not None and ratio >= 0.35:
            reasons.append("High amount-to-limit ratio")

        if is_foreign == 1:
            reasons.append("Foreign merchant")

        if is_ecom == 1:
            reasons.append("E-commerce transaction")

        if is_night == 1:
            reasons.append("Late-night transaction")

        if risky_mcc == 1:
            reasons.append("High-risk merchant category (MCC)")

        if v15 is not None and v15 >= 3:
            reasons.append("High short-window velocity (15m)")

        if v60 is not None and v60 >= 6:
            reasons.append("High medium-window velocity (60m)")

        if s60 is not None and s60 >= 1200:
            reasons.append("High spend in last hour")

        if tenure_months is not None and tenure_months <= 3:
            reasons.append("New customer tenure")

        # Merchant risk table (if available)
        if merchant_risk is not None and merchant_id is not None:
            for mid_col in ["merchant_id", "merchant", "merchant_key"]:
                if mid_col in merchant_risk.columns:
                    try:
                        row = merchant_risk[merchant_risk[mid_col] == merchant_id]
                        if len(row) > 0:
                            for risk_col in ["fraud_rate", "risk_score", "merchant_fraud_rate"]:
                                if risk_col in row.columns:
                                    mr = float(row.iloc[0][risk_col])
                                    if mr >= 0.05:
                                        reasons.append("Historically high-risk merchant")
                                    break
                    except Exception:
                        pass
                    break

        if not reasons and prob >= threshold:
            reasons.append("Model risk above threshold")

        return reasons[:6]


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> int:
    try:
        if v is None or v == "":
            return 0
        return int(float(v))
    except Exception:
        return 0