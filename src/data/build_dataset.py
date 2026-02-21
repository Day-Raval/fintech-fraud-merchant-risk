from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.utils import load_yaml
from src.data.schema import (
    CUSTOMERS_FILE,
    MERCHANTS_FILE,
    CARDS_FILE,
    TRANSACTIONS_FILE,
    DATASET_FILE,
)

logger = get_logger(__name__)

#                                                                                                                                   #
# This script builds the final flat dataset for modeling by merging raw CSVs and computing features.                                #
# The key part is computing velocity-like features using only historical transactions, which mimics real fraud feature engineering. #
#                                                                                                                                   #

def _compute_velocity_features(txns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-card velocity-like features using only historical transactions.
    This mimics real fraud feature engineering (counts/sums in last X minutes/hours).
    """
    txns = txns.sort_values("timestamp").copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])

    # Rolling windows by card_id. We use time-based rolling via groupby + rolling on datetime index.
    # Approach: for each card, set timestamp as index and compute rolling counts.
    features = []
    for card_id, g in txns.groupby("card_id", sort=False):
        g = g.sort_values("timestamp").set_index("timestamp")

        # number of txns in last 15 minutes / 1 hour
        v15 = g["transaction_id"].rolling("15min").count().shift(1).fillna(0)
        v60 = g["transaction_id"].rolling("60min").count().shift(1).fillna(0)

        # sum amounts in last 1 hour
        s60 = g["amount_usd"].rolling("60min").sum().shift(1).fillna(0)

        # days since last txn
        last_ts = g.index.to_series().shift(1)
        delta = (g.index.to_series() - last_ts).dt.total_seconds().fillna(np.inf)
        mins_since_last = np.where(np.isfinite(delta), delta / 60.0, 99999.0)

        out = pd.DataFrame(
            {
                "transaction_id": g["transaction_id"].values,
                "card_txn_count_15m": v15.values,
                "card_txn_count_60m": v60.values,
                "card_amt_sum_60m": s60.values,
                "mins_since_last_txn": mins_since_last,
            }
        )
        features.append(out)

    return pd.concat(features, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(".").resolve()
    raw_dir = root / cfg["data"]["output_dir"] / "raw"
    proc_dir = root / cfg["data"]["output_dir"] / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(raw_dir / CUSTOMERS_FILE)
    merchants = pd.read_csv(raw_dir / MERCHANTS_FILE)
    cards = pd.read_csv(raw_dir / CARDS_FILE)
    txns = pd.read_csv(raw_dir / TRANSACTIONS_FILE, parse_dates=["timestamp"])

    logger.info("Computing velocity features...")
    vel = _compute_velocity_features(txns)

    logger.info("Building flat dataset...")
    df = (
        txns.merge(cards, on="card_id", how="left")
            .merge(customers, on="customer_id", how="left")
            .merge(merchants, on="merchant_id", how="left")
            .merge(vel, on="transaction_id", how="left")
    )

    # Additional derived features (kept simple but realistic)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_foreign_merchant"] = (df["merchant_country"] != "US").astype(int)
    df["is_ecommerce"] = (df["channel"] == "ecommerce").astype(int)
    df["risky_mcc"] = df["mcc"].isin(["7995", "5967", "4829", "6011", "5944"]).astype(int)

    # Amount vs card limit ratio (proxy for "unusual spend")
    df["amount_to_limit_ratio"] = (df["amount_usd"] / (df["card_limit_usd"] + 1e-6)).clip(0, 5)

    # Persist
    df.to_csv(proc_dir / DATASET_FILE, index=False)
    logger.info(f"Saved dataset: {proc_dir / DATASET_FILE} | rows={len(df)} cols={df.shape[1]}")


if __name__ == "__main__":
    main()