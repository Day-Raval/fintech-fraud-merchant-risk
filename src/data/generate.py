from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.utils import load_yaml
from src.data.schema import CUSTOMERS_FILE, MERCHANTS_FILE, CARDS_FILE, TRANSACTIONS_FILE
from src.common.experiment_log import log_experiment

logger = get_logger(__name__)

#                                                                                                                     #
# Generates synthetic-but-realistic customers, merchants, cards, transactions; writes CSVs + sample API payload       #
# It reads the config from config.yaml, which controls dataset size, time range, and random seed for reproducibility. #
#                                                                                                                     #


# -------------------------
# Realistic domain constants
# -------------------------
US_STATES = [
    "NY", "NJ", "CT", "MA", "PA", "MD", "VA", "NC", "SC", "GA",
    "FL", "OH", "MI", "IL", "TX", "AZ", "CO", "WA", "OR", "CA"
]

# Merchant Category Codes (MCC) mapped to human-ish category names.
MCC = [
    ("5411", "Grocery Stores"),
    ("5812", "Restaurants"),
    ("5912", "Drug Stores"),
    ("5541", "Service Stations"),
    ("5999", "Misc Retail"),
    ("5732", "Electronics"),
    ("7011", "Hotels"),
    ("4111", "Transit"),
    ("7995", "Gambling"),
    ("4814", "Telecom"),
    ("5944", "Jewelry"),
    ("6011", "ATM/Cash"),
    ("5967", "Direct Marketing"),
    ("4829", "Money Transfer"),
]

CHANNELS = ["chip", "swipe", "contactless", "ecommerce"]
DEVICE_TYPES = ["mobile", "desktop", "pos_terminal"]
ENTRY_MODES = ["card_present", "card_not_present"]
COUNTRIES = ["US", "CA", "GB", "FR", "DE", "MX", "BR", "IN", "NG"]


@dataclass
class GenConfig:
    seed: int
    n_customers: int
    n_merchants: int
    n_cards: int
    n_transactions: int
    start_date: str
    days: int
    output_dir: str


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# def _random_dates(rng: np.random.Generator, start: datetime, days: int, n: int) -> pd.Series:
#     # Uniformly sample seconds across a window to mimic continuous activity.
#     total_seconds = days * 24 * 60 * 60
#     offsets = rng.integers(0, total_seconds, size=n)
#     return pd.to_datetime([start + timedelta(seconds=int(s)) for s in offsets])

def _random_dates(rng: np.random.Generator, start: datetime, days: int, n: int) -> pd.Series:
    # Uniformly sample seconds across a window to mimic continuous activity.
    total_seconds = days * 24 * 60 * 60
    offsets = rng.integers(0, total_seconds, size=n)
    # Return a pandas Series so downstream code can use `.dt`
    return pd.Series([start + timedelta(seconds=int(s)) for s in offsets], name="timestamp").astype("datetime64[ns]")


def _build_customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    customer_id = np.arange(1, n + 1)

    # Ages: realistic adult distribution (skew to 20–45)
    age = np.clip(rng.normal(36, 12, n).round().astype(int), 18, 85)

    # Income: lognormal-ish, capped
    income = np.clip(rng.lognormal(mean=10.5, sigma=0.45, size=n), 25000, 250000).round()

    # Tenure in months (0–120)
    tenure_months = rng.integers(0, 121, size=n)

    home_state = rng.choice(US_STATES, size=n, replace=True)

    # A "baseline risk" latent score influenced by low tenure and low income stability.
    base_risk = (
        0.9 * (tenure_months < 6).astype(float)
        + 0.4 * (age < 23).astype(float)
        + 0.2 * (income < 45000).astype(float)
        + rng.normal(0, 0.35, n)
    )
    base_risk = np.clip(base_risk, -1.5, 2.5)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "income_usd": income,
            "tenure_months": tenure_months,
            "home_state": home_state,
            "customer_base_risk": base_risk,
        }
    )


def _build_merchants(rng: np.random.Generator, n: int) -> pd.DataFrame:
    merchant_id = np.arange(1, n + 1)
    mcc_codes = rng.choice(len(MCC), size=n, replace=True)
    mcc = [MCC[i][0] for i in mcc_codes]
    mcc_desc = [MCC[i][1] for i in mcc_codes]

    country = rng.choice(COUNTRIES, size=n, p=[0.78, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02, 0.01, 0.01])

    # Chargeback_rate: higher for certain categories/countries.
    # Keep realistic small rates (e.g., 0.1%–3% typical, with long tail).
    base_cb = rng.beta(a=2, b=120, size=n)  # mean ~1.6%
    risky_mcc = np.isin(mcc, ["7995", "5967", "4829", "6011", "5944"])
    base_cb += risky_mcc * rng.uniform(0.005, 0.02, size=n)
    base_cb += (country != "US") * rng.uniform(0.001, 0.01, size=n)
    chargeback_rate = np.clip(base_cb, 0.0005, 0.06)

    # Merchant size: drives transaction volume and typical ticket size
    merchant_size = rng.choice(["small", "mid", "enterprise"], size=n, p=[0.62, 0.30, 0.08])

    return pd.DataFrame(
        {
            "merchant_id": merchant_id,
            "mcc": mcc,
            "mcc_desc": mcc_desc,
            "merchant_country": country,
            "chargeback_rate": chargeback_rate,
            "merchant_size": merchant_size,
        }
    )


def _build_cards(rng: np.random.Generator, n_cards: int, n_customers: int) -> pd.DataFrame:
    card_id = np.arange(1, n_cards + 1)
    customer_id = rng.integers(1, n_customers + 1, size=n_cards)

    card_type = rng.choice(["debit", "credit"], size=n_cards, p=[0.35, 0.65])
    network = rng.choice(["visa", "mastercard", "amex", "discover"], size=n_cards, p=[0.52, 0.35, 0.08, 0.05])

    # Credit limits: only for credit; debit uses a proxy daily limit.
    credit_limit = np.where(
        card_type == "credit",
        np.clip(rng.lognormal(mean=8.8, sigma=0.5, size=n_cards), 500, 30000),
        np.clip(rng.lognormal(mean=7.6, sigma=0.35, size=n_cards), 200, 5000),
    ).round()

    return pd.DataFrame(
        {
            "card_id": card_id,
            "customer_id": customer_id,
            "card_type": card_type,
            "network": network,
            "card_limit_usd": credit_limit,
        }
    )


def _amount_by_mcc(rng: np.random.Generator, mcc: str, n: int) -> np.ndarray:
    """
    Generate realistic transaction amounts by category using lognormal-ish distributions.
    Values are clipped to sensible bounds to avoid absurd amounts.
    """
    # mean/sigma chosen to roughly match typical ticket sizes
    params = {
        "5411": (3.1, 0.55, 5, 450),     # groceries
        "5812": (3.0, 0.65, 5, 350),     # restaurants
        "5912": (2.8, 0.55, 3, 220),     # pharmacies
        "5541": (3.2, 0.35, 5, 180),     # gas
        "5732": (4.4, 0.55, 20, 3000),   # electronics
        "7011": (4.6, 0.55, 30, 5000),   # hotels
        "4111": (2.4, 0.55, 2, 120),     # transit
        "7995": (4.3, 0.70, 10, 4000),   # gambling
        "5944": (4.2, 0.60, 20, 6000),   # jewelry
        "6011": (4.6, 0.50, 20, 2000),   # ATM cash-like
        "5967": (4.1, 0.75, 10, 5000),   # direct marketing
        "4829": (4.5, 0.65, 10, 7000),   # money transfer
        "4814": (3.0, 0.50, 5, 300),     # telecom
        "5999": (3.4, 0.70, 5, 900),     # misc retail
    }
    mu, sigma, lo, hi = params.get(mcc, (3.3, 0.7, 5, 1000))
    amt = rng.lognormal(mean=mu, sigma=sigma, size=n)
    return np.clip(amt, lo, hi).round(2)


def _simulate_transactions(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    cards: pd.DataFrame,
    merchants: pd.DataFrame,
    start: datetime,
    days: int,
    n_txn: int,
    shift: float,
) -> pd.DataFrame:
    """
    Creates realistic transaction streams with:
    - time-of-day patterns
    - category-dependent amounts
    - customer/card/merchant signals
    - injected fraud using plausible patterns (velocity bursts, geo jumps, risky MCC, etc.)
    """
    # Pick cards & merchants for each transaction
    card_ids = rng.integers(1, len(cards) + 1, size=n_txn)
    merchant_ids = rng.integers(1, len(merchants) + 1, size=n_txn)

    ts = _random_dates(rng, start, days, n_txn)
    hour = ts.dt.hour.values

    # time-of-day bias: more transactions daytime/evening than late night
    # We'll resample some timestamps to reflect that shape.
    # This step is intentionally lightweight (keeps run-time reasonable).
    night_mask = (hour >= 1) & (hour <= 5)
    flip = rng.random(n_txn) < 0.60
    ts.loc[night_mask & flip] = ts.loc[night_mask & flip] + pd.to_timedelta(rng.integers(6, 14, size=(night_mask & flip).sum()), unit="h")

    channel = rng.choice(CHANNELS, size=n_txn, p=[0.35, 0.30, 0.15, 0.20])
    device_type = rng.choice(DEVICE_TYPES, size=n_txn, p=[0.42, 0.20, 0.38])
    entry_mode = np.where(channel == "ecommerce", "card_not_present", "card_present")

    # Join merchant attributes for amount generation
    m = merchants.set_index("merchant_id").loc[merchant_ids]
    amounts = np.zeros(n_txn, dtype=float)
    for code in pd.unique(m["mcc"]):
        idx = (m["mcc"].values == code)
        amounts[idx] = _amount_by_mcc(rng, code, idx.sum())

    # Customer/card joins
    c_map = cards.set_index("card_id")[["customer_id", "card_limit_usd", "card_type", "network"]]
    card_join = c_map.loc[card_ids].reset_index(drop=True)
    cust_map = customers.set_index("customer_id")[["home_state", "customer_base_risk", "tenure_months", "income_usd", "age"]]
    cust_join = cust_map.loc[card_join["customer_id"].values].reset_index(drop=True)

    # Location signals: state-based for US transactions; foreign for non-US merchants.
    # For simplicity: transaction_state usually matches home_state unless travel.
    travel_prob = 0.06
    txn_state = cust_join["home_state"].values.copy()
    travel_mask = rng.random(n_txn) < travel_prob
    txn_state[travel_mask] = rng.choice(US_STATES, size=travel_mask.sum(), replace=True)

    merchant_country = m["merchant_country"].values
    is_foreign = merchant_country != "US"

    # Fraud injection logic: combine risk factors into a probability, then sample label.
    # These are common signals in real fraud models (velocity, new device, risky MCC, foreign, high amount).
    risky_mcc = np.isin(m["mcc"].values, ["7995", "5967", "4829", "6011", "5944"])
    high_amount = amounts > np.quantile(amounts, 0.97)
    late_night = pd.Series(ts.dt.hour).between(1, 5).values
    new_account = cust_join["tenure_months"].values < 3
    ecommerce = channel == "ecommerce"

    # "Geo jump": customer transacting in non-home state + foreign merchant increases suspicion
    geo_jump = (txn_state != cust_join["home_state"].values) & is_foreign

    # Merchant chargeback rate is a strong prior
    cb_rate = m["chargeback_rate"].values

    # Latent score -> probability via sigmoid (yields ~2–6% fraud depending on knobs)
    score = (
        1.2 * cust_join["customer_base_risk"].values
        + 1.0 * risky_mcc.astype(float)
        + 0.9 * is_foreign.astype(float)
        + 0.9 * geo_jump.astype(float)
        + 0.7 * ecommerce.astype(float)
        + 0.7 * late_night.astype(float)
        + 1.1 * high_amount.astype(float)
        + 14.0 * cb_rate  # small cb_rate but meaningful
        + 0.8 * new_account.astype(float)
        + rng.normal(0, 0.8, n_txn)
    )

    # fraud_prob = _sigmoid(score - 3.1)  # shift controls base rate
    # shift = float(cfg["data"].get("sigmoid_shift", 3.1))
    fraud_prob = _sigmoid(score - shift)

    is_fraud = (rng.random(n_txn) < fraud_prob).astype(int)

    from src.common.experiment_log import log_experiment

    # Add a "velocity" feature by injecting bursts for a subset of fraudulent transactions:
    # We'll mark some fraud txns as part of a burst and later compute per-card velocity in build_dataset.py.
    burst_flag = np.zeros(n_txn, dtype=int)
    fraud_idx = np.where(is_fraud == 1)[0]
    if len(fraud_idx) > 0:
        chosen = rng.choice(fraud_idx, size=min(2500, len(fraud_idx)), replace=False)
        burst_flag[chosen] = (rng.random(len(chosen)) < 0.55).astype(int)

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, n_txn + 1),
            "timestamp": ts.sort_values().values,  # enforce time ordering globally
            "card_id": card_ids,
            "merchant_id": merchant_ids,
            "amount_usd": amounts,
            "channel": channel,
            "device_type": device_type,
            "entry_mode": entry_mode,
            "txn_state": txn_state,
            "merchant_country": merchant_country,
            "is_fraud": is_fraud,
            "burst_flag": burst_flag,
        }
    ).sort_values("timestamp", ignore_index=True)

    return df


def _write_sample_requests(output_dir: Path, transactions: pd.DataFrame) -> None:
    """
    Create a sample API request JSON from a real row so users can immediately test /predict.
    """
    sample_dir = output_dir / "sample_requests"
    sample_dir.mkdir(parents=True, exist_ok=True)

    row = transactions.sample(1, random_state=1).iloc[0].to_dict()
    # API schema expects a subset; keep only transaction fields.
    payload = {
        "transaction_id": int(row["transaction_id"]),
        "timestamp": pd.to_datetime(row["timestamp"]).isoformat(),
        "card_id": int(row["card_id"]),
        "merchant_id": int(row["merchant_id"]),
        "amount_usd": float(row["amount_usd"]),
        "channel": str(row["channel"]),
        "device_type": str(row["device_type"]),
        "entry_mode": str(row["entry_mode"]),
        "txn_state": str(row["txn_state"]),
        "merchant_country": str(row["merchant_country"]),
    }

    (sample_dir / "transaction_001.json").write_text(
        pd.Series(payload).to_json(), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    g = GenConfig(
        seed=int(cfg["project"]["seed"]),
        n_customers=int(cfg["data"]["n_customers"]),
        n_merchants=int(cfg["data"]["n_merchants"]),
        n_cards=int(cfg["data"]["n_cards"]),
        n_transactions=int(cfg["data"]["n_transactions"]),
        start_date=str(cfg["data"]["start_date"]),
        days=int(cfg["data"]["days"]),
        output_dir=str(cfg["data"]["output_dir"]),
    )

    rng = np.random.default_rng(g.seed)
    root = Path(".").resolve()
    out = root / g.output_dir / "raw"
    out.mkdir(parents=True, exist_ok=True)

    start = datetime.fromisoformat(g.start_date)

    logger.info("Generating customers...")
    customers = _build_customers(rng, g.n_customers)

    logger.info("Generating merchants...")
    merchants = _build_merchants(rng, g.n_merchants)

    logger.info("Generating cards...")
    cards = _build_cards(rng, g.n_cards, g.n_customers)

    logger.info("Generating transactions (this can take a moment for large N)...")

    shift = float(cfg["data"].get("sigmoid_shift", 3.1))
    txns = _simulate_transactions(rng, customers, cards, merchants, start, g.days, g.n_transactions, shift)

    # Save
    customers.to_csv(out / CUSTOMERS_FILE, index=False)
    merchants.to_csv(out / MERCHANTS_FILE, index=False)
    cards.to_csv(out / CARDS_FILE, index=False)
    txns.to_csv(out / TRANSACTIONS_FILE, index=False)

    # Sample API request JSON
    _write_sample_requests(root / g.output_dir, txns)

    fraud_rate = txns["is_fraud"].mean()
    logger.info(f"Done. Rows: customers={len(customers)}, merchants={len(merchants)}, cards={len(cards)}, txns={len(txns)}")
    logger.info(f"Fraud rate: {fraud_rate:.3%} (realistic imbalanced classification)")

    log_experiment({
    "stage": "generate",
    "sigmoid_shift": float(cfg["data"].get("sigmoid_shift", 3.1)),
    "n_transactions": int(cfg["data"]["n_transactions"]),
    "fraud_rate": float(fraud_rate),
    })


if __name__ == "__main__":
    main()