from __future__ import annotations

# Central place for canonical column names used across modules.
# Keeping schema consistent prevents subtle training/serving mismatches.

CUSTOMERS_FILE = "customers.csv"
MERCHANTS_FILE = "merchants.csv"
CARDS_FILE = "cards.csv"
TRANSACTIONS_FILE = "transactions.csv"

DATASET_FILE = "fraud_dataset.csv"  # flat training table