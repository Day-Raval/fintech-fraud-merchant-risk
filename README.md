**Goal**: Build a production-style fraud detection system that scores card transactions in real time and helps investigators act efficiently under operational constraints.

I am solving to: Predict the probability that a transaction is fraudulent before authorization completes, using customer/card/merchant/transaction signals.

1. Enforce business constraints:

- Fraud dollars prevented (catch high-$ fraud)

- Investigation cost / false positives (limit unnecessary alerts)

- Alert volume cap (only Top-K alerts/day)

- Low latency (serve predictions fast; track p95)

2. Outputs:

- Transaction-level fraud probability + decision (flag/pass)

- Human-friendly reason codes (why flagged)

- Merchant risk score (aggregated from observed behavior)