from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

EXPERIMENTS_CSV = Path("artifacts/metrics/experiments.csv")


def log_experiment(row: Dict[str, Any]) -> None:
    """
    Append a row to experiments.csv safely even if different calls provide different keys.
    Expands header when new keys appear and rewrites the file (keeping old rows aligned).
    """
    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    r = dict(row)
    r.setdefault("run_ts", datetime.now(timezone.utc).isoformat())

    if not EXPERIMENTS_CSV.exists():
        with EXPERIMENTS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(r.keys()))
            w.writeheader()
            w.writerow(r)
        return

    with EXPERIMENTS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing_header = next(reader, [])

    new_header: List[str] = list(existing_header)
    for k in r.keys():
        if k not in new_header:
            new_header.append(k)

    if new_header == existing_header:
        with EXPERIMENTS_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=existing_header)
            w.writerow({k: r.get(k, "") for k in existing_header})
        return

    df = pd.read_csv(EXPERIMENTS_CSV)
    for k in new_header:
        if k not in df.columns:
            df[k] = ""
    df = df[new_header]

    df = pd.concat([df, pd.DataFrame([{k: r.get(k, "") for k in new_header}])], ignore_index=True)
    df.to_csv(EXPERIMENTS_CSV, index=False)