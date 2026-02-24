from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

EXPERIMENTS_CSV = Path("artifacts/metrics/experiments.csv")
GENERATION_CSV = Path("artifacts/metrics/generation.csv")


def log_experiment(row: Dict[str, Any], csv_path: Optional[Path] = None) -> None:
    """
    Append a row to a CSV safely even if different calls provide different keys.
    Expands header when new keys appear and rewrites the file (keeping old rows aligned).
    """
    target = csv_path or EXPERIMENTS_CSV
    target.parent.mkdir(parents=True, exist_ok=True)

    r = dict(row)
    r.setdefault("run_ts", datetime.now(timezone.utc).isoformat())

    if not target.exists():
        with target.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(r.keys()))
            w.writeheader()
            w.writerow(r)
        return

    with target.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing_header = next(reader, [])

    new_header: List[str] = list(existing_header)
    for k in r.keys():
        if k not in new_header:
            new_header.append(k)

    if new_header == existing_header:
        with target.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=existing_header)
            w.writerow({k: r.get(k, "") for k in existing_header})
        return

    df = pd.read_csv(target)
    for k in new_header:
        if k not in df.columns:
            df[k] = ""
    df = df[new_header]

    df = pd.concat([df, pd.DataFrame([{k: r.get(k, "") for k in new_header}])], ignore_index=True)
    df.to_csv(target, index=False)