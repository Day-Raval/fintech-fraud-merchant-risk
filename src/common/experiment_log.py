from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd


def log_experiment(row: Dict[str, Any], path: str = "artifacts/metrics/experiments.csv") -> None:
    """
    Appends a single experiment row to a CSV file.
    Creates the file if it doesn't exist.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    row = {"run_ts": datetime.now().isoformat(timespec="seconds"), **row}
    df_new = pd.DataFrame([row])

    if out.exists():
        df_old = pd.read_csv(out)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(out, index=False)