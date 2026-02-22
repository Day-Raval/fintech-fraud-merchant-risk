from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass(frozen=True)
class RunPaths:
    """Resolved paths for a specific run folder under artifacts/runs/<run_id>/."""
    run_id: str
    root: Path
    model_path: Path
    merchant_risk_path: Path
    metrics_path: Path
    threshold_report_path: Path
    model_card_path: Path


def list_runs(runs_root: Path = Path("artifacts/runs")) -> List[str]:
    """List available run_ids (folder names) sorted by modified time (newest first)."""
    if not runs_root.exists():
        return []
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in run_dirs]


def resolve_run(run_id: str, runs_root: Path = Path("artifacts/runs")) -> RunPaths:
    """Resolve and validate required artifact files for a given run_id."""
    root = runs_root / run_id
    if not root.exists():
        raise FileNotFoundError(f"Run not found: {root}")

    model_path = root / "models" / "fraud_model.joblib"
    merchant_risk_path = root / "models" / "merchant_risk.joblib"
    metrics_path = root / "metrics" / "metrics.json"
    threshold_report_path = root / "metrics" / "threshold_report.json"
    model_card_path = root / "reports" / "model_card.md"

    missing = [p for p in [model_path, merchant_risk_path, metrics_path, threshold_report_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run '{run_id}' is missing required artifacts: {missing}")

    return RunPaths(
        run_id=run_id,
        root=root,
        model_path=model_path,
        merchant_risk_path=merchant_risk_path,
        metrics_path=metrics_path,
        threshold_report_path=threshold_report_path,
        model_card_path=model_card_path,
    )


def latest_run_id(runs_root: Path = Path("artifacts/runs")) -> Optional[str]:
    """Get the newest run_id by modified time, or None if no runs exist."""
    runs = list_runs(runs_root)
    return runs[0] if runs else None