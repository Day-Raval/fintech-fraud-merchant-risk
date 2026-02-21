from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

## This module defines the RepoPaths dataclass, which provides properties for commonly used directories in the project.

@dataclass(frozen=True)
class RepoPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sample_requests_dir(self) -> Path:
        return self.data_dir / "sample_requests"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"