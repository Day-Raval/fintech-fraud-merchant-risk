from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.common.run_registry import list_runs, latest_run_id
from src.api.service import ArtifactService

app = FastAPI(
    title="Fintech Fraud + Merchant Risk API",
    version="1.0.0",
)

service = ArtifactService(runs_root=Path("artifacts/runs"))


class PredictRequest(BaseModel):
    """
    Flexible payload: accept arbitrary transaction fields.
    The model pipeline expects the feature columns used during training.
    Missing columns are filled with NaN to avoid KeyError.
    """
    data: Dict[str, Any] = Field(..., description="Transaction fields for scoring")


class PredictResponse(BaseModel):
    run_id: str
    fraud_probability: float
    threshold: float
    is_fraud_alert: int
    reasons: List[str]


@app.get("/health")
def health() -> Dict[str, Any]:
    rid = latest_run_id(Path("artifacts/runs"))
    return {"status": "ok", "latest_run_id": rid}


@app.get("/runs")
def runs() -> Dict[str, Any]:
    return {"runs": list_runs(Path("artifacts/runs"))}


@app.post("/load")
def load_run(run_id: Optional[str] = Query(default=None, description="Run ID folder under artifacts/runs. If omitted, loads latest.")) -> Dict[str, Any]:
    try:
        loaded = service.load(run_id)
        return {"loaded_run_id": loaded.run.run_id, "root": str(loaded.run.root)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        out = service.predict_one(req.data)
        return PredictResponse(**out)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")