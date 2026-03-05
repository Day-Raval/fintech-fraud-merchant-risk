from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.common.run_registry import list_runs, latest_run_id
from src.api.service import ArtifactService

app = FastAPI(
    title="Fintech Fraud + Merchant Risk API",
    version="1.0.0",
)

service = ArtifactService(runs_root=Path("artifacts/runs"))

HTTP_REQUESTS_TOTAL = Counter(
    "fraud_api_http_requests_total",
    "Total number of HTTP requests received by the API.",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "fraud_api_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)

PREDICTION_TOTAL = Counter(
    "fraud_api_predictions_total",
    "Total number of prediction requests.",
    ["result"],
)

PREDICTION_DURATION_SECONDS = Histogram(
    "fraud_api_prediction_duration_seconds",
    "Prediction handler latency in seconds.",
)


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


@app.middleware("http")
async def prometheus_http_metrics(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    path = request.url.path
    status_code = str(response.status_code)
    method = request.method

    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status_code=status_code).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(elapsed)
    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    rid = latest_run_id(Path("artifacts/runs"))
    return {
        "status": "ok",        # required by tests
        "latest_run_id": rid,  # required by tests
        "health": "up",        # optional/back-compat for existing monitoring conventions
    }


@app.get("/runs")
def runs() -> Dict[str, Any]:
    return {"runs": list_runs(Path("artifacts/runs"))}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/load")
def load_run(
    run_id: Optional[str] = Query(
        default=None,
        description="Run ID folder under artifacts/runs. If omitted, loads latest.",
    )
) -> Dict[str, Any]:
    try:
        loaded = service.load(run_id)
        return {"loaded_run_id": loaded.run.run_id, "root": str(loaded.run.root)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    try:
        out = service.predict_one(req.data)
        result = "fraud_alert" if int(out["is_fraud_alert"]) == 1 else "pass"
        PREDICTION_TOTAL.labels(result=result).inc()
        return PredictResponse(**out)
    except FileNotFoundError as e:
        PREDICTION_TOTAL.labels(result="not_loaded").inc()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        PREDICTION_TOTAL.labels(result="error").inc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    finally:
        PREDICTION_DURATION_SECONDS.observe(time.perf_counter() - start)
