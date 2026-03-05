from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "latest_run_id" in payload


def test_metrics_endpoint_exposes_prometheus_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "fraud_api_http_requests_total" in body
    assert "fraud_api_http_request_duration_seconds" in body
