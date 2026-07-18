"""Tests for the FastAPI app wiring -- /health and /ingest.

/ingest is tested by monkeypatching ingest_all itself (the actual ingestion
logic already has thorough coverage in test_rag_ingestion.py); this file
only proves the endpoint calls it correctly and shapes the response.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import customer_support_agent.api.ingestion as ingestion_module
from customer_support_agent.api.deps import get_db
from customer_support_agent.integrations.rag import IngestResult
from main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_endpoint_calls_ingest_all_and_shapes_response(monkeypatch):
    fake_results = [
        IngestResult(
            source_file="knowledge_base/motor_comprehensive_policy_v3_2.md",
            product_type="motor_comprehensive",
            version="v3.2",
            chunks_ingested=24,
            chunk_ids=["motor_comprehensive::v3.2::OD-1.0"],
            deleted_stale_count=0,
        ),
        IngestResult(
            source_file="knowledge_base/motor_third_party_policy_v1_0.md",
            product_type="motor_third_party",
            version="v1.0",
            chunks_ingested=12,
            chunk_ids=["motor_third_party::v1.0::TP-1.0"],
            deleted_stale_count=3,
        ),
    ]
    calls = []

    def fake_ingest_all(db, *args, **kwargs):
        calls.append(db)
        return fake_results

    monkeypatch.setattr(ingestion_module, "ingest_all", fake_ingest_all)

    def override_get_db():
        yield MagicMock()

    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        response = client.post("/ingest")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(calls) == 1  # ingest_all was actually invoked with a db session

    data = response.json()
    assert data["documents_ingested"] == 2
    assert data["results"][0]["product_type"] == "motor_comprehensive"
    assert data["results"][0]["chunks_ingested"] == 24
    assert data["results"][1]["deleted_stale_count"] == 3
