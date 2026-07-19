"""Tests for POST /chat -- the customer chat assistant endpoint.

Mirrors the same monkeypatching pattern used for /tickets/{id}/process:
run_chat_turn's actual behavior is already covered in test_chat_agent.py,
this file only proves the endpoint wiring."""

import json

from fastapi.testclient import TestClient

import customer_support_agent.api.chat as chat_module
from customer_support_agent.api.deps import get_db
from customer_support_agent.repositories import CustomerRepository
from main import app


def _client(session):
    def override_get_db():
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_chat_endpoint_calls_run_chat_turn_and_shapes_response(session, monkeypatch):
    customer = CustomerRepository(session).create(name="Rohan Sharma", contact_no="+91-9800000001")
    session.commit()

    calls = []

    def fake_run_chat_turn(db, customer_id, customer_name, history, message, **kwargs):
        calls.append((customer_id, customer_name, history, message))
        updated = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Your policy is active."},
        ]
        return "Your policy is active.", updated

    monkeypatch.setattr(chat_module, "run_chat_turn", fake_run_chat_turn)

    client = _client(session)
    response = client.post(
        "/chat",
        json={"customer_id": customer.id, "message": "What's my policy status?", "history": []},
    )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Your policy is active."
    assert len(data["history"]) == 2
    assert calls[0][0] == customer.id
    assert calls[0][1] == "Rohan Sharma"


def test_chat_endpoint_customer_not_found_returns_404(session):
    client = _client(session)
    response = client.post("/chat", json={"customer_id": 999999, "message": "hello", "history": []})
    app.dependency_overrides.clear()

    assert response.status_code == 404


def test_chat_endpoint_rejects_empty_message(session):
    customer = CustomerRepository(session).create(name="Rohan Sharma", contact_no="+91-9800000002")
    session.commit()

    client = _client(session)
    response = client.post("/chat", json={"customer_id": customer.id, "message": "", "history": []})
    app.dependency_overrides.clear()

    assert response.status_code == 422


def test_chat_endpoint_passes_through_conversation_history(session, monkeypatch):
    customer = CustomerRepository(session).create(name="Rohan Sharma", contact_no="+91-9800000003")
    session.commit()

    captured_history = []

    def fake_run_chat_turn(db, customer_id, customer_name, history, message, **kwargs):
        captured_history.append(history)
        return "ok", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "ok"},
        ]

    monkeypatch.setattr(chat_module, "run_chat_turn", fake_run_chat_turn)

    client = _client(session)
    prior_history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    client.post(
        "/chat",
        json={"customer_id": customer.id, "message": "Thanks", "history": prior_history},
    )
    app.dependency_overrides.clear()

    assert captured_history[0] == prior_history


class _FlakyThenSucceeds:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.call_count = 0

    def __call__(self, db, customer_id, customer_name, history, message, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise json.JSONDecodeError("Extra data", "bad body", 42)
        return "recovered reply", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "recovered reply"},
        ]


def test_chat_endpoint_retries_on_transient_json_error_and_succeeds(session, monkeypatch):
    customer = CustomerRepository(session).create(name="Rohan Sharma", contact_no="+91-9800000004")
    session.commit()

    flaky = _FlakyThenSucceeds(fail_times=2)
    monkeypatch.setattr(chat_module, "run_chat_turn", flaky)

    client = _client(session)
    response = client.post(
        "/chat", json={"customer_id": customer.id, "message": "hello", "history": []}
    )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["reply"] == "recovered reply"
    assert flaky.call_count == 3


def test_chat_endpoint_returns_502_after_exhausting_retries(session, monkeypatch):
    customer = CustomerRepository(session).create(name="Rohan Sharma", contact_no="+91-9800000005")
    session.commit()

    def always_fails(db, customer_id, customer_name, history, message, **kwargs):
        raise json.JSONDecodeError("Extra data", "bad body", 42)

    monkeypatch.setattr(chat_module, "run_chat_turn", always_fails)

    client = _client(session)
    response = client.post(
        "/chat", json={"customer_id": customer.id, "message": "hello", "history": []}
    )
    app.dependency_overrides.clear()

    assert response.status_code == 502
