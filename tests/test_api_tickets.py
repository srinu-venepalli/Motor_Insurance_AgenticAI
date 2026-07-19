"""Tests for the /tickets API endpoints.

/process is tested by monkeypatching build_graph itself -- the graph's
actual behavior already has thorough coverage in test_agent_graph.py; this
file only proves the endpoint wiring (request validation, DB writes,
response shape).
"""

from datetime import date, timedelta
import json

import pytest
from fastapi.testclient import TestClient

import customer_support_agent.api.tickets as tickets_module
from customer_support_agent.api.deps import get_db
from customer_support_agent.models import PolicyStatus
from customer_support_agent.repositories import (
    CustomerPolicyRepository,
    CustomerRepository,
    PolicyDocumentRepository,
)
from main import app


@pytest.fixture()
def client(session):
    def override_get_db():
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def _seed_customer(session):
    customers = CustomerRepository(session)
    docs = PolicyDocumentRepository(session)
    policies = CustomerPolicyRepository(session)

    customer = customers.create(name="Rohan Sharma", contact_no="+91-9800000001")
    doc = docs.create(product_type="motor_comprehensive", version="v3.2", source_file="x.md")
    policies.create(
        customer_id=customer.id,
        policy_document_id=doc.id,
        policy_number="POL-MC-000001",
        vehicle_reg_no="TS01AA1111",
        start_date=date.today() - timedelta(days=10),
        expiry_date=date.today() + timedelta(days=355),
        premium_amount="10000.00",
        status=PolicyStatus.ACTIVE,
    )
    session.commit()
    return customer


# --- POST /tickets ----------------------------------------------------


def test_create_ticket_success(client, session):
    customer = _seed_customer(session)

    response = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Does my policy cover a cracked windscreen?"}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["customer_id"] == customer.id
    assert data["status"] == "open"
    assert "ticket_id" in data

    from customer_support_agent.repositories import TicketMessageRepository

    messages = TicketMessageRepository(session).get_thread(data["ticket_id"])
    assert len(messages) == 1
    assert "cracked windscreen" in messages[0].text


def test_create_ticket_unknown_customer_returns_404(client):
    response = client.post("/tickets", json={"customer_id": 999999, "ticket_text": "Hello"})
    assert response.status_code == 404


def test_create_ticket_rejects_empty_text(client, session):
    customer = _seed_customer(session)
    response = client.post("/tickets", json={"customer_id": customer.id, "ticket_text": ""})
    assert response.status_code == 422  # pydantic min_length validation


# --- POST /tickets/{id}/process -----------------------------------------


class FakeGraph:
    def __init__(self, fixed_result):
        self._fixed_result = fixed_result
        self.invoked_with = None

    def invoke(self, state, config=None):
        self.invoked_with = state
        return self._fixed_result


FIXED_RESULT = {
    "category": "coverage_question",
    "tool_calls_made": ["policy_lookup"],
    "retrieved_clauses": [{"clause_id": "OD-4.2"}],
    "customer_context": {"has_active_policy": True},
    "claimed_amount": None,
    "faithfulness_pass": True,
    "faithfulness_reason": "All cited clauses match retrieved clauses.",
    "escalated": False,
    "escalation_reason": None,
    "summary": "Customer asked about windscreen cover.",
    "draft_response": "Dear Rohan Sharma,\n\nYes, covered.\n\nWarm regards,\nMotor Insurance Support Team",
}


def test_process_ticket_calls_graph_and_shapes_response(client, session, monkeypatch):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    fake_graph = FakeGraph(FIXED_RESULT)
    monkeypatch.setattr(tickets_module, "build_graph", lambda db: fake_graph)

    response = client.post(f"/tickets/{ticket_id}/process")

    assert response.status_code == 200
    data = response.json()
    assert data["ticket_id"] == ticket_id
    assert data["category"] == "coverage_question"
    assert data["tool_calls_made"] == ["policy_lookup"]
    assert data["retrieved_clauses_count"] == 1
    assert data["escalated"] is False
    assert "Yes, covered." in data["draft_response"]

    # Confirm the graph was actually invoked with the ticket's stored text.
    assert fake_graph.invoked_with["ticket_text"] == "Cracked windscreen?"
    assert fake_graph.invoked_with["customer_id"] == customer.id


def test_process_ticket_not_found_returns_404(client, monkeypatch):
    monkeypatch.setattr(tickets_module, "build_graph", lambda db: FakeGraph(FIXED_RESULT))
    response = client.post("/tickets/999999/process")
    assert response.status_code == 404


class FlakyThenSucceedsGraph:
    """Simulates the real-world proxy bug: raises json.JSONDecodeError on
    the first N calls, then returns a valid result -- proves the retry
    wrapper actually retries rather than just being decorative."""

    def __init__(self, fail_times: int, fixed_result: dict):
        self.fail_times = fail_times
        self.fixed_result = fixed_result
        self.call_count = 0

    def invoke(self, state, config=None):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise json.JSONDecodeError("Extra data", "bad json body", 42)
        return self.fixed_result


class AlwaysFailsGraph:
    def invoke(self, state, config=None):
        raise json.JSONDecodeError("Extra data", "bad json body", 42)


def test_process_ticket_retries_on_transient_json_error_and_succeeds(client, session, monkeypatch):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    flaky_graph = FlakyThenSucceedsGraph(fail_times=2, fixed_result=FIXED_RESULT)
    monkeypatch.setattr(tickets_module, "build_graph", lambda db: flaky_graph)

    response = client.post(f"/tickets/{ticket_id}/process")

    assert response.status_code == 200
    assert flaky_graph.call_count == 3  # failed twice, succeeded on the 3rd
    assert response.json()["category"] == "coverage_question"


def test_process_ticket_returns_502_after_exhausting_retries(client, session, monkeypatch):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    monkeypatch.setattr(tickets_module, "build_graph", lambda db: AlwaysFailsGraph())

    response = client.post(f"/tickets/{ticket_id}/process")

    assert response.status_code == 502
    assert "transient" in response.json()["detail"].lower() or "try again" in response.json()["detail"].lower()

    # No AI_DRAFT message should have been written for a failed run.
    from customer_support_agent.models import MessageSender
    from customer_support_agent.repositories import TicketMessageRepository

    thread = TicketMessageRepository(session).get_thread(ticket_id)
    assert not any(m.sender == MessageSender.AI_DRAFT for m in thread)


def test_process_ticket_without_message_returns_400(client, session, monkeypatch):
    """A ticket created directly via the repository (bypassing POST
    /tickets) has no message thread -- process should fail clearly, not
    call the graph with empty text."""
    customer = _seed_customer(session)
    from customer_support_agent.repositories import TicketRepository

    ticket = TicketRepository(session).create(customer_id=customer.id)
    session.commit()

    monkeypatch.setattr(tickets_module, "build_graph", lambda db: FakeGraph(FIXED_RESULT))
    response = client.post(f"/tickets/{ticket.id}/process")
    assert response.status_code == 400


# --- GET /tickets/{id} ---------------------------------------------------


def test_get_ticket_detail_includes_interactions(client, session):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    from customer_support_agent.repositories import InteractionRepository

    InteractionRepository(session).create(
        ticket_id=ticket_id,
        summary="Answered directly.",
        faithfulness_pass=True,
        escalated=True,
        escalation_reason="Repeat customer: 3 tickets opened in the last 30 days (threshold: 2).",
    )
    session.commit()

    response = client.get(f"/tickets/{ticket_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["interactions"][0]["escalation_reason"] == (
        "Repeat customer: 3 tickets opened in the last 30 days (threshold: 2)."
    )
    assert data["customer_name"] == "Rohan Sharma"
    assert len(data["interactions"]) == 1
    assert data["interactions"][0]["summary"] == "Answered directly."


def test_get_ticket_not_found_returns_404(client):
    response = client.get("/tickets/999999")
    assert response.status_code == 404


# --- draft persistence (the actual gap this session's changes fixed) ----


def test_process_ticket_persists_draft_as_ai_draft_message(client, session, monkeypatch):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    fake_graph = FakeGraph(FIXED_RESULT)
    monkeypatch.setattr(tickets_module, "build_graph", lambda db: fake_graph)
    client.post(f"/tickets/{ticket_id}/process")

    from customer_support_agent.models import MessageSender
    from customer_support_agent.repositories import TicketMessageRepository

    thread = TicketMessageRepository(session).get_thread(ticket_id)
    ai_drafts = [m for m in thread if m.sender == MessageSender.AI_DRAFT]
    assert len(ai_drafts) == 1
    assert "Yes, covered." in ai_drafts[0].text


def test_process_ticket_persists_classified_category_to_ticket_row(client, session, monkeypatch):
    """Regression test for a real bug: classify_ticket correctly determined
    the category, but /process only ever included it in this endpoint's
    own response -- it never wrote it back to the Ticket row. Every ticket
    is created with category=OTHER by default (the customer never picks
    one), so GET /tickets/{id} kept showing 'other' regardless of what the
    AI actually classified, since that endpoint reads the persisted
    column, not the ephemeral graph result."""
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]
    assert client.get(f"/tickets/{ticket_id}").json()["category"] == "other"  # default at creation

    fake_graph = FakeGraph(FIXED_RESULT)  # FIXED_RESULT["category"] == "coverage_question"
    monkeypatch.setattr(tickets_module, "build_graph", lambda db: fake_graph)
    client.post(f"/tickets/{ticket_id}/process")

    detail = client.get(f"/tickets/{ticket_id}").json()
    assert detail["category"] == "coverage_question"

    listed = client.get("/tickets").json()
    this_ticket = next(t for t in listed if t["ticket_id"] == ticket_id)
    assert this_ticket["category"] == "coverage_question"


def test_process_ticket_ignores_unrecognized_category_gracefully(client, session, monkeypatch):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    bad_result = dict(FIXED_RESULT, category="not_a_real_category")
    monkeypatch.setattr(tickets_module, "build_graph", lambda db: FakeGraph(bad_result))

    response = client.post(f"/tickets/{ticket_id}/process")
    assert response.status_code == 200  # doesn't crash the whole request

    detail = client.get(f"/tickets/{ticket_id}").json()
    assert detail["category"] == "other"  # left at its original default, not corrupted


# --- GET /tickets (list / queue) -----------------------------------------


def test_list_tickets_returns_all_by_default(client, session):
    customer = _seed_customer(session)
    client.post("/tickets", json={"customer_id": customer.id, "ticket_text": "First ticket"})
    client.post("/tickets", json={"customer_id": customer.id, "ticket_text": "Second ticket"})

    response = client.get("/tickets")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["customer_name"] == "Rohan Sharma"
    assert data[0]["escalated"] is None  # not yet processed
    assert data[0]["latest_summary"] is None  # not yet processed


def test_list_tickets_includes_latest_summary_after_interaction(client, session):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Fire damage question"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    from customer_support_agent.repositories import InteractionRepository

    InteractionRepository(session).create(
        ticket_id=ticket_id,
        summary="Customer asked about fire damage coverage.",
        faithfulness_pass=True,
        escalated=False,
    )
    session.commit()

    response = client.get("/tickets")
    data = response.json()
    assert data[0]["latest_summary"] == "Customer asked about fire damage coverage."


def test_list_tickets_filters_by_resolution(client, session, monkeypatch):
    customer = _seed_customer(session)

    approved_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Approved one"}
    )
    rejected_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Rejected one"}
    )
    approved_id = approved_resp.json()["ticket_id"]
    rejected_id = rejected_resp.json()["ticket_id"]

    monkeypatch.setattr(tickets_module, "build_graph", lambda db: FakeGraph(FIXED_RESULT))
    client.post(f"/tickets/{approved_id}/process")
    client.post(f"/tickets/{rejected_id}/process")

    client.post(f"/tickets/{approved_id}/approve", json={"resolution": "approved"})
    client.post(f"/tickets/{rejected_id}/approve", json={"resolution": "rejected"})

    approved_only = client.get("/tickets?resolution=approved").json()
    rejected_only = client.get("/tickets?resolution=rejected").json()

    assert [t["ticket_id"] for t in approved_only] == [approved_id]
    assert [t["ticket_id"] for t in rejected_only] == [rejected_id]


def test_list_tickets_rejects_invalid_resolution_filter(client):
    response = client.get("/tickets?resolution=maybe")
    assert response.status_code == 422


def test_list_tickets_filters_by_customer_id(client, session):
    customer_a = _seed_customer(session)
    from customer_support_agent.repositories import CustomerRepository

    customer_b = CustomerRepository(session).create(name="Anita Rao", contact_no="+91-9000000002")
    session.commit()

    client.post("/tickets", json={"customer_id": customer_a.id, "ticket_text": "A's ticket"})
    client.post("/tickets", json={"customer_id": customer_b.id, "ticket_text": "B's ticket"})

    response = client.get(f"/tickets?customer_id={customer_a.id}")

    data = response.json()
    assert len(data) == 1
    assert data[0]["customer_id"] == customer_a.id


def test_list_tickets_reflects_escalated_flag_after_processing(client, session):
    """FakeGraph.invoke() just returns a dict -- it doesn't replicate what
    the real graph's terminal nodes (present_to_human/escalate_to_agent)
    actually do, which is write an Interaction row. So this test seeds that
    row directly rather than going through /process with a fake graph."""
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    from customer_support_agent.repositories import InteractionRepository

    InteractionRepository(session).create(
        ticket_id=ticket_id, summary="Escalated case.", faithfulness_pass=True, escalated=True
    )
    session.commit()

    response = client.get("/tickets")
    data = response.json()
    assert data[0]["escalated"] is True


# --- POST /tickets/{id}/approve ------------------------------------------


def test_approve_ticket_sends_latest_draft_and_closes_ticket(client, session, monkeypatch):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    monkeypatch.setattr(tickets_module, "build_graph", lambda db: FakeGraph(FIXED_RESULT))
    client.post(f"/tickets/{ticket_id}/process")

    response = client.post(f"/tickets/{ticket_id}/approve", json={})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "closed"
    assert "Yes, covered." in data["sent_text"]

    from customer_support_agent.models import MessageSender
    from customer_support_agent.repositories import TicketMessageRepository

    thread = TicketMessageRepository(session).get_thread(ticket_id)
    human_messages = [m for m in thread if m.sender == MessageSender.HUMAN_AGENT]
    assert len(human_messages) == 1
    assert human_messages[0].text == data["sent_text"]


def test_approve_ticket_uses_edited_text_when_provided(client, session, monkeypatch):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    monkeypatch.setattr(tickets_module, "build_graph", lambda db: FakeGraph(FIXED_RESULT))
    client.post(f"/tickets/{ticket_id}/process")

    edited_text = "Dear Rohan, edited by human agent for clarity.\n\nRegards, Support"
    response = client.post(f"/tickets/{ticket_id}/approve", json={"final_response": edited_text})

    assert response.status_code == 200
    assert response.json()["sent_text"] == edited_text


def test_approve_ticket_without_draft_returns_400(client, session):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    # Never processed -- no AI draft exists yet.
    response = client.post(f"/tickets/{ticket_id}/approve", json={})
    assert response.status_code == 400


def test_approve_ticket_not_found_returns_404(client):
    response = client.post("/tickets/999999/approve", json={})
    assert response.status_code == 404


def test_reject_claim_closes_ticket_with_rejected_resolution(client, session, monkeypatch):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    monkeypatch.setattr(tickets_module, "build_graph", lambda db: FakeGraph(FIXED_RESULT))
    client.post(f"/tickets/{ticket_id}/process")

    rejection_text = "Dear Rohan Sharma,\n\nWe regret this claim does not meet policy criteria.\n\nRegards"
    response = client.post(
        f"/tickets/{ticket_id}/approve",
        json={"final_response": rejection_text, "resolution": "rejected"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "closed"
    assert data["resolution"] == "rejected"
    assert data["sent_text"] == rejection_text

    detail = client.get(f"/tickets/{ticket_id}").json()
    assert detail["resolution"] == "rejected"


def test_approve_ticket_rejects_invalid_resolution_value(client, session, monkeypatch):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    monkeypatch.setattr(tickets_module, "build_graph", lambda db: FakeGraph(FIXED_RESULT))
    client.post(f"/tickets/{ticket_id}/process")

    response = client.post(f"/tickets/{ticket_id}/approve", json={"resolution": "maybe"})
    assert response.status_code == 422


def test_approve_ticket_records_edit_distance_feedback(client, session, monkeypatch):
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    monkeypatch.setattr(tickets_module, "build_graph", lambda db: FakeGraph(FIXED_RESULT))
    client.post(f"/tickets/{ticket_id}/process")

    # FakeGraph.invoke() just returns a dict -- it doesn't replicate what the
    # real graph's terminal nodes do (write an Interaction row), so seed one
    # directly to exercise the Feedback-recording path.
    from customer_support_agent.repositories import FeedbackRepository, InteractionRepository

    interaction = InteractionRepository(session).create(
        ticket_id=ticket_id, summary="Answered directly.", faithfulness_pass=True, escalated=False
    )
    session.commit()

    original_draft = FIXED_RESULT["draft_response"]
    edited_text = original_draft + " Additional edited note."
    response = client.post(
        f"/tickets/{ticket_id}/approve", json={"final_response": edited_text}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["edit_distance"] == len(" Additional edited note.")

    feedback = FeedbackRepository(session).list_for_interaction(interaction.id)
    assert len(feedback) == 1
    assert feedback[0].edit_distance == len(" Additional edited note.")
    assert "approved" in feedback[0].notes


def test_approve_ticket_no_edit_distance_when_no_prior_draft(client, session):
    """Approving without ever processing (explicit final_response, no
    AI draft exists) shouldn't try to compute a meaningless edit distance."""
    customer = _seed_customer(session)
    create_resp = client.post(
        "/tickets", json={"customer_id": customer.id, "ticket_text": "Cracked windscreen?"}
    )
    ticket_id = create_resp.json()["ticket_id"]

    response = client.post(
        f"/tickets/{ticket_id}/approve", json={"final_response": "Manually written reply."}
    )

    assert response.status_code == 200
    assert response.json()["edit_distance"] is None
