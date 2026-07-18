"""Tests for graph/tools.py -- the two LLM-callable tools.

policy_lookup_fn is tested with a fake embed_fn/index (mirroring the
pattern in test_rag_ingestion.py). customer_history_lookup_fn is tested
against the real repositories on the sqlite test session, reusing the same
seeded-customer pattern as test_repositories.py.
"""

from datetime import date, datetime, timedelta, timezone

from customer_support_agent.graph.tools import (
    customer_history_lookup_fn,
    make_customer_history_tool,
    make_policy_lookup_tool,
    policy_lookup_fn,
)
from customer_support_agent.models import PolicyStatus
from customer_support_agent.repositories import (
    CustomerPolicyRepository,
    CustomerRepository,
    InteractionRepository,
    PolicyDocumentRepository,
    TicketRepository,
)


class FakeQueryIndex:
    def __init__(self, matches):
        self._matches = matches
        self.query_calls = []

    def query(self, vector, top_k, include_metadata, filter=None):
        self.query_calls.append({"vector": vector, "top_k": top_k, "filter": filter})
        return {"matches": self._matches[:top_k]}


def fake_embed_fn(texts):
    return [[float(len(t)), 0.0] for t in texts]


def test_policy_lookup_fn_returns_expected_shape():
    fake_matches = [
        {
            "score": 0.91,
            "metadata": {
                "clause_id": "OD-4.2",
                "clause_title": "Parked Vehicle",
                "section": "What This Policy Covers",
                "text": "Own damage cover applies while parked...",
                "product_type": "motor_comprehensive",
                "policy_version": "v3.2",
            },
        }
    ]
    index = FakeQueryIndex(fake_matches)

    results = policy_lookup_fn(
        "does parked car damage get covered", embed_fn=fake_embed_fn, index=index
    )

    assert len(results) == 1
    assert results[0]["clause_id"] == "OD-4.2"
    assert results[0]["score"] == 0.91
    assert index.query_calls[0]["top_k"] == 5


def test_policy_lookup_fn_applies_product_type_filter():
    index = FakeQueryIndex([])
    policy_lookup_fn("test", product_type="motor_third_party", embed_fn=fake_embed_fn, index=index)

    assert index.query_calls[0]["filter"] == {"product_type": {"$eq": "motor_third_party"}}


def _seed_customer_with_policy_and_history(session, status=PolicyStatus.ACTIVE):
    customers = CustomerRepository(session)
    docs = PolicyDocumentRepository(session)
    policies = CustomerPolicyRepository(session)
    tickets = TicketRepository(session)
    interactions = InteractionRepository(session)

    customer = customers.create(name="Rohan Sharma", contact_no="+91-9876500000")
    doc = docs.create(
        product_type="motor_comprehensive", version="v3.2", source_file="x.md",
        ingested_at=datetime.now(timezone.utc),
    )
    policy = policies.create(
        customer_id=customer.id,
        policy_document_id=doc.id,
        policy_number="POL-MC-999999",
        vehicle_reg_no="TS09AB1234",
        start_date=date.today() - timedelta(days=30),
        expiry_date=date.today() + timedelta(days=335) if status == PolicyStatus.ACTIVE
        else date.today() - timedelta(days=10),
        premium_amount="10000.00",
        status=status,
    )
    ticket = tickets.create(customer_id=customer.id, customer_policy_id=policy.id)
    session.flush()
    interactions.create(
        ticket_id=ticket.id, summary="Asked about glass cover.", faithfulness_pass=True
    )
    session.commit()
    return customer


def test_customer_history_lookup_fn_active_policy(session):
    customer = _seed_customer_with_policy_and_history(session, status=PolicyStatus.ACTIVE)

    result = customer_history_lookup_fn(customer.id, session)

    assert result["has_active_policy"] is True
    assert result["policy_status"] == "active"
    assert len(result["recent_interactions"]) == 1
    assert "glass cover" in result["recent_interactions"][0]["summary"]


def test_customer_history_lookup_fn_lapsed_policy(session):
    customer = _seed_customer_with_policy_and_history(session, status=PolicyStatus.LAPSED)

    result = customer_history_lookup_fn(customer.id, session)

    assert result["has_active_policy"] is False
    # Even though there's no *active* policy, the agent still needs the
    # actual details to give a useful answer ("your policy lapsed on X").
    assert result["policy_status"] == "lapsed"
    assert result["policy_number"] == "POL-MC-999999"
    assert result["policy_expiry_date"] is not None


def test_make_policy_lookup_tool_is_invokable_langchain_tool():
    index = FakeQueryIndex([])
    tool_obj = make_policy_lookup_tool(embed_fn=fake_embed_fn, index=index)

    assert tool_obj.name == "policy_lookup"
    result = tool_obj.invoke({"query": "test question"})
    assert result == []


def test_make_customer_history_tool_is_invokable_langchain_tool(session):
    customer = _seed_customer_with_policy_and_history(session)
    tool_obj = make_customer_history_tool(session)

    assert tool_obj.name == "customer_history_lookup"
    result = tool_obj.invoke({"customer_id": customer.id})
    assert result["has_active_policy"] is True
