"""End-to-end test of the compiled LangGraph graph, using a fake LLM and
fake Pinecone client so no real API calls happen. This proves the full
wiring (nodes + conditional edge + DB writes), building on the
already-tested individual pieces in test_agent_nodes.py / test_agent_tools.py.
"""

import json
from datetime import date, timedelta

from customer_support_agent.graph.build_graph import build_graph
from customer_support_agent.models import (
    Escalation,
    Interaction,
    PolicyStatus,
    Ticket,
)
from customer_support_agent.repositories import (
    AgentRepository,
    CustomerPolicyRepository,
    CustomerRepository,
    PolicyDocumentRepository,
    TicketRepository,
)


class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    """Shared queue consumed strictly in call order across every node that
    invokes this same llm instance (classify, agent_reasoning after
    bind_tools returns self, and summarize) -- matches real execution order
    since the graph runs nodes sequentially per its edges."""

    def __init__(self, responses):
        self._responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._responses.pop(0)


class FakeQueryIndex:
    def __init__(self, matches):
        self._matches = matches

    def query(self, vector, top_k, include_metadata, filter=None):
        return {"matches": self._matches[:top_k]}


def fake_embed_fn(texts):
    return [[float(len(t)), 0.0] for t in texts]


FAKE_CLAUSE_MATCHES = [
    {
        "score": 0.92,
        "metadata": {
            "clause_id": "OD-4.2",
            "clause_title": "Parked Vehicle -- Unknown Cause",
            "section": "What This Policy Covers",
            "text": "Own damage cover applies while parked, subject to excess.",
            "product_type": "motor_comprehensive",
            "policy_version": "v3.2",
        },
    }
]


def _seed_ticket_with_active_policy(session):
    customers = CustomerRepository(session)
    docs = PolicyDocumentRepository(session)
    policies = CustomerPolicyRepository(session)
    tickets = TicketRepository(session)
    agents = AgentRepository(session)

    agents.get_or_create_by_name("Arjun Mehta", role="supervisor")

    customer = customers.create(name="Rohan Sharma", contact_no="+91-9800000001")
    doc = docs.create(product_type="motor_comprehensive", version="v3.2", source_file="x.md")
    policy = policies.create(
        customer_id=customer.id,
        policy_document_id=doc.id,
        policy_number="POL-MC-000555",
        vehicle_reg_no="TS09AB1234",
        start_date=date.today() - timedelta(days=30),
        expiry_date=date.today() + timedelta(days=335),
        premium_amount="12500.00",
        status=PolicyStatus.ACTIVE,
    )
    ticket = tickets.create(customer_id=customer.id, customer_policy_id=policy.id)
    session.commit()
    return customer, ticket


def test_graph_happy_path_presents_to_human_not_escalated(session):
    customer, ticket = _seed_ticket_with_active_policy(session)

    summarize_payload = {
        "summary": "Customer asked whether parked-car damage is covered.",
        "draft_body": "Yes, this is covered under clause OD-4.2, subject to your excess.",
        "cited_clause_ids": ["OD-4.2"],
        "needs_escalation": False,
        "escalation_reason": None,
    }
    llm = FakeLLM(
        [
            FakeMessage(content="coverage_question"),  # classify
            FakeMessage(  # agent_reasoning: call policy_lookup
                tool_calls=[
                    {"name": "policy_lookup", "args": {"query": "parked car damage"}, "id": "call_1"}
                ]
            ),
            FakeMessage(content="done", tool_calls=[]),  # agent_reasoning: stop
            FakeMessage(content=json.dumps(summarize_payload)),  # summarize
        ]
    )

    graph = build_graph(
        session, llm=llm, embed_fn=fake_embed_fn, index=FakeQueryIndex(FAKE_CLAUSE_MATCHES)
    )
    final_state = graph.invoke(
        {
            "ticket_id": ticket.id,
            "customer_id": customer.id,
            "ticket_text": "My car was hit while parked, is this covered?",
        }
    )
    session.commit()

    assert final_state["category"] == "coverage_question"
    assert final_state["faithfulness_pass"] is True
    assert final_state["escalated"] is False
    assert final_state["customer_name"] == "Rohan Sharma"
    assert final_state["draft_response"].startswith("Dear Rohan Sharma,")
    assert final_state["draft_response"].endswith("Warm regards,\nMotor Insurance Support Team")
    assert "OD-4.2" in final_state["draft_response"] or "covered" in final_state["draft_response"]

    interaction = session.query(Interaction).filter_by(ticket_id=ticket.id).one()
    assert interaction.escalated is False
    assert interaction.faithfulness_pass is True

    assert session.query(Escalation).count() == 0


def test_graph_escalates_on_unsupported_citation(session):
    """The LLM cites a clause that was never retrieved -- faithfulness_check
    should catch this and the gate should route to escalation, with no
    manual escalation flag needed from the model itself."""
    customer, ticket = _seed_ticket_with_active_policy(session)

    summarize_payload = {
        "summary": "Customer asked about parked-car cover.",
        "draft_body": "This is covered under clause ZZ-9.9.",
        "cited_clause_ids": ["ZZ-9.9"],  # never retrieved -- fabricated citation
        "needs_escalation": False,
        "escalation_reason": None,
    }
    llm = FakeLLM(
        [
            FakeMessage(content="coverage_question"),
            FakeMessage(
                tool_calls=[
                    {"name": "policy_lookup", "args": {"query": "parked car damage"}, "id": "call_1"}
                ]
            ),
            FakeMessage(content="done", tool_calls=[]),
            FakeMessage(content=json.dumps(summarize_payload)),
        ]
    )

    graph = build_graph(
        session, llm=llm, embed_fn=fake_embed_fn, index=FakeQueryIndex(FAKE_CLAUSE_MATCHES)
    )
    final_state = graph.invoke(
        {
            "ticket_id": ticket.id,
            "customer_id": customer.id,
            "ticket_text": "My car was hit while parked, is this covered?",
        }
    )
    session.commit()

    assert final_state["faithfulness_pass"] is False
    assert final_state["escalated"] is True

    escalation = session.query(Escalation).one()
    assert escalation.priority.value == "high"  # faithfulness failure -> high priority
    assert escalation.assigned_agent.name == "Arjun Mehta"


def test_graph_escalates_on_high_value_claim(session):
    """End-to-end proof of the Rs. 1L claim-value escalation rule."""
    customer, ticket = _seed_ticket_with_active_policy(session)

    summarize_payload = {
        "summary": "Customer is claiming Rs. 2,50,000 for major accident repair.",
        "draft_body": "We've noted the repair estimate you provided.",
        "cited_clause_ids": [],
        "claimed_amount": 250000,
        "needs_escalation": False,
        "escalation_reason": None,
    }
    llm = FakeLLM(
        [
            FakeMessage(content="claim_status"),
            FakeMessage(content="done", tool_calls=[]),  # no tools needed
            FakeMessage(content=json.dumps(summarize_payload)),
        ]
    )

    graph = build_graph(session, llm=llm, embed_fn=fake_embed_fn, index=FakeQueryIndex([]))
    final_state = graph.invoke(
        {
            "ticket_id": ticket.id,
            "customer_id": customer.id,
            "ticket_text": "My repair estimate came to Rs 2,50,000, please process my claim.",
        }
    )
    session.commit()

    assert final_state["claimed_amount"] == 250000
    assert final_state["escalated"] is True
    assert "High-value claim" in final_state["escalation_reason"]


def test_graph_escalates_on_complaint_category(session):
    """A complaint should always escalate, even with a clean faithfulness
    pass and no soft signal from the model."""
    customer, ticket = _seed_ticket_with_active_policy(session)

    summarize_payload = {
        "summary": "Customer is unhappy with claim handling delays.",
        "draft_body": "We're sorry to hear about the delay.",
        "cited_clause_ids": [],
        "needs_escalation": False,
        "escalation_reason": None,
    }
    llm = FakeLLM(
        [
            FakeMessage(content="complaint"),
            FakeMessage(content="done", tool_calls=[]),  # no tools needed for a complaint
            FakeMessage(content=json.dumps(summarize_payload)),
        ]
    )

    graph = build_graph(session, llm=llm, embed_fn=fake_embed_fn, index=FakeQueryIndex([]))
    final_state = graph.invoke(
        {
            "ticket_id": ticket.id,
            "customer_id": customer.id,
            "ticket_text": "This is the third time my claim has been delayed, I'm furious.",
        }
    )
    session.commit()

    assert final_state["category"] == "complaint"
    assert final_state["escalated"] is True
