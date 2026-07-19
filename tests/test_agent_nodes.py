"""Tests for graph/nodes.py, using a fake LLM double so no real API calls
happen. Each node is tested in isolation before the full graph is wired
together in test_agent_graph.py.
"""

import json
from datetime import date, timedelta

from customer_support_agent.graph.nodes import (
    faithfulness_check,
    make_agent_reasoning_node,
    make_classify_node,
    make_escalate_node,
    make_escalation_gate,
    make_fetch_customer_context_node,
    make_present_node,
    make_summarize_node,
)
from customer_support_agent.graph.tools import make_policy_lookup_tool
from customer_support_agent.models import PolicyStatus, TicketCategory
from customer_support_agent.repositories import (
    AgentRepository,
    CustomerPolicyRepository,
    CustomerRepository,
    FeedbackRepository,
    InteractionRepository,
    PolicyDocumentRepository,
    TicketRepository,
)


class FakeMessage:
    """Minimal stand-in for a LangChain AIMessage."""

    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    """Scripted fake: .invoke() pops responses off a queue in call order.
    .bind_tools() returns self (mutating in place), matching how the real
    ChatOpenAI.bind_tools() is used in agent_reasoning."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.invoke_call_count = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invoke_call_count += 1
        return self._responses.pop(0)


# --- classify_ticket ---------------------------------------------------


def test_classify_node_matches_known_category():
    llm = FakeLLM([FakeMessage(content="coverage_question")])
    node = make_classify_node(llm)

    result = node({"ticket_text": "Does my policy cover a cracked windscreen?"})

    assert result["category"] == "coverage_question"


def test_classify_node_falls_back_to_other_on_garbage_output():
    llm = FakeLLM([FakeMessage(content="I'm not sure honestly")])
    node = make_classify_node(llm)

    result = node({"ticket_text": "asdkjhaskjdh"})

    assert result["category"] == "other"


def test_classify_node_handles_title_case_with_spaces():
    """This is the actual bug that was misclassifying most tickets as
    'other': the model very naturally answers in human-readable form
    ('Coverage Question') rather than the literal snake_case value the
    prompt asked for, and a plain substring match never catches that --
    space vs underscore -- so it silently fell through to 'other' on
    nearly every real ticket."""
    llm = FakeLLM([FakeMessage(content="Coverage Question")])
    node = make_classify_node(llm)

    result = node({"ticket_text": "Does my policy cover a cracked windscreen?"})

    assert result["category"] == "coverage_question"


def test_classify_node_handles_hyphenated_response():
    llm = FakeLLM([FakeMessage(content="claim-status")])
    node = make_classify_node(llm)

    result = node({"ticket_text": "What's the status of my claim?"})

    assert result["category"] == "claim_status"


def test_classify_node_handles_trailing_punctuation_and_quotes():
    llm = FakeLLM([FakeMessage(content='"Accident Report."')])
    node = make_classify_node(llm)

    result = node({"ticket_text": "I was in an accident yesterday."})

    assert result["category"] == "accident_report"


def test_classify_node_still_matches_substring_when_wrapped_in_a_sentence():
    """Fallback path: if the model ignores instructions and wraps the
    category in a short sentence, substring matching against the
    normalized text should still catch it."""
    llm = FakeLLM([FakeMessage(content="This ticket is about a renewal request.")])
    node = make_classify_node(llm)

    result = node({"ticket_text": "I'd like to renew my policy."})

    assert result["category"] == "renewal"


def test_classify_node_parses_json_response():
    """The primary path this node now uses -- same reliable JSON-response
    pattern as summarize_and_draft, instead of parsing loose free text."""
    llm = FakeLLM([FakeMessage(content='{"category": "coverage_question"}')])
    node = make_classify_node(llm)

    result = node({"ticket_text": "Does my policy cover a cracked windscreen?"})

    assert result["category"] == "coverage_question"


def test_classify_node_json_response_normalizes_title_case_value():
    llm = FakeLLM([FakeMessage(content='{"category": "Claim Status"}')])
    node = make_classify_node(llm)

    result = node({"ticket_text": "What's the status of my claim?"})

    assert result["category"] == "claim_status"


def test_classify_node_handles_bare_json_string_without_crashing():
    """json.loads('"Accident Report."') is valid JSON (a bare string, not
    an object) -- this must not crash, and should fall through to the
    text-parsing fallback instead."""
    llm = FakeLLM([FakeMessage(content='"Accident Report."')])
    node = make_classify_node(llm)

    result = node({"ticket_text": "I was in an accident yesterday."})

    assert result["category"] == "accident_report"


# --- agent_reasoning (tool-calling loop) --------------------------------


class FakeQueryIndex:
    def __init__(self, matches):
        self._matches = matches

    def query(self, vector, top_k, include_metadata, filter=None):
        return {"matches": self._matches[:top_k]}


def fake_embed_fn(texts):
    return [[float(len(t)), 0.0] for t in texts]


def test_agent_reasoning_calls_policy_lookup_and_stops():
    fake_matches = [
        {
            "score": 0.9,
            "metadata": {
                "clause_id": "OD-4.2",
                "clause_title": "Parked Vehicle",
                "section": "Cover",
                "text": "Own damage cover applies while parked.",
                "product_type": "motor_comprehensive",
                "policy_version": "v3.2",
            },
        }
    ]
    policy_tool = make_policy_lookup_tool(embed_fn=fake_embed_fn, index=FakeQueryIndex(fake_matches))

    class DummyHistoryTool:
        name = "customer_history_lookup"

        def invoke(self, args):
            raise AssertionError("history tool should not have been called")

    llm = FakeLLM(
        [
            FakeMessage(
                tool_calls=[
                    {"name": "policy_lookup", "args": {"query": "parked car damage"}, "id": "call_1"}
                ]
            ),
            FakeMessage(content="done", tool_calls=[]),
        ]
    )
    node = make_agent_reasoning_node(llm, policy_tool, DummyHistoryTool())

    result = node({"category": "coverage_question", "ticket_text": "Is parked damage covered?", "customer_id": 1})

    assert result["tool_calls_made"] == ["policy_lookup"]
    assert len(result["retrieved_clauses"]) == 1
    assert result["retrieved_clauses"][0]["clause_id"] == "OD-4.2"
    # customer_context key should be absent (not None) since the history
    # tool wasn't called this turn -- omitting it is what lets
    # fetch_customer_context's baseline value survive the merge.
    assert "customer_context" not in result


def test_agent_reasoning_respects_max_iterations_guard():
    """A model that never stops calling tools should be cut off, not looped
    forever -- the Phase 5 'safeguards against misuse or loops' requirement."""
    policy_tool = make_policy_lookup_tool(embed_fn=fake_embed_fn, index=FakeQueryIndex([]))

    class DummyHistoryTool:
        name = "customer_history_lookup"

    responses = [
        FakeMessage(tool_calls=[{"name": "policy_lookup", "args": {"query": "x"}, "id": f"c{i}"}])
        for i in range(5)
    ]
    llm = FakeLLM(responses)
    node = make_agent_reasoning_node(llm, policy_tool, DummyHistoryTool(), max_iterations=3)

    result = node({"category": "other", "ticket_text": "test", "customer_id": 1})

    assert llm.invoke_call_count == 3
    assert result["tool_calls_made"] == ["policy_lookup"] * 3


def test_agent_reasoning_refreshes_customer_context_when_history_tool_called():
    """When the LLM does call customer_history_lookup itself, the returned
    dict should include the fresh customer_context (overwriting the
    baseline from fetch_customer_context)."""
    policy_tool = make_policy_lookup_tool(embed_fn=fake_embed_fn, index=FakeQueryIndex([]))

    class DummyHistoryTool:
        name = "customer_history_lookup"

        def invoke(self, args):
            return {"has_active_policy": True, "policy_status": "active", "recent_interactions": []}

    llm = FakeLLM(
        [
            FakeMessage(
                tool_calls=[
                    {"name": "customer_history_lookup", "args": {"customer_id": 1}, "id": "call_1"}
                ]
            ),
            FakeMessage(content="done", tool_calls=[]),
        ]
    )
    node = make_agent_reasoning_node(llm, policy_tool, DummyHistoryTool())

    result = node({"category": "claim_status", "ticket_text": "What's my claim status?", "customer_id": 1})

    assert "customer_context" in result
    assert result["customer_context"]["has_active_policy"] is True


def test_agent_reasoning_leaves_customer_context_key_absent_when_not_refreshed():
    """The complement of the above: if the LLM never calls the history
    tool, the returned dict must NOT include a customer_context key at all
    -- including it as None would clobber fetch_customer_context's baseline
    value when LangGraph merges the partial update into state."""
    policy_tool = make_policy_lookup_tool(embed_fn=fake_embed_fn, index=FakeQueryIndex([]))

    class DummyHistoryTool:
        name = "customer_history_lookup"

    llm = FakeLLM([FakeMessage(content="done", tool_calls=[])])
    node = make_agent_reasoning_node(llm, policy_tool, DummyHistoryTool())

    result = node({"category": "other", "ticket_text": "test", "customer_id": 1})

    assert "customer_context" not in result


# --- fetch_customer_context (deterministic) ------------------------------


def test_agent_reasoning_overrides_llm_supplied_customer_id():
    """The core bug fix: even if the LLM's tool call supplies a wrong (or
    fabricated) customer_id -- which it has no reliable way to know unless
    told -- the tool must always be invoked with the real customer_id from
    state, never the model's guess."""
    policy_tool = make_policy_lookup_tool(embed_fn=fake_embed_fn, index=FakeQueryIndex([]))

    received_args = {}

    class RecordingHistoryTool:
        name = "customer_history_lookup"

        def invoke(self, args):
            received_args.update(args)
            return {"has_active_policy": True, "policy_status": "active", "recent_interactions": []}

    llm = FakeLLM(
        [
            FakeMessage(
                tool_calls=[
                    {
                        "name": "customer_history_lookup",
                        "args": {"customer_id": 999999},  # the model's fabricated guess
                        "id": "call_1",
                    }
                ]
            ),
            FakeMessage(content="done", tool_calls=[]),
        ]
    )
    node = make_agent_reasoning_node(llm, policy_tool, RecordingHistoryTool())

    result = node({"category": "claim_status", "ticket_text": "What's my status?", "customer_id": 3})

    assert received_args["customer_id"] == 3  # overridden, not the model's 999999
    assert result["customer_context"]["has_active_policy"] is True


def test_fetch_customer_context_node_populates_name_and_context(session):
    customers = CustomerRepository(session)
    docs = PolicyDocumentRepository(session)
    policies = CustomerPolicyRepository(session)

    customer = customers.create(name="Anita Rao", contact_no="+91-9800000123")
    doc = docs.create(product_type="motor_comprehensive", version="v3.2", source_file="x.md")
    policies.create(
        customer_id=customer.id,
        policy_document_id=doc.id,
        policy_number="POL-MC-000777",
        vehicle_reg_no="TS02BB2222",
        start_date=date.today() - timedelta(days=5),
        expiry_date=date.today() + timedelta(days=360),
        premium_amount="9500.00",
        status=PolicyStatus.ACTIVE,
    )
    session.commit()

    node = make_fetch_customer_context_node(session)
    result = node({"customer_id": customer.id})

    assert result["customer_name"] == "Anita Rao"
    assert result["customer_context"]["has_active_policy"] is True
    # No category in state (not yet classified in this call) -> no adaptive signal.
    assert result["category_avg_edit_distance"] is None


def test_fetch_customer_context_node_computes_adaptive_signal_from_feedback(session):
    """This is the actual Phase 7 wiring test: seed enough Feedback rows
    with a high edit_distance for a category, and confirm
    fetch_customer_context surfaces the resulting average in state --
    that's what escalation_gate's adaptive rule reads."""
    customer = CustomerRepository(session).create(name="Vikram Singh", contact_no="+91-9800000456")
    tickets_repo = TicketRepository(session)
    interactions_repo = InteractionRepository(session)
    feedback_repo = FeedbackRepository(session)
    session.commit()

    for _ in range(3):
        t = tickets_repo.create(customer_id=customer.id, category=TicketCategory.CLAIM_STATUS)
        session.commit()
        i = interactions_repo.create(ticket_id=t.id, summary="x", faithfulness_pass=True)
        session.commit()
        feedback_repo.create(interaction_id=i.id, edit_distance=150)
        session.commit()

    node = make_fetch_customer_context_node(session, adaptive_min_feedback_samples=3)
    result = node({"customer_id": customer.id, "category": "claim_status"})

    assert result["category_avg_edit_distance"] == 150.0


def test_fetch_customer_context_node_handles_unknown_customer_gracefully(session):
    node = make_fetch_customer_context_node(session)
    result = node({"customer_id": 999999})

    assert result["customer_name"] is None
    assert result["customer_context"]["has_active_policy"] is False


# --- summarize_and_draft -------------------------------------------------


def test_summarize_node_parses_valid_json_response():
    payload = {
        "summary": "Customer asked about parked-car cover.",
        "draft_body": "Yes, this is covered under clause OD-4.2.",
        "cited_clause_ids": ["OD-4.2"],
        "needs_escalation": False,
        "escalation_reason": None,
    }
    llm = FakeLLM([FakeMessage(content=json.dumps(payload))])
    node = make_summarize_node(llm)

    result = node(
        {
            "ticket_text": "Is parked damage covered?",
            "retrieved_clauses": [{"clause_id": "OD-4.2", "text": "..."}],
            "customer_context": None,
            "customer_name": "Rohan Sharma",
        }
    )

    assert result["cited_clause_ids"] == ["OD-4.2"]
    assert result["needs_escalation_soft_signal"] is False
    assert result["draft_response"].startswith("Dear Rohan Sharma,")
    assert "Yes, this is covered under clause OD-4.2." in result["draft_response"]
    assert result["draft_response"].endswith("Warm regards,\nMotor Insurance Support Team")


def test_summarize_node_falls_back_to_generic_greeting_without_customer_name():
    payload = {
        "summary": "x",
        "draft_body": "Body text.",
        "cited_clause_ids": [],
        "needs_escalation": False,
        "escalation_reason": None,
    }
    llm = FakeLLM([FakeMessage(content=json.dumps(payload))])
    node = make_summarize_node(llm)

    result = node(
        {"ticket_text": "test", "retrieved_clauses": [], "customer_context": None, "customer_name": None}
    )

    assert result["draft_response"].startswith("Dear Customer,")


def test_summarize_node_escalates_on_invalid_json():
    llm = FakeLLM([FakeMessage(content="this is not json at all")])
    node = make_summarize_node(llm)

    result = node({"ticket_text": "test", "retrieved_clauses": [], "customer_context": None})

    assert result["needs_escalation_soft_signal"] is True
    assert "structured output" in result["escalation_soft_reason"]


# --- faithfulness_check (deterministic) ----------------------------------


def test_faithfulness_check_passes_when_cited_ids_are_retrieved():
    state = {
        "retrieved_clauses": [{"clause_id": "OD-4.2"}, {"clause_id": "GL-2.1"}],
        "cited_clause_ids": ["OD-4.2"],
    }
    result = faithfulness_check(state)
    assert result["faithfulness_pass"] is True


def test_faithfulness_check_fails_on_unsupported_citation():
    state = {
        "retrieved_clauses": [{"clause_id": "OD-4.2"}],
        "cited_clause_ids": ["OD-4.2", "MADE-UP-9.9"],
    }
    result = faithfulness_check(state)
    assert result["faithfulness_pass"] is False
    assert "MADE-UP-9.9" in result["faithfulness_reason"]


def test_faithfulness_check_passes_trivially_with_no_citations():
    result = faithfulness_check({"retrieved_clauses": [], "cited_clause_ids": []})
    assert result["faithfulness_pass"] is True


# --- escalation_gate (deterministic) -------------------------------------


def test_escalation_gate_escalates_on_faithfulness_failure():
    assert make_escalation_gate()({"faithfulness_pass": False}) == "escalate"


def test_escalation_gate_escalates_on_soft_signal():
    assert make_escalation_gate()({"faithfulness_pass": True, "needs_escalation_soft_signal": True}) == "escalate"


def test_escalation_gate_escalates_on_complaint_category():
    state = {"faithfulness_pass": True, "needs_escalation_soft_signal": False, "category": "complaint"}
    assert make_escalation_gate()(state) == "escalate"


def test_escalation_gate_escalates_on_lapsed_policy_coverage_question():
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "coverage_question",
        "customer_context": {"has_active_policy": False},
    }
    assert make_escalation_gate()(state) == "escalate"


def test_escalation_gate_presents_normal_case():
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "coverage_question",
        "customer_context": {"has_active_policy": True},
    }
    assert make_escalation_gate()(state) == "present"


def test_escalation_gate_escalates_on_repeat_customer():
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "coverage_question",
        "customer_context": {"has_active_policy": True, "tickets_last_30_days": 3},
    }
    assert make_escalation_gate()(state) == "escalate"


def test_escalation_gate_does_not_escalate_at_exact_repeat_ticket_threshold():
    """Boundary check: exactly 2 tickets in 30 days should NOT escalate --
    the rule is 'more than 2', not 'at least 2'."""
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "coverage_question",
        "customer_context": {"has_active_policy": True, "tickets_last_30_days": 2},
    }
    assert make_escalation_gate()(state) == "present"


def test_escalation_gate_escalates_on_high_value_claim():
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "claim_status",
        "customer_context": {"has_active_policy": True},
        "claimed_amount": 150_000,
    }
    assert make_escalation_gate()(state) == "escalate"


def test_escalation_gate_does_not_escalate_at_exact_claim_threshold():
    """Boundary check: exactly Rs. 1,00,000 should NOT escalate -- the rule
    is 'more than 1L', not 'at least 1L'."""
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "claim_status",
        "customer_context": {"has_active_policy": True},
        "claimed_amount": 100_000,
    }
    assert make_escalation_gate()(state) == "present"


def test_escalation_gate_ignores_claimed_amount_below_threshold():
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "claim_status",
        "customer_context": {"has_active_policy": True},
        "claimed_amount": 25_000,
    }
    assert make_escalation_gate()(state) == "present"


def test_escalation_gate_presents_when_no_adaptive_signal_yet():
    """The 'before' half of the before/after story: no feedback history
    exists yet for this category (category_avg_edit_distance is None), so
    the adaptive rule shouldn't fire even though every other check passes."""
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "claim_status",
        "customer_context": {"has_active_policy": True},
        "category_avg_edit_distance": None,
    }
    assert make_escalation_gate()(state) == "present"


def test_escalation_gate_escalates_on_adaptive_high_edit_distance():
    """The 'after' half: enough feedback has accumulated showing agents
    heavily rewrite this category's drafts (avg edit distance above the
    threshold) -- the system should now auto-escalate, even with a clean
    faithfulness pass and no other trigger."""
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "claim_status",
        "customer_context": {"has_active_policy": True},
        "category_avg_edit_distance": 150.0,  # above default threshold of 80
    }
    assert make_escalation_gate()(state) == "escalate"


def test_escalation_gate_adaptive_rule_respects_explicit_threshold_override():
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "claim_status",
        "customer_context": {"has_active_policy": True},
        "category_avg_edit_distance": 150.0,
    }
    # Same data, but a looser explicit threshold should not escalate.
    assert make_escalation_gate(adaptive_edit_distance_threshold=200)(state) == "present"


def test_make_escalation_gate_explicit_override_changes_behavior():
    """Passing an explicit threshold to the factory should change behavior
    independent of whatever settings.max_tickets_per_30_days currently is."""
    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "coverage_question",
        "customer_context": {"has_active_policy": True, "tickets_last_30_days": 3},
    }
    # Default threshold (2) would escalate at 3 tickets...
    assert make_escalation_gate()(state) == "escalate"
    # ...but a looser explicit override should not.
    assert make_escalation_gate(max_tickets_per_30_days=5)(state) == "present"


def test_make_escalation_gate_uses_settings_when_not_overridden(monkeypatch):
    """This is the actual proof the externalization works end-to-end: change
    the *settings* value (as .env would), not a factory argument, and
    confirm the gate picks it up with zero code changes."""
    from customer_support_agent.graph import nodes as nodes_module

    monkeypatch.setattr(nodes_module.settings, "high_value_claim_threshold", 500_000)

    state = {
        "faithfulness_pass": True,
        "needs_escalation_soft_signal": False,
        "category": "claim_status",
        "customer_context": {"has_active_policy": True},
        "claimed_amount": 150_000,  # would have escalated under the old 100k default
    }
    assert make_escalation_gate()(state) == "present"


def test_make_escalate_node_uses_settings_supervisor_name(session, monkeypatch):
    from customer_support_agent.graph import nodes as nodes_module

    monkeypatch.setattr(nodes_module.settings, "supervisor_agent_name", "Fatima Sheikh")

    ticket = _seed_ticket(session)
    agents = AgentRepository(session)
    agents.get_or_create_by_name("Fatima Sheikh", role="support_agent")
    session.commit()

    node = make_escalate_node(session)
    result = node(
        {
            "ticket_id": ticket.id,
            "summary": "Large claim",
            "cited_clause_ids": [],
            "faithfulness_pass": True,
            "claimed_amount": 250_000,
        }
    )

    from customer_support_agent.models import Escalation

    escalation = session.query(Escalation).filter_by(id=result["escalation_id"]).one()
    assert escalation.assigned_agent.name == "Fatima Sheikh"


# --- terminal nodes: escalate_to_agent / present_to_human ----------------


def _seed_ticket(session):
    customers = CustomerRepository(session)
    docs = PolicyDocumentRepository(session)
    policies = CustomerPolicyRepository(session)
    tickets = TicketRepository(session)

    customer = customers.create(name="Rohan Sharma", contact_no="+91-9800000099")
    doc = docs.create(product_type="motor_comprehensive", version="v3.2", source_file="x.md")
    policy = policies.create(
        customer_id=customer.id,
        policy_document_id=doc.id,
        policy_number="POL-MC-000999",
        vehicle_reg_no="TS01AA1111",
        start_date=date.today() - timedelta(days=10),
        expiry_date=date.today() + timedelta(days=355),
        premium_amount="10000.00",
        status=PolicyStatus.ACTIVE,
    )
    ticket = tickets.create(customer_id=customer.id, customer_policy_id=policy.id)
    session.commit()
    return ticket


def test_escalate_node_creates_interaction_and_escalation(session):
    ticket = _seed_ticket(session)
    agents = AgentRepository(session)
    agents.get_or_create_by_name("Arjun Mehta", role="supervisor")
    session.commit()

    node = make_escalate_node(session)
    result = node(
        {
            "ticket_id": ticket.id,
            "summary": "Ambiguous case",
            "cited_clause_ids": [],
            "faithfulness_pass": False,
            "faithfulness_reason": "Cited clause not found",
        }
    )

    assert result["escalated"] is True
    assert result["escalation_id"] is not None


def test_present_node_creates_non_escalated_interaction(session):
    ticket = _seed_ticket(session)
    node = make_present_node(session)

    result = node(
        {
            "ticket_id": ticket.id,
            "summary": "Answered directly",
            "cited_clause_ids": ["OD-4.2"],
            "faithfulness_pass": True,
        }
    )

    assert result["escalated"] is False


def test_escalate_node_high_value_claim_gets_high_priority_and_supervisor(session):
    ticket = _seed_ticket(session)
    agents = AgentRepository(session)
    agents.get_or_create_by_name("Arjun Mehta", role="supervisor")
    session.commit()

    node = make_escalate_node(session)
    result = node(
        {
            "ticket_id": ticket.id,
            "summary": "Large claim",
            "cited_clause_ids": [],
            "faithfulness_pass": True,
            "claimed_amount": 250_000,
        }
    )

    assert result["escalated"] is True
    assert "High-value claim" in result["escalation_reason"]

    from customer_support_agent.models import Escalation

    escalation = session.query(Escalation).filter_by(id=result["escalation_id"]).one()
    assert escalation.priority.value == "high"
    assert escalation.assigned_agent.name == "Arjun Mehta"


def test_escalate_node_repeat_customer_reason_text(session):
    ticket = _seed_ticket(session)
    node = make_escalate_node(session)

    result = node(
        {
            "ticket_id": ticket.id,
            "summary": "Another ticket this month",
            "cited_clause_ids": [],
            "faithfulness_pass": True,
            "category": "coverage_question",
            "customer_context": {"has_active_policy": True, "tickets_last_30_days": 4},
        }
    )

    assert result["escalated"] is True
    assert "Repeat customer" in result["escalation_reason"]


def test_escalate_node_assigns_supervisor_even_at_medium_priority(session):
    """The fix: a MEDIUM-priority escalation (repeat customer, not a
    faithfulness failure or high-value claim) should still be assigned to
    the supervisor, not left unassigned."""
    ticket = _seed_ticket(session)
    agents = AgentRepository(session)
    agents.get_or_create_by_name("Arjun Mehta", role="supervisor")
    session.commit()

    node = make_escalate_node(session)
    result = node(
        {
            "ticket_id": ticket.id,
            "summary": "Repeat customer case",
            "cited_clause_ids": [],
            "faithfulness_pass": True,  # not a faithfulness failure
            "category": "coverage_question",
            "customer_context": {"has_active_policy": True, "tickets_last_30_days": 5},
        }
    )

    from customer_support_agent.models import Escalation, EscalationPriority

    escalation = session.query(Escalation).filter_by(id=result["escalation_id"]).one()
    assert escalation.priority == EscalationPriority.MEDIUM
    assert escalation.assigned_agent.name == "Arjun Mehta"
