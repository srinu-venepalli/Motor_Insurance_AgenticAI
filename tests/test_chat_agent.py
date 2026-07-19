"""Tests for graph/chat_agent.py -- the customer chat assistant's
tool-calling loop, using a fake LLM so no real API calls happen."""

from datetime import date, timedelta

from customer_support_agent.graph.chat_agent import MAX_TOOL_ITERATIONS, run_chat_turn
from customer_support_agent.models import PolicyStatus
from customer_support_agent.repositories import (
    CustomerPolicyRepository,
    CustomerRepository,
    PolicyDocumentRepository,
)


class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.invoke_call_count = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invoke_call_count += 1
        return self._responses.pop(0)


def _seed_customer_with_policy(session):
    customer = CustomerRepository(session).create(name="Rohan Sharma", contact_no="+91-9800000111")
    doc = PolicyDocumentRepository(session).create(
        product_type="motor_comprehensive", version="v3.2", source_file="x.md"
    )
    CustomerPolicyRepository(session).create(
        customer_id=customer.id,
        policy_document_id=doc.id,
        policy_number="POL-MC-000777",
        vehicle_reg_no="TS09AB1234",
        start_date=date.today() - timedelta(days=10),
        expiry_date=date.today() + timedelta(days=355),
        premium_amount="12000.00",
        status=PolicyStatus.ACTIVE,
    )
    session.commit()
    return customer


def test_run_chat_turn_answers_directly_with_no_tool_needed(session):
    customer = _seed_customer_with_policy(session)
    llm = FakeLLM([FakeMessage(content="Hi there! How can I help?", tool_calls=[])])

    reply, history = run_chat_turn(
        session, customer.id, customer.name, history=[], user_message="Hello", llm=llm
    )

    assert reply == "Hi there! How can I help?"
    assert history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help?"},
    ]


def test_run_chat_turn_calls_policy_info_tool_and_answers(session):
    customer = _seed_customer_with_policy(session)
    llm = FakeLLM(
        [
            FakeMessage(
                tool_calls=[{"name": "get_my_policy_info", "args": {}, "id": "call_1"}]
            ),
            FakeMessage(content="Your policy POL-MC-000777 is active.", tool_calls=[]),
        ]
    )

    reply, history = run_chat_turn(
        session,
        customer.id,
        customer.name,
        history=[],
        user_message="What's my policy number?",
        llm=llm,
    )

    assert "POL-MC-000777" in reply
    assert len(history) == 2


def test_run_chat_turn_preserves_prior_history(session):
    customer = _seed_customer_with_policy(session)
    llm = FakeLLM([FakeMessage(content="Sure, anything else?", tool_calls=[])])

    prior_history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    reply, updated_history = run_chat_turn(
        session,
        customer.id,
        customer.name,
        history=prior_history,
        user_message="Thanks",
        llm=llm,
    )

    assert len(updated_history) == 4
    assert updated_history[0] == prior_history[0]
    assert updated_history[1] == prior_history[1]
    assert updated_history[2] == {"role": "user", "content": "Thanks"}
    assert updated_history[3] == {"role": "assistant", "content": "Sure, anything else?"}


def test_run_chat_turn_falls_back_gracefully_after_max_iterations(session):
    """A model that never stops calling tools should be cut off, not looped
    forever -- same safeguard principle as agent_reasoning's max_iterations."""
    customer = _seed_customer_with_policy(session)
    responses = [
        FakeMessage(tool_calls=[{"name": "get_my_policy_info", "args": {}, "id": f"c{i}"}])
        for i in range(MAX_TOOL_ITERATIONS + 2)
    ]
    llm = FakeLLM(responses)

    reply, history = run_chat_turn(
        session, customer.id, customer.name, history=[], user_message="test", llm=llm
    )

    assert llm.invoke_call_count == MAX_TOOL_ITERATIONS
    assert "trouble completing" in reply.lower() or "support ticket" in reply.lower()
