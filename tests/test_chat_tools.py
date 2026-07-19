"""Tests for graph/chat_tools.py -- the customer chat assistant's tools."""

from datetime import date, timedelta

from customer_support_agent.graph.chat_tools import make_chat_tools
from customer_support_agent.models import MessageSender, PolicyStatus
from customer_support_agent.repositories import (
    CustomerPolicyRepository,
    CustomerRepository,
    PolicyDocumentRepository,
    TicketMessageRepository,
    TicketRepository,
)


def _seed_customer_with_policy(session):
    customer = CustomerRepository(session).create(name="Rohan Sharma", contact_no="+91-9800000111")
    doc = PolicyDocumentRepository(session).create(
        product_type="motor_comprehensive", version="v3.2", source_file="x.md"
    )
    policy = CustomerPolicyRepository(session).create(
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
    return customer, policy


def test_get_my_policy_info_returns_active_policy(session):
    customer, policy = _seed_customer_with_policy(session)
    tools = make_chat_tools(session, customer.id)
    get_my_policy_info = next(t for t in tools if t.name == "get_my_policy_info")

    result = get_my_policy_info.invoke({})

    assert result["has_policy"] is True
    assert result["policy_number"] == "POL-MC-000777"
    assert result["status"] == "active"


def test_get_my_policy_info_no_policy(session):
    customer = CustomerRepository(session).create(name="No Policy Yet", contact_no="+91-9000000222")
    session.commit()
    tools = make_chat_tools(session, customer.id)
    get_my_policy_info = next(t for t in tools if t.name == "get_my_policy_info")

    result = get_my_policy_info.invoke({})

    assert result["has_policy"] is False


def test_list_my_tickets_scoped_to_customer(session):
    customer, policy = _seed_customer_with_policy(session)
    other_customer = CustomerRepository(session).create(
        name="Someone Else", contact_no="+91-9000000333"
    )
    session.commit()

    tickets_repo = TicketRepository(session)
    tickets_repo.create(customer_id=customer.id, customer_policy_id=policy.id)
    tickets_repo.create(customer_id=other_customer.id)
    session.commit()

    tools = make_chat_tools(session, customer.id)
    list_my_tickets = next(t for t in tools if t.name == "list_my_tickets")

    result = list_my_tickets.invoke({})

    assert len(result) == 1  # only this customer's ticket, not the other customer's


def test_create_support_ticket_creates_ticket_and_message(session):
    customer, _ = _seed_customer_with_policy(session)

    class FakeGraph:
        def invoke(self, state, config=None):
            return {"category": "coverage_question", "escalated": False}

    import customer_support_agent.graph.chat_tools as chat_tools_module

    original = chat_tools_module.build_graph
    chat_tools_module.build_graph = lambda session: FakeGraph()
    try:
        tools = make_chat_tools(session, customer.id)
        create_support_ticket = next(t for t in tools if t.name == "create_support_ticket")
        result = create_support_ticket.invoke({"issue_description": "My windscreen cracked."})
    finally:
        chat_tools_module.build_graph = original

    assert result["status"] == "open"
    ticket_id = result["ticket_id"]

    thread = TicketMessageRepository(session).get_thread(ticket_id)
    assert len(thread) == 1
    assert thread[0].sender == MessageSender.CUSTOMER
    assert thread[0].text == "My windscreen cracked."


def test_create_support_ticket_succeeds_even_if_auto_processing_fails(session):
    """Matches the same resilience pattern as the Customer Portal's 'New
    Ticket' tab: if auto-processing fails, ticket creation itself must
    still succeed."""
    customer, _ = _seed_customer_with_policy(session)

    class FailingGraph:
        def invoke(self, state, config=None):
            raise RuntimeError("simulated upstream failure")

    import customer_support_agent.graph.chat_tools as chat_tools_module

    original = chat_tools_module.build_graph
    chat_tools_module.build_graph = lambda session: FailingGraph()
    try:
        tools = make_chat_tools(session, customer.id)
        create_support_ticket = next(t for t in tools if t.name == "create_support_ticket")
        result = create_support_ticket.invoke({"issue_description": "Test issue"})
    finally:
        chat_tools_module.build_graph = original

    assert result["status"] == "open"
    assert "ticket_id" in result
