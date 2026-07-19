"""Repository smoke tests.

Runs against the same conftest.py session fixture as test_models.py, so it
defaults to in-memory SQLite and can be pointed at real Postgres via:

    TEST_DATABASE_URL="postgresql+psycopg2://agent:localdev@localhost:5432/insurance_support" \
        uv run pytest tests/test_repositories.py -v
"""

from datetime import date, datetime, timedelta, timezone

from customer_support_agent.models import (
    EscalationPriority,
    EscalationStatus,
    MessageSender,
    PolicyStatus,
    TicketCategory,
    TicketStatus,
)
from customer_support_agent.repositories import (
    AgentRepository,
    CustomerPolicyRepository,
    CustomerRepository,
    EscalationRepository,
    FeedbackRepository,
    InteractionRepository,
    PolicyDocumentRepository,
    TicketMessageRepository,
    TicketRepository,
)


def _seed_customer_with_active_policy(session):
    customers = CustomerRepository(session)
    policy_docs = PolicyDocumentRepository(session)
    policies = CustomerPolicyRepository(session)

    customer = customers.create(name="Rohan Sharma", contact_no="+91-9876543210")
    doc = policy_docs.create(
        product_type="motor_comprehensive",
        version="v3.2",
        source_file="motor_comprehensive_v3_2.pdf",
        ingested_at=datetime.now(timezone.utc),
    )
    policy = policies.create(
        customer_id=customer.id,
        policy_document_id=doc.id,
        policy_number="POL-MC-000123",
        vehicle_reg_no="TS09AB1234",
        start_date=date.today() - timedelta(days=30),
        expiry_date=date.today() + timedelta(days=335),
        premium_amount="12500.00",
        status=PolicyStatus.ACTIVE,
    )
    session.commit()
    return customer, doc, policy


def test_customer_repository_create_and_lookup_by_contact(session):
    repo = CustomerRepository(session)
    repo.create(name="Anita Rao", contact_no="+91-9000011111")
    session.commit()

    found = repo.get_by_contact_no("+91-9000011111")
    assert found is not None
    assert found.name == "Anita Rao"
    assert repo.get_by_contact_no("+91-0000000000") is None


def test_policy_document_repository_lookup(session):
    repo = PolicyDocumentRepository(session)
    repo.create(product_type="motor_comprehensive", version="v3.2", source_file="v3_2.pdf")
    session.commit()

    found = repo.get_by_product_and_version("motor_comprehensive", "v3.2")
    assert found is not None
    assert found.source_file == "v3_2.pdf"


def test_customer_policy_repository_active_policy_lookup(session):
    customer, _, policy = _seed_customer_with_active_policy(session)

    repo = CustomerPolicyRepository(session)
    active = repo.get_active_policy_for_customer(customer.id)
    assert active is not None
    assert active.policy_number == policy.policy_number

    # A policy that expired last month should not be returned as active.
    stale_as_of = date.today() + timedelta(days=400)  # past this policy's expiry
    assert repo.get_active_policy_for_customer(customer.id, as_of=stale_as_of) is None


def test_ticket_repository_count_recent_for_customer(session):
    from datetime import datetime, timedelta, timezone

    customer, _, policy = _seed_customer_with_active_policy(session)
    tickets = TicketRepository(session)

    now = datetime.now(timezone.utc)
    # 3 tickets within the last 30 days, 1 well outside it.
    tickets.create(customer_id=customer.id, customer_policy_id=policy.id, opened_at=now - timedelta(days=1))
    tickets.create(customer_id=customer.id, customer_policy_id=policy.id, opened_at=now - timedelta(days=10))
    tickets.create(customer_id=customer.id, customer_policy_id=policy.id, opened_at=now - timedelta(days=20))
    tickets.create(customer_id=customer.id, customer_policy_id=policy.id, opened_at=now - timedelta(days=60))
    session.commit()

    count = tickets.count_recent_for_customer(customer.id, since=now - timedelta(days=30))
    assert count == 3


def test_ticket_and_message_repositories(session):
    customer, _, policy = _seed_customer_with_active_policy(session)

    tickets = TicketRepository(session)
    messages = TicketMessageRepository(session)

    ticket = tickets.create(
        customer_id=customer.id,
        customer_policy_id=policy.id,
        category=TicketCategory.COVERAGE_QUESTION,
    )
    session.commit()

    messages.add_message(ticket.id, MessageSender.CUSTOMER, "Does my policy cover this?")
    messages.add_message(ticket.id, MessageSender.AI_DRAFT, "Based on clause OD-4.2, yes.")
    session.commit()

    thread = messages.get_thread(ticket.id)
    assert [m.sender for m in thread] == [MessageSender.CUSTOMER, MessageSender.AI_DRAFT]

    open_tickets = tickets.list_for_customer(customer.id, status=TicketStatus.OPEN)
    assert len(open_tickets) == 1

    closed = tickets.update_status(ticket.id, TicketStatus.CLOSED)
    assert closed.closed_at is not None


def test_interaction_repository_memory_lookup_across_tickets(session):
    """This is the actual 'memory' behaviour: a repeat customer's second
    ticket should be able to see interactions from their first ticket."""
    customer, _, policy = _seed_customer_with_active_policy(session)

    tickets = TicketRepository(session)
    interactions = InteractionRepository(session)

    first_ticket = tickets.create(customer_id=customer.id, customer_policy_id=policy.id)
    session.commit()
    interactions.create(
        ticket_id=first_ticket.id,
        summary="Customer asked about windscreen cover.",
        cited_clauses=[{"clause_id": "GL-2.1", "text": "Glass cover applies."}],
        faithfulness_pass=True,
    )
    session.commit()

    second_ticket = tickets.create(customer_id=customer.id, customer_policy_id=policy.id)
    session.commit()
    interactions.create(
        ticket_id=second_ticket.id,
        summary="Customer asked about rental car cover.",
        cited_clauses=[{"clause_id": "RC-1.0", "text": "Rental cover applies for 5 days."}],
        faithfulness_pass=True,
    )
    session.commit()

    history = interactions.get_recent_for_customer(customer.id, limit=5)
    assert len(history) == 2
    # Most recent first.
    assert history[0].summary.startswith("Customer asked about rental")
    assert history[1].summary.startswith("Customer asked about windscreen")


def test_escalation_and_feedback_repositories(session):
    customer, _, policy = _seed_customer_with_active_policy(session)

    tickets = TicketRepository(session)
    interactions = InteractionRepository(session)
    escalations = EscalationRepository(session)
    feedback = FeedbackRepository(session)
    agents = AgentRepository(session)

    ticket = tickets.create(customer_id=customer.id, customer_policy_id=policy.id)
    session.commit()

    interaction = interactions.create(
        ticket_id=ticket.id,
        summary="Ambiguous accident description.",
        escalated=True,
        escalation_reason="Multiple clauses could apply; needs human judgement.",
    )
    session.commit()

    agent = agents.get_or_create_by_name("Priya Menon", role="senior_support_agent")
    session.commit()

    escalation = escalations.create(
        interaction_id=interaction.id,
        reason=interaction.escalation_reason,
        priority=EscalationPriority.HIGH,
        assigned_agent_id=agent.id,
    )
    session.commit()
    assert escalation.status == EscalationStatus.ASSIGNED

    open_escalations = escalations.list_open()
    assert len(open_escalations) == 1

    resolved = escalations.update_status(escalation.id, EscalationStatus.RESOLVED)
    assert resolved.resolved_at is not None
    assert escalations.list_open() == []

    feedback.create(interaction_id=interaction.id, edit_distance=5, rating=4, notes="Good catch.")
    session.commit()
    assert len(feedback.list_for_interaction(interaction.id)) == 1


def test_agent_repository_get_or_create_is_idempotent(session):
    repo = AgentRepository(session)
    first = repo.get_or_create_by_name("Priya Menon")
    session.commit()
    second = repo.get_or_create_by_name("Priya Menon")
    session.commit()
    assert first.id == second.id


def test_average_edit_distance_returns_none_below_min_samples(session):
    customer, _, policy = _seed_customer_with_active_policy(session)
    tickets = TicketRepository(session)
    interactions = InteractionRepository(session)
    feedback = FeedbackRepository(session)

    ticket = tickets.create(
        customer_id=customer.id, customer_policy_id=policy.id, category=TicketCategory.CLAIM_STATUS
    )
    session.commit()
    interaction = interactions.create(ticket_id=ticket.id, summary="x", faithfulness_pass=True)
    session.commit()
    feedback.create(interaction_id=interaction.id, edit_distance=200)
    session.commit()

    # Only 1 sample recorded, default min_samples=3 -- should be None, not
    # overreact to a single heavily-edited draft.
    avg = feedback.average_edit_distance_for_category(TicketCategory.CLAIM_STATUS, min_samples=3)
    assert avg is None


def test_average_edit_distance_computes_correctly_once_enough_samples(session):
    customer, _, policy = _seed_customer_with_active_policy(session)
    tickets = TicketRepository(session)
    interactions = InteractionRepository(session)
    feedback = FeedbackRepository(session)

    edit_distances = [100, 200, 300]
    for ed in edit_distances:
        ticket = tickets.create(
            customer_id=customer.id,
            customer_policy_id=policy.id,
            category=TicketCategory.CLAIM_STATUS,
        )
        session.commit()
        interaction = interactions.create(ticket_id=ticket.id, summary="x", faithfulness_pass=True)
        session.commit()
        feedback.create(interaction_id=interaction.id, edit_distance=ed)
        session.commit()

    avg = feedback.average_edit_distance_for_category(TicketCategory.CLAIM_STATUS, min_samples=3)
    assert avg == 200  # (100 + 200 + 300) / 3


def test_average_edit_distance_is_scoped_to_the_given_category(session):
    customer, _, policy = _seed_customer_with_active_policy(session)
    tickets = TicketRepository(session)
    interactions = InteractionRepository(session)
    feedback = FeedbackRepository(session)

    # 3 samples for CLAIM_STATUS (high edit distance)...
    for _ in range(3):
        t = tickets.create(
            customer_id=customer.id,
            customer_policy_id=policy.id,
            category=TicketCategory.CLAIM_STATUS,
        )
        session.commit()
        i = interactions.create(ticket_id=t.id, summary="x", faithfulness_pass=True)
        session.commit()
        feedback.create(interaction_id=i.id, edit_distance=500)
        session.commit()

    # ...and 3 for COVERAGE_QUESTION (low edit distance) -- must not blend.
    for _ in range(3):
        t = tickets.create(
            customer_id=customer.id,
            customer_policy_id=policy.id,
            category=TicketCategory.COVERAGE_QUESTION,
        )
        session.commit()
        i = interactions.create(ticket_id=t.id, summary="x", faithfulness_pass=True)
        session.commit()
        feedback.create(interaction_id=i.id, edit_distance=10)
        session.commit()

    assert feedback.average_edit_distance_for_category(TicketCategory.CLAIM_STATUS) == 500
    assert feedback.average_edit_distance_for_category(TicketCategory.COVERAGE_QUESTION) == 10
