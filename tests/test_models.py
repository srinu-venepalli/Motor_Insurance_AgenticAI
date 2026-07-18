"""Schema smoke tests.

Runs against an in-memory SQLite DB so this needs no Docker/Postgres to
execute -- it verifies the models/relationships are wired correctly. The
real app should point DATABASE_URL at the docker-compose Postgres instance;
swapping the engine URL is the only difference (see core/settings.py).
"""

from datetime import date, datetime, timedelta, timezone

from sqlalchemy.orm import Session

from customer_support_agent.models import (
    Agent,
    Customer,
    CustomerPolicy,
    Escalation,
    EscalationPriority,
    EscalationStatus,
    Feedback,
    Interaction,
    MessageSender,
    PolicyDocument,
    PolicyStatus,
    Ticket,
    TicketCategory,
    TicketMessage,
    TicketStatus,
)


def _make_full_ticket(session: Session):
    """Build one customer, one active policy, one ticket with a full
    interaction -> escalation -> feedback chain, and return the pieces."""
    customer = Customer(name="Rohan Sharma", contact_no="+91-9876543210")
    policy_doc = PolicyDocument(
        product_type="motor_comprehensive",
        version="v3.2",
        source_file="knowledge_base/motor_comprehensive_v3_2.pdf",
        ingested_at=datetime.now(timezone.utc),
    )
    session.add_all([customer, policy_doc])
    session.flush()  # assign ids without committing

    policy = CustomerPolicy(
        customer_id=customer.id,
        policy_document_id=policy_doc.id,
        policy_number="POL-MC-000123",
        vehicle_reg_no="TS09AB1234",
        start_date=date.today() - timedelta(days=30),
        expiry_date=date.today() + timedelta(days=335),
        premium_amount="12500.00",
        status=PolicyStatus.ACTIVE,
    )
    session.add(policy)
    session.flush()

    ticket = Ticket(
        customer_id=customer.id,
        customer_policy_id=policy.id,
        category=TicketCategory.COVERAGE_QUESTION,
        status=TicketStatus.OPEN,
        opened_at=datetime.now(timezone.utc),
    )
    session.add(ticket)
    session.flush()

    session.add(
        TicketMessage(
            ticket_id=ticket.id,
            sender=MessageSender.CUSTOMER,
            text="My car was hit while parked, does my policy cover the repair?",
        )
    )

    interaction = Interaction(
        ticket_id=ticket.id,
        summary="Customer asks about own-damage cover for a parked-car collision.",
        cited_clauses=[
            {
                "clause_id": "OD-4.2",
                "source_file": "motor_comprehensive_v3_2.pdf",
                "text": "Own damage cover applies to collision while parked, subject to excess.",
            }
        ],
        faithfulness_pass=True,
        escalated=False,
    )
    session.add(interaction)
    session.flush()

    return customer, policy, ticket, interaction


def test_customer_policy_is_valid_today(session):
    _, policy, _, _ = _make_full_ticket(session)
    assert policy.is_valid_on(date.today()) is True


def test_expired_policy_is_not_valid(session):
    customer = Customer(name="Anita Rao", contact_no="+91-9000011111")
    policy_doc = PolicyDocument(
        product_type="motor_comprehensive", version="v3.1", source_file="v3_1.pdf"
    )
    session.add_all([customer, policy_doc])
    session.flush()

    expired_policy = CustomerPolicy(
        customer_id=customer.id,
        policy_document_id=policy_doc.id,
        policy_number="POL-MC-000999",
        vehicle_reg_no="TS10CD5678",
        start_date=date.today() - timedelta(days=400),
        expiry_date=date.today() - timedelta(days=35),
        premium_amount="9800.00",
        status=PolicyStatus.EXPIRED,
    )
    session.add(expired_policy)
    session.flush()

    assert expired_policy.is_valid_on(date.today()) is False


def test_ticket_relationships_and_cited_clauses_json(session):
    customer, policy, ticket, interaction = _make_full_ticket(session)

    session.commit()
    session.expire_all()

    fetched_ticket = session.get(Ticket, ticket.id)
    assert fetched_ticket.customer.name == "Rohan Sharma"
    assert fetched_ticket.customer_policy.policy_number == "POL-MC-000123"
    assert len(fetched_ticket.messages) == 1
    assert fetched_ticket.messages[0].sender == MessageSender.CUSTOMER

    fetched_interaction = fetched_ticket.interactions[0]
    assert fetched_interaction.faithfulness_pass is True
    assert fetched_interaction.cited_clauses[0]["clause_id"] == "OD-4.2"


def test_escalation_lifecycle_and_feedback(session):
    customer, policy, ticket, interaction = _make_full_ticket(session)

    agent = Agent(name="Priya Menon", role="senior_support_agent")
    session.add(agent)
    session.flush()

    interaction.escalated = True
    interaction.escalation_reason = "Ambiguous accident description, multiple clauses could apply."

    escalation = Escalation(
        interaction_id=interaction.id,
        assigned_agent_id=agent.id,
        reason=interaction.escalation_reason,
        priority=EscalationPriority.HIGH,
        status=EscalationStatus.ASSIGNED,
    )
    session.add(escalation)

    feedback = Feedback(
        interaction_id=interaction.id,
        edit_distance=12,
        rating=4,
        notes="Draft was accurate but escalation priority should have been critical.",
    )
    session.add(feedback)

    session.commit()
    session.expire_all()

    fetched_interaction = session.get(Interaction, interaction.id)
    assert fetched_interaction.escalated is True
    assert fetched_interaction.escalation.status == EscalationStatus.ASSIGNED
    assert fetched_interaction.escalation.assigned_agent.name == "Priya Menon"
    assert fetched_interaction.feedback_entries[0].rating == 4


def test_customer_cascade_delete_removes_tickets(session):
    customer, _, ticket, _ = _make_full_ticket(session)
    session.commit()

    ticket_id = ticket.id
    session.delete(session.get(Customer, customer.id))
    session.commit()

    assert session.get(Ticket, ticket_id) is None
