"""Tests for the synthetic seed-data generator."""

from datetime import date

from customer_support_agent.models import PolicyStatus
from customer_support_agent.repositories import CustomerPolicyRepository
from customer_support_agent.services.seed_data import seed_agents, seed_customers


def test_seed_creates_20_customers_with_one_policy_each(session):
    created = seed_customers(session, count=20)
    session.commit()

    assert len(created) == 20
    for customer in created:
        assert len(customer.policies) == 1


def test_seed_status_distribution_matches_plan(session):
    created = seed_customers(session, count=20)
    session.commit()

    statuses = [c.policies[0].status for c in created]
    assert statuses.count(PolicyStatus.ACTIVE) == 14
    assert statuses.count(PolicyStatus.EXPIRED) == 3
    assert statuses.count(PolicyStatus.LAPSED) == 2
    assert statuses.count(PolicyStatus.CANCELLED) == 1


def test_plan_generation_is_deterministic_given_same_seed():
    """Determinism is a property of generate_plan() alone -- checking it
    should never need a second database round-trip (the old version of this
    test called drop_all/create_all mid-test, which could hang against
    Postgres if another connection still held the tables open)."""
    from customer_support_agent.services.seed_data import generate_plan

    first = generate_plan(count=20, seed=123)
    second = generate_plan(count=20, seed=123)

    assert [p.policy_number for p in first] == [p.policy_number for p in second]
    assert [p.vehicle_reg_no for p in first] == [p.vehicle_reg_no for p in second]
    assert [p.start_date for p in first] == [p.start_date for p in second]


def test_plan_generation_differs_across_seeds():
    from customer_support_agent.services.seed_data import generate_plan

    a = generate_plan(count=20, seed=1)
    b = generate_plan(count=20, seed=2)
    assert [p.vehicle_reg_no for p in a] != [p.vehicle_reg_no for p in b]


def test_active_policies_are_actually_valid_today(session):
    created = seed_customers(session, count=20)
    session.commit()

    repo = CustomerPolicyRepository(session)
    for customer in created:
        policy = customer.policies[0]
        is_valid = policy.is_valid_on(date.today())
        assert is_valid == (policy.status == PolicyStatus.ACTIVE)

        # Cross-check against the repository method too.
        active = repo.get_active_policy_for_customer(customer.id)
        if policy.status == PolicyStatus.ACTIVE:
            assert active is not None
        else:
            assert active is None


def test_seed_uses_both_policy_documents(session):
    created = seed_customers(session, count=20)
    session.commit()

    product_types = {c.policies[0].policy_document.product_type for c in created}
    # With 20 customers and a 75/25 split, both products should appear.
    assert "motor_comprehensive" in product_types
    assert "motor_third_party" in product_types


def test_seed_agents_creates_expected_roles(session):
    agents = seed_agents(session)
    session.commit()

    assert len(agents) == 5
    roles = {a.role for a in agents}
    assert "supervisor" in roles
    assert "senior_support_agent" in roles
    assert "support_agent" in roles


def test_seed_agents_is_idempotent(session):
    first = seed_agents(session)
    session.commit()
    second = seed_agents(session)
    session.commit()

    assert len(first) == len(second) == 5
    assert {a.id for a in first} == {a.id for a in second}
