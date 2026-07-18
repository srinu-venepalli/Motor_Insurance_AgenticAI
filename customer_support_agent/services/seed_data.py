"""Synthetic seed data: 20 customers with policies, for local dev/demo/testing.

Not real customer data -- names and numbers are fictional. This populates
enough volume to meaningfully exercise the memory-lookup and policy-validity
repository methods, and to give the Streamlit UI / demo script real records
to work against.

Kept as a plain function (not a script) so it can be called both from
scripts/seed_customers.py (against real Postgres) and from a test (against
in-memory SQLite) without duplicating the logic.
"""

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy.orm import Session

from customer_support_agent.core.observability import log_transaction
from customer_support_agent.models import Agent, Customer, PolicyStatus
from customer_support_agent.repositories import (
    AgentRepository,
    CustomerPolicyRepository,
    CustomerRepository,
    PolicyDocumentRepository,
)

DEFAULT_SEED = 42

_CUSTOMER_NAMES = [
    "Rohan Sharma", "Anita Rao", "Priya Menon", "Vikram Singh", "Sneha Iyer",
    "Arjun Nair", "Kavya Reddy", "Rahul Gupta", "Divya Krishnan", "Karthik Subramaniam",
    "Neha Joshi", "Aditya Kumar", "Pooja Desai", "Suresh Pillai", "Meera Bhatt",
    "Sanjay Verma", "Lakshmi Narayan", "Amit Choudhary", "Ritu Malhotra", "Vivek Rao",
]
assert len(_CUSTOMER_NAMES) == 20

# Human support agents -- referenced by Escalation.assigned_agent_id.
# A mix of roles so escalation routing/priority logic has someone realistic
# to assign to (a supervisor for critical cases, support agents for the rest).
_AGENT_SPECS: list[tuple[str, str]] = [
    ("Priya Menon", "senior_support_agent"),
    ("Karthik Iyer", "support_agent"),
    ("Fatima Sheikh", "support_agent"),
    ("Arjun Mehta", "supervisor"),
    ("Divya Nair", "support_agent"),
]

# Public export so callers (e.g. the Streamlit UI's login dropdown) don't
# need to duplicate this list or reach into a private module attribute.
AGENT_NAMES: list[str] = [name for name, _ in _AGENT_SPECS]

# Name -> role, so the UI can gate escalated-ticket visibility to the
# supervisor without a DB round-trip at login time.
AGENT_ROLES: dict[str, str] = dict(_AGENT_SPECS)

# Distribution across 20 customers: 14 active, 3 expired, 2 lapsed, 1 cancelled.
# Deliberately not all-active, so the policy-validity check has real
# negative cases to demonstrate (e.g. the lapsed-policy failure mode from
# the Problem Framing Document's known failure cases).
_STATUS_PLAN = (
    [PolicyStatus.ACTIVE] * 14
    + [PolicyStatus.EXPIRED] * 3
    + [PolicyStatus.LAPSED] * 2
    + [PolicyStatus.CANCELLED] * 1
)
assert len(_STATUS_PLAN) == 20


@dataclass(frozen=True)
class PolicyDocSpec:
    product_type: str
    version: str
    source_file: str


COMPREHENSIVE_DOC = PolicyDocSpec(
    product_type="motor_comprehensive",
    version="v3.2",
    source_file="knowledge_base/motor_comprehensive_policy_v3_2.md",
)
THIRD_PARTY_DOC = PolicyDocSpec(
    product_type="motor_third_party",
    version="v1.0",
    source_file="knowledge_base/motor_third_party_policy_v1_0.md",
)


def _get_or_create_policy_doc(session: Session, spec: PolicyDocSpec):
    repo = PolicyDocumentRepository(session)
    existing = repo.get_by_product_and_version(spec.product_type, spec.version)
    if existing:
        return existing
    doc = repo.create(
        product_type=spec.product_type,
        version=spec.version,
        source_file=spec.source_file,
        ingested_at=datetime.now(timezone.utc),
    )
    session.flush()
    return doc


def _dates_for_status(status: PolicyStatus, rng: random.Random) -> tuple[date, date]:
    today = date.today()
    if status == PolicyStatus.ACTIVE:
        start = today - timedelta(days=rng.randint(1, 300))
        expiry = today + timedelta(days=rng.randint(65, 364))
    elif status == PolicyStatus.EXPIRED:
        expiry = today - timedelta(days=rng.randint(10, 60))
        start = expiry - timedelta(days=365)
    elif status == PolicyStatus.LAPSED:
        expiry = today - timedelta(days=rng.randint(20, 45))
        start = expiry - timedelta(days=365)
    else:  # CANCELLED
        start = today - timedelta(days=rng.randint(30, 200))
        expiry = start + timedelta(days=365)
    return start, expiry


@dataclass(frozen=True)
class CustomerPlan:
    """Everything needed to create one customer + policy, computed purely
    from (index, seed) with no DB access -- this is what makes seeding
    deterministic and testable without a database round-trip."""

    name: str
    contact_no: str
    contact_email: str
    status: PolicyStatus
    use_comprehensive: bool
    policy_number: str
    vehicle_reg_no: str
    start_date: date
    expiry_date: date
    premium_amount: Decimal


def _plan_for_index(i: int, rng: random.Random) -> CustomerPlan:
    name = _CUSTOMER_NAMES[i]
    status = _STATUS_PLAN[i]
    contact_no = f"+91-{9800000001 + i}"
    contact_email = name.lower().replace(" ", ".") + "@example.com"

    use_comprehensive = rng.random() < 0.75  # ~75% comprehensive, ~25% third-party
    product_prefix = "MC" if use_comprehensive else "TP"

    start, expiry = _dates_for_status(status, rng)
    policy_number = f"POL-{product_prefix}-{100 + i:06d}"
    vehicle_reg_no = (
        f"TS{rng.randint(1, 99):02d}"
        f"{rng.choice('ABCDEFGH')}{rng.choice('ABCDEFGH')}"
        f"{rng.randint(1000, 9999)}"
    )
    premium_amount = Decimal(rng.randint(8000, 18000))

    return CustomerPlan(
        name=name,
        contact_no=contact_no,
        contact_email=contact_email,
        status=status,
        use_comprehensive=use_comprehensive,
        policy_number=policy_number,
        vehicle_reg_no=vehicle_reg_no,
        start_date=start,
        expiry_date=expiry,
        premium_amount=premium_amount,
    )


def generate_plan(count: int, seed: int = DEFAULT_SEED) -> list[CustomerPlan]:
    """Pure, DB-free: same (count, seed) always produces the same list of
    CustomerPlan objects. Used both by seed_customers() below and directly
    by tests to verify determinism without needing a database at all."""
    if count > len(_CUSTOMER_NAMES):
        raise ValueError(
            f"Only {len(_CUSTOMER_NAMES)} canned names available, got count={count}"
        )
    rng = random.Random(seed)
    return [_plan_for_index(i, rng) for i in range(count)]


def seed_agents(session: Session) -> list[Agent]:
    """Create the canned set of human support agents, if they don't already
    exist (idempotent via AgentRepository.get_or_create_by_name -- safe to
    call this every run, unlike seed_customers which is not idempotent)."""
    repo = AgentRepository(session)
    created: list[Agent] = []
    with log_transaction("seed_agents", count=len(_AGENT_SPECS)):
        for name, role in _AGENT_SPECS:
            agent = repo.get_or_create_by_name(name, role=role)
            created.append(agent)
        session.flush()
    return created


def seed_customers(
    session: Session, count: int = 20, seed: int = DEFAULT_SEED
) -> list[Customer]:
    """Create `count` synthetic customers, each with one CustomerPolicy, from
    a deterministic plan (see generate_plan()).

    Does NOT check for existing data first; calling this twice will create
    duplicate customers (policy_number is unique though, so a genuine
    duplicate run will raise an IntegrityError on the second pass with the
    same seed -- that's intentional, see scripts/seed_customers.py for the
    --force guard).
    """
    plan = generate_plan(count=count, seed=seed)

    customers_repo = CustomerRepository(session)
    policies_repo = CustomerPolicyRepository(session)

    comprehensive_doc = _get_or_create_policy_doc(session, COMPREHENSIVE_DOC)
    third_party_doc = _get_or_create_policy_doc(session, THIRD_PARTY_DOC)

    created: list[Customer] = []
    with log_transaction("seed_customers", count=count, seed=seed):
        for item in plan:
            customer = customers_repo.create(
                name=item.name, contact_no=item.contact_no, contact_email=item.contact_email
            )
            session.flush()

            doc = comprehensive_doc if item.use_comprehensive else third_party_doc
            policies_repo.create(
                customer_id=customer.id,
                policy_document_id=doc.id,
                policy_number=item.policy_number,
                vehicle_reg_no=item.vehicle_reg_no,
                start_date=item.start_date,
                expiry_date=item.expiry_date,
                premium_amount=item.premium_amount,
                status=item.status,
            )
            created.append(customer)

        session.flush()

    return created
