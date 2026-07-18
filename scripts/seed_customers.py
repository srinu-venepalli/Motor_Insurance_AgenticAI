"""Seed 20 synthetic customers (with policies) and 5 human support agents
into the configured database.

Usage:
    uv run python scripts/seed_customers.py
    uv run python scripts/seed_customers.py --count 10
    uv run python scripts/seed_customers.py --force   # wipe existing customers/agents first

Reads DATABASE_URL from .env via core.settings -- point it at your
docker-compose Postgres instance before running, or leave the sqlite
default for a quick local check.
"""

import argparse

from sqlalchemy import delete

from customer_support_agent.core import get_session, settings
from customer_support_agent.models import Agent, Customer, CustomerPolicy, PolicyDocument
from customer_support_agent.services.seed_data import (
    DEFAULT_SEED,
    seed_agents,
    seed_customers,
)


def _wipe_existing(session) -> None:
    # Order matters for FK constraints: policies before customers/doc;
    # agents last since nothing seeded here references them yet, but a
    # future escalations seed would need agents to still exist at that point.
    session.execute(delete(CustomerPolicy))
    session.execute(delete(Customer))
    session.execute(delete(PolicyDocument))
    session.execute(delete(Agent))
    session.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20, help="Number of customers to create")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--force", action="store_true", help="Delete existing customers/agents/policies first"
    )
    args = parser.parse_args()

    print(f"Target database: {settings.database_url}")

    with get_session() as session:
        existing_count = session.query(Customer).count()
        if existing_count > 0 and not args.force:
            print(
                f"Found {existing_count} existing customer(s). "
                "Re-run with --force to wipe and reseed, or this run will stop here "
                "to avoid duplicate/conflicting policy numbers."
            )
            return

        if existing_count > 0 and args.force:
            print(f"--force set: wiping {existing_count} existing customer(s)...")
            _wipe_existing(session)

        agents = seed_agents(session)
        print(f"\nCreated/verified {len(agents)} human support agents:")
        print(f"{'Name':<20} {'Role':<20}")
        print("-" * 40)
        for agent in agents:
            print(f"{agent.name:<20} {agent.role:<20}")

        created = seed_customers(session, count=args.count, seed=args.seed)

        print(f"\nCreated {len(created)} customers:")
        print(f"{'Name':<20} {'Contact No':<16} {'Policy #':<16} {'Status':<10}")
        print("-" * 64)
        for customer in created:
            policy = customer.policies[0] if customer.policies else None
            policy_no = policy.policy_number if policy else "-"
            status = policy.status.value if policy else "-"
            print(f"{customer.name:<20} {customer.contact_no:<16} {policy_no:<16} {status:<10}")


if __name__ == "__main__":
    main()
