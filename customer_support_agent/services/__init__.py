"""Services package -- business-logic helpers that sit above the
repositories (seeding, and later ticket-lifecycle orchestration)."""

from customer_support_agent.services.seed_data import AGENT_NAMES, AGENT_ROLES

__all__ = ["AGENT_NAMES", "AGENT_ROLES"]
