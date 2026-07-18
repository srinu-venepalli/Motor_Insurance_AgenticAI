"""Repositories package -- one class per table, each wrapping a SQLAlchemy
Session passed in by the caller (a FastAPI dependency in the API layer, or
the tests/conftest.py fixture)."""

from customer_support_agent.repositories.agent_repository import AgentRepository
from customer_support_agent.repositories.base import BaseRepository
from customer_support_agent.repositories.customer_policy_repository import (
    CustomerPolicyRepository,
)
from customer_support_agent.repositories.customer_repository import CustomerRepository
from customer_support_agent.repositories.escalation_repository import EscalationRepository
from customer_support_agent.repositories.feedback_repository import FeedbackRepository
from customer_support_agent.repositories.interaction_repository import InteractionRepository
from customer_support_agent.repositories.policy_document_repository import (
    PolicyDocumentRepository,
)
from customer_support_agent.repositories.ticket_message_repository import (
    TicketMessageRepository,
)
from customer_support_agent.repositories.ticket_repository import TicketRepository

__all__ = [
    "BaseRepository",
    "AgentRepository",
    "CustomerRepository",
    "CustomerPolicyRepository",
    "EscalationRepository",
    "FeedbackRepository",
    "InteractionRepository",
    "PolicyDocumentRepository",
    "TicketRepository",
    "TicketMessageRepository",
]
