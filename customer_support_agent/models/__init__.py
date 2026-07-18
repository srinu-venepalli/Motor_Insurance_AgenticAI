"""ORM models package.

Import every model module here so that string-based relationship() targets
resolve correctly, and so `Base.metadata.create_all(engine)` sees every
table. Always `import customer_support_agent.models` (or import from this
package) before creating tables -- importing a single model file in
isolation will leave the others unregistered on Base.metadata.
"""

from customer_support_agent.models.base import Base
from customer_support_agent.models.agent import Agent
from customer_support_agent.models.customer import Customer
from customer_support_agent.models.customer_policy import CustomerPolicy
from customer_support_agent.models.escalation import Escalation
from customer_support_agent.models.feedback import Feedback
from customer_support_agent.models.interaction import Interaction
from customer_support_agent.models.policy_document import PolicyDocument
from customer_support_agent.models.ticket import Ticket
from customer_support_agent.models.ticket_message import TicketMessage
from customer_support_agent.models.enums import (
    EscalationPriority,
    EscalationStatus,
    MessageSender,
    PolicyStatus,
    TicketCategory,
    TicketResolution,
    TicketStatus,
)

__all__ = [
    "Base",
    "Agent",
    "Customer",
    "CustomerPolicy",
    "Escalation",
    "Feedback",
    "Interaction",
    "PolicyDocument",
    "Ticket",
    "TicketMessage",
    "EscalationPriority",
    "EscalationStatus",
    "MessageSender",
    "PolicyStatus",
    "TicketCategory",
    "TicketResolution",
    "TicketStatus",
]
