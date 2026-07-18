"""Enumerations shared across ORM models.

Kept in one module so services/repositories can import the same enum the
database column is constrained to, instead of comparing raw strings.
"""

import enum


class PolicyStatus(str, enum.Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    LAPSED = "lapsed"


class TicketCategory(str, enum.Enum):
    COVERAGE_QUESTION = "coverage_question"
    CLAIM_STATUS = "claim_status"
    ACCIDENT_REPORT = "accident_report"
    COMPLAINT = "complaint"
    RENEWAL = "renewal"
    OTHER = "other"


class TicketStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketResolution(str, enum.Enum):
    """Outcome of a closed ticket -- separate from TicketStatus (lifecycle
    state) since 'closed' alone doesn't say whether the claim/question was
    approved or the claim was rejected."""

    APPROVED = "approved"
    REJECTED = "rejected"


class MessageSender(str, enum.Enum):
    CUSTOMER = "customer"
    AI_DRAFT = "ai_draft"
    HUMAN_AGENT = "human_agent"


class EscalationPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationStatus(str, enum.Enum):
    OPEN = "open"
    ASSIGNED = "assigned"
    RESOLVED = "resolved"
