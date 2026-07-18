"""Pydantic request/response contracts for the API layer.

Kept separate from the SQLAlchemy ORM models (customer_support_agent.models)
-- these describe the wire format, not the database schema.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TicketCreateRequest(BaseModel):
    customer_id: int
    ticket_text: str = Field(min_length=1, description="The raw ticket text as submitted.")
    category: Optional[str] = Field(
        default=None, description="Optional pre-set category; usually left null and classified by the agent."
    )


class TicketCreateResponse(BaseModel):
    ticket_id: int
    customer_id: int
    status: str
    opened_at: datetime


class TicketProcessResponse(BaseModel):
    ticket_id: int
    category: Optional[str]
    tool_calls_made: list[str]
    retrieved_clauses_count: int
    customer_context: Optional[dict]
    claimed_amount: Optional[float]
    faithfulness_pass: bool
    faithfulness_reason: Optional[str]
    escalated: bool
    escalation_reason: Optional[str]
    summary: Optional[str]
    draft_response: Optional[str]


class InteractionSummary(BaseModel):
    interaction_id: int
    summary: str
    escalated: bool
    escalation_reason: Optional[str]
    faithfulness_pass: bool
    created_at: datetime


class MessageOut(BaseModel):
    sender: str
    text: str
    created_at: datetime


class TicketDetailResponse(BaseModel):
    ticket_id: int
    customer_id: int
    customer_name: Optional[str]
    category: str
    status: str
    resolution: Optional[str]
    opened_at: datetime
    closed_at: Optional[datetime]
    interactions: list[InteractionSummary]
    messages: list[MessageOut]


class TicketSummary(BaseModel):
    """Lightweight row for the agent console's ticket queue -- one row per
    ticket, not the full detail."""

    ticket_id: int
    customer_id: int
    customer_name: Optional[str]
    category: str
    status: str
    resolution: Optional[str]
    opened_at: datetime
    escalated: Optional[bool] = Field(
        default=None, description="From the latest interaction; null if not yet processed."
    )
    latest_summary: Optional[str] = Field(
        default=None, description="The latest interaction's summary, for the customer-history table."
    )


class TicketApproveRequest(BaseModel):
    final_response: Optional[str] = Field(
        default=None,
        description="Edited text to send. If omitted, sends the latest AI draft as-is.",
    )
    resolution: str = Field(
        default="approved",
        description="'approved' (claim/question resolved positively) or 'rejected' (claim denied).",
    )


class TicketApproveResponse(BaseModel):
    ticket_id: int
    status: str
    resolution: str
    sent_text: str
    edit_distance: Optional[int] = Field(
        default=None, description="Character-level edit distance vs the original AI draft, if one existed."
    )


class CustomerOut(BaseModel):
    id: int
    name: str
