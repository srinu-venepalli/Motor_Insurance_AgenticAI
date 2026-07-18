"""Interaction table -- the AI's reasoning/audit trail for one ticket.

This is the core explainability artifact: what the agent retrieved, what it
cited, whether the faithfulness check passed, and whether/why it escalated.
It also doubles as the raw material customer_history_lookup reads from for
memory -- it is not a "memory table" on its own, memory is a query pattern
over this table (see integrations/memory.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from customer_support_agent.models.base import Base, IntPKMixin, TimestampMixin

if TYPE_CHECKING:
    from customer_support_agent.models.escalation import Escalation
    from customer_support_agent.models.feedback import Feedback
    from customer_support_agent.models.ticket import Ticket


class Interaction(Base, IntPKMixin, TimestampMixin):
    __tablename__ = "interactions"

    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), nullable=False)

    summary: Mapped[str] = mapped_column(Text, nullable=False)
    # List of {"clause_id": ..., "source_file": ..., "text": ...} dicts.
    cited_clauses: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    faithfulness_pass: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    escalation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    ticket: Mapped["Ticket"] = relationship(back_populates="interactions")
    escalation: Mapped["Escalation | None"] = relationship(
        back_populates="interaction", uselist=False, cascade="all, delete-orphan"
    )
    feedback_entries: Mapped[list["Feedback"]] = relationship(
        back_populates="interaction", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"Interaction(id={self.id}, ticket_id={self.ticket_id}, escalated={self.escalated})"
