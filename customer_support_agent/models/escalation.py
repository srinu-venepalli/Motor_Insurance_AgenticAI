"""Escalation table -- workflow lifecycle for a case flagged for a human.

Kept separate from Interaction.escalated/escalation_reason: those two columns
say *that* and *why* the AI flagged it; this table tracks the human-side
workflow of resolving it (assignment, priority, status).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from customer_support_agent.models.base import Base, IntPKMixin, TimestampMixin
from customer_support_agent.models.enums import EscalationPriority, EscalationStatus

if TYPE_CHECKING:
    from customer_support_agent.models.agent import Agent
    from customer_support_agent.models.interaction import Interaction


class Escalation(Base, IntPKMixin, TimestampMixin):
    __tablename__ = "escalations"

    interaction_id: Mapped[int] = mapped_column(
        ForeignKey("interactions.id"), unique=True, nullable=False
    )
    assigned_agent_id: Mapped[int | None] = mapped_column(ForeignKey("agents.id"), nullable=True)

    reason: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[EscalationPriority] = mapped_column(
        default=EscalationPriority.MEDIUM, nullable=False
    )
    status: Mapped[EscalationStatus] = mapped_column(
        default=EscalationStatus.OPEN, nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    interaction: Mapped["Interaction"] = relationship(back_populates="escalation")
    assigned_agent: Mapped["Agent | None"] = relationship(back_populates="escalations")

    def __repr__(self) -> str:  # pragma: no cover
        return f"Escalation(id={self.id}, priority={self.priority}, status={self.status})"
