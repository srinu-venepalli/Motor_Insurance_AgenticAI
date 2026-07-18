"""Agent table -- human support agents who review/escalate/send AI drafts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from customer_support_agent.models.base import Base, IntPKMixin, TimestampMixin

if TYPE_CHECKING:
    from customer_support_agent.models.escalation import Escalation


class Agent(Base, IntPKMixin, TimestampMixin):
    __tablename__ = "agents"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(100), default="support_agent", nullable=False)

    escalations: Mapped[list["Escalation"]] = relationship(back_populates="assigned_agent")

    def __repr__(self) -> str:  # pragma: no cover
        return f"Agent(id={self.id}, name={self.name!r})"
