"""Ticket message table -- individual turns within a ticket's conversation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from customer_support_agent.models.base import Base, IntPKMixin, TimestampMixin
from customer_support_agent.models.enums import MessageSender

if TYPE_CHECKING:
    from customer_support_agent.models.ticket import Ticket


class TicketMessage(Base, IntPKMixin, TimestampMixin):
    __tablename__ = "ticket_messages"

    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), nullable=False)
    sender: Mapped[MessageSender] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    ticket: Mapped["Ticket"] = relationship(back_populates="messages")

    def __repr__(self) -> str:  # pragma: no cover
        return f"TicketMessage(id={self.id}, ticket_id={self.ticket_id}, sender={self.sender})"
