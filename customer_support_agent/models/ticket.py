"""Ticket table -- one row per customer support ticket."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from customer_support_agent.models.base import Base, IntPKMixin, TimestampMixin
from customer_support_agent.models.enums import TicketCategory, TicketResolution, TicketStatus

if TYPE_CHECKING:
    from customer_support_agent.models.customer import Customer
    from customer_support_agent.models.customer_policy import CustomerPolicy
    from customer_support_agent.models.interaction import Interaction
    from customer_support_agent.models.ticket_message import TicketMessage


class Ticket(Base, IntPKMixin, TimestampMixin):
    __tablename__ = "tickets"

    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)
    customer_policy_id: Mapped[int | None] = mapped_column(
        ForeignKey("customer_policies.id"), nullable=True
    )

    category: Mapped[TicketCategory] = mapped_column(default=TicketCategory.OTHER, nullable=False)
    status: Mapped[TicketStatus] = mapped_column(default=TicketStatus.OPEN, nullable=False)
    # Set only when status becomes CLOSED -- distinguishes "approved and
    # sent" from "claim rejected" for a ticket that's otherwise in the same
    # lifecycle state.
    resolution: Mapped[TicketResolution | None] = mapped_column(nullable=True)

    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    customer: Mapped["Customer"] = relationship(back_populates="tickets")
    customer_policy: Mapped["CustomerPolicy | None"] = relationship(back_populates="tickets")
    messages: Mapped[list["TicketMessage"]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan", order_by="TicketMessage.created_at"
    )
    interactions: Mapped[list["Interaction"]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"Ticket(id={self.id}, customer_id={self.customer_id}, status={self.status})"
