"""Customer table.

Holds real PII (name, contact details). This is the system-of-record table;
the "must not store personal data in logs" safety requirement is enforced at
the logging/tracing boundary (see core/observability.py), not by hiding data
here -- the agent needs the real name to draft a reply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from customer_support_agent.models.base import Base, IntPKMixin, TimestampMixin

if TYPE_CHECKING:
    from customer_support_agent.models.customer_policy import CustomerPolicy
    from customer_support_agent.models.ticket import Ticket


class Customer(Base, IntPKMixin, TimestampMixin):
    __tablename__ = "customers"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    contact_no: Mapped[str] = mapped_column(String(20), nullable=False)
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    policies: Mapped[list["CustomerPolicy"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )
    tickets: Mapped[list["Ticket"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"Customer(id={self.id}, name={self.name!r})"
