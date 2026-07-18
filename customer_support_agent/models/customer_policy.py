"""Customer policy table.

The specific coverage instance: this customer, this vehicle, these dates.
This is what answers "is this customer's policy currently valid" -- a check
that should happen before the agent reasons about coverage clauses at all.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Date, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from customer_support_agent.models.base import Base, IntPKMixin, TimestampMixin
from customer_support_agent.models.enums import PolicyStatus

if TYPE_CHECKING:
    from customer_support_agent.models.customer import Customer
    from customer_support_agent.models.policy_document import PolicyDocument
    from customer_support_agent.models.ticket import Ticket


class CustomerPolicy(Base, IntPKMixin, TimestampMixin):
    __tablename__ = "customer_policies"

    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)
    policy_document_id: Mapped[int] = mapped_column(
        ForeignKey("policy_documents.id"), nullable=False
    )

    policy_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    vehicle_reg_no: Mapped[str] = mapped_column(String(20), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False)
    premium_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    status: Mapped[PolicyStatus] = mapped_column(
        default=PolicyStatus.ACTIVE, nullable=False
    )

    customer: Mapped["Customer"] = relationship(back_populates="policies")
    policy_document: Mapped["PolicyDocument"] = relationship(back_populates="customer_policies")
    tickets: Mapped[list["Ticket"]] = relationship(back_populates="customer_policy")

    def is_valid_on(self, as_of: date) -> bool:
        """Quick validity check the agent should run before reasoning about coverage."""
        return self.status == PolicyStatus.ACTIVE and self.start_date <= as_of <= self.expiry_date

    def __repr__(self) -> str:  # pragma: no cover
        return f"CustomerPolicy(id={self.id}, policy_number={self.policy_number!r}, status={self.status})"
