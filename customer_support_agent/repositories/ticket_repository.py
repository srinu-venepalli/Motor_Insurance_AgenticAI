"""Ticket repository."""

from datetime import datetime, timezone

from sqlalchemy import func, select

from customer_support_agent.models import Ticket, TicketCategory, TicketResolution, TicketStatus
from customer_support_agent.repositories.base import BaseRepository


class TicketRepository(BaseRepository[Ticket]):
    model = Ticket

    def create(
        self,
        customer_id: int,
        category: TicketCategory = TicketCategory.OTHER,
        customer_policy_id: int | None = None,
        opened_at: datetime | None = None,
    ) -> Ticket:
        return self.add(
            Ticket(
                customer_id=customer_id,
                customer_policy_id=customer_policy_id,
                category=category,
                status=TicketStatus.OPEN,
                opened_at=opened_at or datetime.now(timezone.utc),
            )
        )

    def list_for_customer(
        self,
        customer_id: int,
        status: TicketStatus | None = None,
        resolution: TicketResolution | None = None,
    ) -> list[Ticket]:
        stmt = select(Ticket).where(Ticket.customer_id == customer_id)
        if status is not None:
            stmt = stmt.where(Ticket.status == status)
        if resolution is not None:
            stmt = stmt.where(Ticket.resolution == resolution)
        stmt = stmt.order_by(Ticket.opened_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    def update_status(self, ticket_id: int, status: TicketStatus) -> Ticket | None:
        ticket = self.get(ticket_id)
        if ticket is None:
            return None
        ticket.status = status
        if status == TicketStatus.CLOSED:
            ticket.closed_at = datetime.now(timezone.utc)
        self.session.flush()
        return ticket

    def count_recent_for_customer(self, customer_id: int, since: datetime) -> int:
        """How many tickets this customer has opened since `since` --
        backs the 'repeat customer' escalation rule (more than N tickets in
        a rolling window should route to a human, not another automated
        pass)."""
        stmt = (
            select(func.count())
            .select_from(Ticket)
            .where(Ticket.customer_id == customer_id, Ticket.opened_at >= since)
        )
        return self.session.execute(stmt).scalar_one()

    def list_all(
        self,
        status: TicketStatus | None = None,
        resolution: TicketResolution | None = None,
        limit: int = 100,
    ) -> list[Ticket]:
        """All tickets across all customers, most recent first -- backs the
        agent console's queue view (unlike list_for_customer, which is
        scoped to one customer for the customer-facing view)."""
        stmt = select(Ticket).order_by(Ticket.opened_at.desc()).limit(limit)
        if status is not None:
            stmt = stmt.where(Ticket.status == status)
        if resolution is not None:
            stmt = stmt.where(Ticket.resolution == resolution)
        return list(self.session.execute(stmt).scalars().all())
