"""Ticket message repository -- individual turns within a ticket's thread."""

from sqlalchemy import select

from customer_support_agent.models import MessageSender, TicketMessage
from customer_support_agent.repositories.base import BaseRepository


class TicketMessageRepository(BaseRepository[TicketMessage]):
    model = TicketMessage

    def add_message(self, ticket_id: int, sender: MessageSender, text: str) -> TicketMessage:
        return self.add(TicketMessage(ticket_id=ticket_id, sender=sender, text=text))

    def get_thread(self, ticket_id: int) -> list[TicketMessage]:
        stmt = (
            select(TicketMessage)
            .where(TicketMessage.ticket_id == ticket_id)
            .order_by(TicketMessage.created_at.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_threads_for_tickets(self, ticket_ids: list[int]) -> dict[int, list[TicketMessage]]:
        """Batch fetch -- one query for N ticket ids, instead of one
        get_thread() call per ticket. Used by list_tickets() to avoid an
        N+1 query per ticket in the queue/history views."""
        if not ticket_ids:
            return {}
        stmt = (
            select(TicketMessage)
            .where(TicketMessage.ticket_id.in_(ticket_ids))
            .order_by(TicketMessage.created_at.asc())
        )
        threads_by_ticket: dict[int, list[TicketMessage]] = {}
        for message in self.session.execute(stmt).scalars().all():
            threads_by_ticket.setdefault(message.ticket_id, []).append(message)
        return threads_by_ticket
