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
