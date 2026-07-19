"""Interaction repository.

This table is the AI's reasoning/audit trail -- get_recent_for_customer is
the "memory" query pattern the customer_history_lookup tool should call: it
is not a separate memory table, just a query joined through Ticket, filtered
by customer_id, most recent first.
"""

from typing import Any

from sqlalchemy import select

from customer_support_agent.models import Interaction, Ticket
from customer_support_agent.repositories.base import BaseRepository


class InteractionRepository(BaseRepository[Interaction]):
    model = Interaction

    def create(
        self,
        ticket_id: int,
        summary: str,
        cited_clauses: list[dict[str, Any]] | None = None,
        faithfulness_pass: bool = False,
        escalated: bool = False,
        escalation_reason: str | None = None,
    ) -> Interaction:
        return self.add(
            Interaction(
                ticket_id=ticket_id,
                summary=summary,
                cited_clauses=cited_clauses or [],
                faithfulness_pass=faithfulness_pass,
                escalated=escalated,
                escalation_reason=escalation_reason,
            )
        )

    def get_for_ticket(self, ticket_id: int) -> list[Interaction]:
        stmt = (
            select(Interaction)
            .where(Interaction.ticket_id == ticket_id)
            .order_by(Interaction.created_at.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_recent_for_customer(self, customer_id: int, limit: int = 5) -> list[Interaction]:
        """The 'does this customer have prior history' lookup.

        Joins through Ticket since Interaction only stores ticket_id, not
        customer_id directly (avoids duplicating that FK on every row).
        """
        stmt = (
            select(Interaction)
            .join(Ticket, Interaction.ticket_id == Ticket.id)
            .where(Ticket.customer_id == customer_id)
            .order_by(Interaction.created_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_latest_for_tickets(self, ticket_ids: list[int]) -> dict[int, Interaction]:
        """Batch fetch -- one query for N ticket ids, instead of one
        get_for_ticket() call per ticket. Used by list_tickets() to avoid
        an N+1 query per ticket in the queue/history views."""
        if not ticket_ids:
            return {}
        stmt = (
            select(Interaction)
            .where(Interaction.ticket_id.in_(ticket_ids))
            .order_by(Interaction.created_at.asc())
        )
        latest_by_ticket: dict[int, Interaction] = {}
        for interaction in self.session.execute(stmt).scalars().all():
            latest_by_ticket[interaction.ticket_id] = interaction  # ascending order -> last write wins
        return latest_by_ticket
