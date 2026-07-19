"""Feedback repository -- Phase 7 adaptation evidence."""

from sqlalchemy import select

from customer_support_agent.models import Feedback, Interaction, Ticket, TicketCategory
from customer_support_agent.repositories.base import BaseRepository


class FeedbackRepository(BaseRepository[Feedback]):
    model = Feedback

    def create(
        self,
        interaction_id: int,
        edit_distance: int | None = None,
        rating: int | None = None,
        notes: str | None = None,
    ) -> Feedback:
        return self.add(
            Feedback(
                interaction_id=interaction_id,
                edit_distance=edit_distance,
                rating=rating,
                notes=notes,
            )
        )

    def list_for_interaction(self, interaction_id: int) -> list[Feedback]:
        stmt = select(Feedback).where(Feedback.interaction_id == interaction_id)
        return list(self.session.execute(stmt).scalars().all())

    def average_edit_distance_for_category(
        self, category: TicketCategory, min_samples: int = 3
    ) -> float | None:
        """Average edit_distance across all feedback recorded for tickets in
        this category -- the raw signal behind the Phase 7 adaptive
        escalation rule (see graph/nodes.py's _evaluate_escalation). Returns
        None if there isn't at least min_samples data points yet, so one
        early heavily-edited draft doesn't overreact and start escalating
        an entire category prematurely.
        """
        stmt = (
            select(Feedback.edit_distance)
            .join(Interaction, Feedback.interaction_id == Interaction.id)
            .join(Ticket, Interaction.ticket_id == Ticket.id)
            .where(Ticket.category == category, Feedback.edit_distance.is_not(None))
        )
        values = list(self.session.execute(stmt).scalars().all())
        if len(values) < min_samples:
            return None
        return sum(values) / len(values)
