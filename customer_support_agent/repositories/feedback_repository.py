"""Feedback repository -- Phase 7 adaptation evidence."""

from sqlalchemy import select

from customer_support_agent.models import Feedback
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
