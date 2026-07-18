"""Feedback table -- human-agent feedback on an AI interaction.

Feeds the Phase 7 "adaptive behaviour" evidence: edit_distance and rating
give a before/after signal you can use to justify a prompt or guardrail
change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from customer_support_agent.models.base import Base, IntPKMixin, TimestampMixin

if TYPE_CHECKING:
    from customer_support_agent.models.interaction import Interaction


class Feedback(Base, IntPKMixin, TimestampMixin):
    __tablename__ = "feedback"

    interaction_id: Mapped[int] = mapped_column(ForeignKey("interactions.id"), nullable=False)

    edit_distance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # e.g. 1-5
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    interaction: Mapped["Interaction"] = relationship(back_populates="feedback_entries")

    def __repr__(self) -> str:  # pragma: no cover
        return f"Feedback(id={self.id}, interaction_id={self.interaction_id}, rating={self.rating})"
