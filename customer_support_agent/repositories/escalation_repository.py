"""Escalation repository -- backs the escalate_to_agent tool."""

from datetime import datetime, timezone

from sqlalchemy import select

from customer_support_agent.models import Escalation, EscalationPriority, EscalationStatus
from customer_support_agent.repositories.base import BaseRepository


class EscalationRepository(BaseRepository[Escalation]):
    model = Escalation

    def create(
        self,
        interaction_id: int,
        reason: str,
        priority: EscalationPriority = EscalationPriority.MEDIUM,
        assigned_agent_id: int | None = None,
    ) -> Escalation:
        return self.add(
            Escalation(
                interaction_id=interaction_id,
                reason=reason,
                priority=priority,
                assigned_agent_id=assigned_agent_id,
                status=EscalationStatus.ASSIGNED
                if assigned_agent_id is not None
                else EscalationStatus.OPEN,
            )
        )

    def update_status(
        self, escalation_id: int, status: EscalationStatus
    ) -> Escalation | None:
        escalation = self.get(escalation_id)
        if escalation is None:
            return None
        escalation.status = status
        if status == EscalationStatus.RESOLVED:
            escalation.resolved_at = datetime.now(timezone.utc)
        self.session.flush()
        return escalation

    def list_open(self, priority: EscalationPriority | None = None) -> list[Escalation]:
        stmt = select(Escalation).where(Escalation.status != EscalationStatus.RESOLVED)
        if priority is not None:
            stmt = stmt.where(Escalation.priority == priority)
        stmt = stmt.order_by(Escalation.created_at.asc())
        return list(self.session.execute(stmt).scalars().all())
