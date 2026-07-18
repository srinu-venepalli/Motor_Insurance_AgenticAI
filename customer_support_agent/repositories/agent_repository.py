"""Agent repository -- human support agents."""

from sqlalchemy import select

from customer_support_agent.models import Agent
from customer_support_agent.repositories.base import BaseRepository


class AgentRepository(BaseRepository[Agent]):
    model = Agent

    def create(self, name: str, role: str = "support_agent") -> Agent:
        return self.add(Agent(name=name, role=role))

    def get_by_name(self, name: str) -> Agent | None:
        stmt = select(Agent).where(Agent.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_or_create_by_name(self, name: str, role: str = "support_agent") -> Agent:
        existing = self.get_by_name(name)
        if existing is not None:
            return existing
        return self.create(name=name, role=role)
