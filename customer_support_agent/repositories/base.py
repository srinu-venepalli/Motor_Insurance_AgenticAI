"""Generic base repository.

Each table-specific repository subclasses this and sets `model`. Repos take
a SQLAlchemy Session in their constructor rather than opening their own --
in FastAPI this session will come from a per-request dependency, and in
tests it comes from the conftest.py fixture (sqlite or real Postgres).
"""

from typing import Generic, Sequence, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from customer_support_agent.models.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    model: Type[ModelType]

    def __init__(self, session: Session):
        self.session = session

    def get(self, id: int) -> ModelType | None:
        return self.session.get(self.model, id)

    def list(self, limit: int = 100, offset: int = 0) -> Sequence[ModelType]:
        stmt = select(self.model).limit(limit).offset(offset)
        return self.session.execute(stmt).scalars().all()

    def add(self, obj: ModelType) -> ModelType:
        """Add and flush (assigns the PK) without committing -- caller controls
        the transaction boundary (commit/rollback), which matters once this
        is used inside a LangGraph node that might fail mid-way."""
        self.session.add(obj)
        self.session.flush()
        return obj

    def delete(self, obj: ModelType) -> None:
        self.session.delete(obj)
        self.session.flush()
