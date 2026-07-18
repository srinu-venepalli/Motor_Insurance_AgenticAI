"""Shared SQLAlchemy declarative base and common column mixins.

All ORM models in this package inherit from `Base`. `TimestampMixin` and
`IntPKMixin` avoid repeating the same id/created_at boilerplate on every
table.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


class IntPKMixin:
    """Standard auto-incrementing integer primary key."""

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)


class TimestampMixin:
    """created_at column, defaulted server-side-equivalent via Python default."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
