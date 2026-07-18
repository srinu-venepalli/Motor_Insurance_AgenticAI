"""FastAPI dependencies."""

from typing import Iterator

from sqlalchemy.orm import Session

from customer_support_agent.core.db import SessionLocal


def get_db() -> Iterator[Session]:
    """Per-request DB session. Commits on success, rolls back on exception,
    always closes -- mirrors core.db.get_session() but as a FastAPI
    dependency generator rather than a context manager."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
