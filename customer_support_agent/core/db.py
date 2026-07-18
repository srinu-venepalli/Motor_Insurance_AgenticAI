"""Shared SQLAlchemy engine/session factory.

Import get_session() as a context manager anywhere a script or service
needs a real DB transaction (commits on success, rolls back on exception).
The API layer will wrap this in a FastAPI dependency later.
"""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from customer_support_agent.core.settings import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
