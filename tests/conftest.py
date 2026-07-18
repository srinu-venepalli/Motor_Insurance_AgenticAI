"""Shared pytest fixtures for the model test suite.

Defaults to an in-memory SQLite DB so tests run with zero setup. To run the
exact same suite against your real Postgres (docker-compose) instance:

    TEST_DATABASE_URL="postgresql+psycopg2://agent:localdev@localhost:5432/insurance_support" \
        uv run pytest tests/ -v

Each test gets a fresh schema (create_all before, drop_all after) so tests
stay isolated regardless of which backend is in play.
"""

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from customer_support_agent.models import Base

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture()
def engine():
    # FastAPI dispatches sync (`def`, not `async def`) route handlers to a
    # thread pool -- SQLite ties a connection to the thread that created it,
    # so without these options a TestClient request from a different thread
    # than the fixture setup would fail with "SQLite objects created in a
    # thread can only be used in that same thread". Postgres (what
    # production actually uses) has no such restriction, so this only
    # matters for the sqlite test default.
    kwargs = {}
    if TEST_DATABASE_URL.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs["poolclass"] = StaticPool

    eng = create_engine(TEST_DATABASE_URL, **kwargs)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def session(engine):
    with Session(engine) as session:
        yield session
