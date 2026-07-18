"""Create all tables against the configured DATABASE_URL.

Usage:
    uv run python scripts/init_db.py

Reads DATABASE_URL from .env via core.settings, so point it at the
docker-compose Postgres instance (or sqlite for a quick local check) before
running this.
"""

from sqlalchemy import create_engine

from customer_support_agent.core import settings
from customer_support_agent.models import Base


def main() -> None:
    print(f"Connecting to: {settings.database_url}")
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        print("Connection OK.")

    Base.metadata.create_all(engine)
    table_names = sorted(Base.metadata.tables.keys())
    print(f"Created {len(table_names)} tables:")
    for name in table_names:
        print(f"  - {name}")

    engine.dispose()


if __name__ == "__main__":
    main()
