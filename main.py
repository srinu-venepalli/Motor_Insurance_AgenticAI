"""FastAPI entrypoint. Run with:

    uv run uvicorn main:app --reload

or:

    uv run python main.py
"""

from fastapi import FastAPI

from customer_support_agent.api import chat, customers, health, ingestion, tickets
from customer_support_agent.core import get_logger, setup_logging, settings

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title=settings.app_name)
app.include_router(health.router)
app.include_router(ingestion.router)
app.include_router(tickets.router)
app.include_router(customers.router)
app.include_router(chat.router)


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
