"""LangSmith tracing setup.

LangChain/LangGraph auto-trace to LangSmith when certain environment
variables are set -- there's no client object to construct, just env vars
read at call time. This module is the single place that translates our own
typed settings (settings.langsmith_*) into those env vars, called once at
process start (graph construction, main.py).
"""

import os

from customer_support_agent.core.logging_config import get_logger
from customer_support_agent.core.settings import settings

logger = get_logger(__name__)

_configured = False


def configure_langsmith() -> None:
    global _configured
    if _configured:
        return

    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.info(
            "LangSmith tracing disabled (set LANGSMITH_TRACING=true and "
            "LANGSMITH_API_KEY in .env to enable)"
        )

    _configured = True
