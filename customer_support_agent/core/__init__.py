"""Core package -- re-exports settings, logging, DB session, tracing, and
the transaction/PII-redaction helper so callers can do e.g.:

    from customer_support_agent.core import settings, get_logger, log_transaction
"""

from customer_support_agent.core.db import get_session
from customer_support_agent.core.logging_config import get_logger, setup_logging
from customer_support_agent.core.observability import log_transaction, redact_for_trace
from customer_support_agent.core.settings import Settings, get_settings, settings
from customer_support_agent.core.tracing import configure_langsmith

__all__ = [
    "Settings",
    "get_settings",
    "settings",
    "get_logger",
    "setup_logging",
    "log_transaction",
    "redact_for_trace",
    "get_session",
    "configure_langsmith",
]
