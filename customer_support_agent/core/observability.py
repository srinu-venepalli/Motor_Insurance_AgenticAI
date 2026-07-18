"""Observability helpers: PII-safe transaction logging.

This is where the "must not store personal data in logs" safety requirement
is actually enforced. Real PII (customer name, phone, email) lives in
Postgres (see models/customer.py) -- fine, that's the system of record.
What must NOT happen is that PII flowing into a log line or a LangSmith
trace. redact_for_trace() is the single choke point for that rule; every
logger call and every future LangSmith client hook should route payloads
through it before they're written anywhere.
"""

import time
from contextlib import contextmanager
from typing import Any, Iterator

from customer_support_agent.core.logging_config import get_logger

logger = get_logger("transactions")

# Field names that must never appear in a log line, wherever they show up
# in a context/payload dict passed to log_transaction().
PII_FIELDS = {"name", "contact_no", "contact_email", "phone", "email", "vehicle_reg_no"}

REDACTED = "[REDACTED]"


def redact_for_trace(payload: Any) -> Any:
    """Recursively redact PII_FIELDS keys from dicts/lists. Non-dict/list
    values are returned unchanged."""
    if isinstance(payload, dict):
        return {
            k: (REDACTED if k in PII_FIELDS else redact_for_trace(v)) for k, v in payload.items()
        }
    if isinstance(payload, (list, tuple)):
        return [redact_for_trace(item) for item in payload]
    return payload


@contextmanager
def log_transaction(action: str, **context: Any) -> Iterator[None]:
    """Wrap any unit of work (a ticket being processed, a script seeding
    data, an API request) so it produces exactly one START line and one
    SUCCESS/FAILURE line in the app log, with PII scrubbed from context.

    Usage:
        with log_transaction("process_ticket", ticket_id=42, customer_id=7):
            ...do the work...
    """
    safe_context = redact_for_trace(context)
    start = time.perf_counter()
    logger.info("START action=%s context=%s", action, safe_context)
    try:
        yield
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            "FAILURE action=%s duration_ms=%s context=%s error=%s",
            action,
            duration_ms,
            safe_context,
            repr(exc),
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "SUCCESS action=%s duration_ms=%s context=%s", action, duration_ms, safe_context
        )
