"""Application logging setup.

Call setup_logging() once at process start (main.py, app.py, scripts, and
the pytest fixture in tests/conftest.py all do this). Everything after that
should use get_logger(__name__) rather than the print/logging module
directly, so all output lands in the same rotating file.

This is deliberately separate from LangSmith tracing (core/observability.py
handles that): this module is plain Python logging for ops-level visibility
(errors, latency, request lifecycle), LangSmith is for LLM-call-level
tracing. Both go through the same PII redaction rule, but they are
different systems for different audiences.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from customer_support_agent.core.settings import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# core/observability.py's log_transaction() writes exclusively to a logger
# named this -- kept as its own file (see setup_logging below) so business
# events (ticket processed, escalated, etc.) are a readable audit trail
# instead of being interleaved with per-request httpx/ui.api_client noise.
TRANSACTIONS_LOGGER_NAME = "transactions"

# httpx logs a line for every single outbound HTTP call (every LLM call is
# one) at INFO -- that's the dominant source of noise in the main app log.
# WARNING still surfaces anything actually going wrong (4xx/5xx, retries).
_NOISY_LOGGERS_MAX_LEVEL = {"httpx": logging.WARNING}

_configured = False


def setup_logging() -> None:
    """Idempotent -- safe to call multiple times (e.g. once in main.py, once
    again in a script that imports main.py's modules)."""
    global _configured
    if _configured:
        return

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / settings.log_file

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    for logger_name, max_level in _NOISY_LOGGERS_MAX_LEVEL.items():
        logging.getLogger(logger_name).setLevel(max_level)

    # Separate audit stream for log_transaction() business events, in its
    # own rotating file rather than mixed into app.log. propagate=False so
    # these lines aren't *also* duplicated into the root handlers above.
    transactions_path = log_dir / "transactions.log"
    transactions_handler = RotatingFileHandler(
        transactions_path,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    transactions_handler.setFormatter(formatter)
    transactions_logger = logging.getLogger(TRANSACTIONS_LOGGER_NAME)
    transactions_logger.setLevel(settings.log_level.upper())
    transactions_logger.addHandler(transactions_handler)
    transactions_logger.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
