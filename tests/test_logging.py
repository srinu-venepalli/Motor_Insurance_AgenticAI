"""Logging/observability tests.

Verifies log_transaction() actually writes to the configured log file, and
that PII fields are redacted while non-PII context (ticket_id, action) is
preserved -- this is the concrete evidence for the "must not store personal
data in logs" safety requirement.
"""

import logging

import pytest

from customer_support_agent.core.observability import log_transaction, redact_for_trace


def test_redact_for_trace_removes_pii_fields_only():
    payload = {
        "customer_id": 7,
        "name": "Rohan Sharma",
        "contact_no": "+91-9876543210",
        "ticket_id": 42,
    }
    redacted = redact_for_trace(payload)

    assert redacted["customer_id"] == 7
    assert redacted["ticket_id"] == 42
    assert redacted["name"] == "[REDACTED]"
    assert redacted["contact_no"] == "[REDACTED]"


def test_redact_for_trace_handles_nested_structures():
    payload = {
        "ticket_id": 1,
        "customer": {"name": "Anita Rao", "contact_email": "anita@example.com"},
        "items": [{"name": "should be redacted"}, {"clause_id": "OD-4.2"}],
    }
    redacted = redact_for_trace(payload)

    assert redacted["customer"]["name"] == "[REDACTED]"
    assert redacted["customer"]["contact_email"] == "[REDACTED]"
    assert redacted["items"][0]["name"] == "[REDACTED]"
    assert redacted["items"][1]["clause_id"] == "OD-4.2"


def test_log_transaction_writes_success_line_without_pii(caplog):
    with caplog.at_level(logging.INFO, logger="transactions"):
        with log_transaction(
            "process_ticket", ticket_id=99, customer_id=3, name="Rohan Sharma"
        ):
            pass

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "SUCCESS" in log_text
    assert "action=process_ticket" in log_text
    assert "ticket_id" in log_text or "99" in log_text
    assert "Rohan Sharma" not in log_text
    assert "REDACTED" in log_text


def test_log_transaction_writes_failure_line_and_reraises(caplog):
    with caplog.at_level(logging.INFO, logger="transactions"):
        with pytest.raises(ValueError):
            with log_transaction("process_ticket", ticket_id=100, name="Anita Rao"):
                raise ValueError("faithfulness check failed")

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "FAILURE" in log_text
    assert "Anita Rao" not in log_text


def test_log_file_is_actually_created(tmp_path, monkeypatch):
    """End-to-end: point log_dir at a temp folder and confirm a real file
    gets written to, not just captured by pytest's log handler."""
    from customer_support_agent.core import logging_config

    monkeypatch.setattr(logging_config.settings, "log_dir", str(tmp_path))
    monkeypatch.setattr(logging_config, "_configured", False)
    logging_config.setup_logging()

    logger = logging_config.get_logger("test.file_check")
    logger.info("hello from test_log_file_is_actually_created")

    for handler in logging.getLogger().handlers:
        handler.flush()

    log_file = tmp_path / logging_config.settings.log_file
    assert log_file.exists()
    assert "hello from test_log_file_is_actually_created" in log_file.read_text()
