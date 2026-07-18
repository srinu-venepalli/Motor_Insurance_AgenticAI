"""Thin HTTP client wrapping the FastAPI backend.

Deliberately calls the API over HTTP rather than importing
customer_support_agent directly -- this keeps the UI and the API properly
decoupled, the same way a real deployment would have them as separate
services.

Every call is logged (see core/logging_config.py) -- this is what makes the
REST calls observable at all from the Streamlit side: they happen entirely
server-side (in this Python process), so the browser's Network tab never
sees them. Check logs/app.log (or the terminal running `streamlit run`) to
see them, not browser dev tools.
"""

from __future__ import annotations

import time

import requests

from customer_support_agent.core import get_logger, settings

logger = get_logger(__name__)

BASE_URL = settings.api_base_url
_TIMEOUT_SHORT = 15
_TIMEOUT_PROCESS = 60  # LLM calls in /process can take a while


class ApiError(RuntimeError):
    """Raised for any non-2xx response, with the server's detail message."""


def _handle(method: str, url: str, response: requests.Response, started_at: float) -> dict:
    duration_ms = round((time.perf_counter() - started_at) * 1000, 1)
    logger.info("UI -> API %s %s -> %s (%sms)", method, url, response.status_code, duration_ms)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"{response.status_code}: {detail}")
    return response.json()


def list_customers() -> list[dict]:
    url = f"{BASE_URL}/customers"
    started_at = time.perf_counter()
    resp = requests.get(url, timeout=_TIMEOUT_SHORT)
    return _handle("GET", url, resp, started_at)


def get_customer(customer_id: int) -> dict:
    url = f"{BASE_URL}/customers/{customer_id}"
    started_at = time.perf_counter()
    resp = requests.get(url, timeout=_TIMEOUT_SHORT)
    return _handle("GET", url, resp, started_at)


def create_ticket(customer_id: int, ticket_text: str) -> dict:
    url = f"{BASE_URL}/tickets"
    started_at = time.perf_counter()
    resp = requests.post(
        url, json={"customer_id": customer_id, "ticket_text": ticket_text}, timeout=_TIMEOUT_SHORT
    )
    return _handle("POST", url, resp, started_at)


def process_ticket(ticket_id: int) -> dict:
    url = f"{BASE_URL}/tickets/{ticket_id}/process"
    started_at = time.perf_counter()
    resp = requests.post(url, timeout=_TIMEOUT_PROCESS)
    return _handle("POST", url, resp, started_at)


def get_ticket(ticket_id: int) -> dict:
    url = f"{BASE_URL}/tickets/{ticket_id}"
    started_at = time.perf_counter()
    resp = requests.get(url, timeout=_TIMEOUT_SHORT)
    return _handle("GET", url, resp, started_at)


def list_tickets(
    customer_id: int | None = None, status: str | None = None, resolution: str | None = None
) -> list[dict]:
    url = f"{BASE_URL}/tickets"
    params = {}
    if customer_id is not None:
        params["customer_id"] = customer_id
    if status is not None:
        params["status"] = status
    if resolution is not None:
        params["resolution"] = resolution
    started_at = time.perf_counter()
    resp = requests.get(url, params=params, timeout=_TIMEOUT_SHORT)
    return _handle("GET", url, resp, started_at)


def approve_ticket(
    ticket_id: int, final_response: str | None = None, resolution: str = "approved"
) -> dict:
    """resolution: 'approved' or 'rejected'. Same endpoint handles both --
    the distinction is just which button the agent clicked in the console."""
    url = f"{BASE_URL}/tickets/{ticket_id}/approve"
    started_at = time.perf_counter()
    resp = requests.post(
        url,
        json={"final_response": final_response, "resolution": resolution},
        timeout=_TIMEOUT_SHORT,
    )
    return _handle("POST", url, resp, started_at)
