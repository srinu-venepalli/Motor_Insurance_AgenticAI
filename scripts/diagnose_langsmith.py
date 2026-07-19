"""Diagnose whether traces are actually reaching LangSmith.

Run this when a ticket/chat action completes cleanly in your own app logs
but never shows up in the LangSmith UI. Your main app can't tell you why --
LangSmith's tracing client submits traces via a background thread,
fire-and-forget by design, specifically so a tracing failure never blocks
or breaks your actual application. That means app.log looks identical
whether traces are landing perfectly or failing 100% of the time.

This script uses the LangSmith SDK directly and prints exactly what
happens at each step, instead of letting failures disappear silently.

Usage:
    uv run python scripts/diagnose_langsmith.py
"""

import time
import uuid

from langsmith import Client, traceable

from customer_support_agent.core import settings

RUN_NAME = f"diagnose-langsmith-{uuid.uuid4().hex[:8]}"


@traceable(name=RUN_NAME)
def _tiny_traced_call() -> str:
    return "ok"


def main() -> None:
    print("=" * 72)
    print("Current LangSmith configuration")
    print("=" * 72)
    print(f"LANGSMITH_PROJECT: {settings.langsmith_project!r}")
    print(f"LANGSMITH_TRACING: {settings.langsmith_tracing!r}")
    masked_key = f"{settings.langsmith_api_key[:8]}..." if settings.langsmith_api_key else "NOT SET"
    print(f"LANGSMITH_API_KEY: {masked_key}")

    if not settings.langsmith_api_key:
        print("\nNo API key configured -- set LANGSMITH_API_KEY in .env first.")
        return
    if not settings.langsmith_tracing:
        print("\nLANGSMITH_TRACING is False -- set it to true in .env first.")
        return

    client = Client(api_key=settings.langsmith_api_key)

    print("\n" + "=" * 72)
    print("TEST 1: Can we reach LangSmith and read this project at all?")
    print("=" * 72)
    try:
        recent = list(client.list_runs(project_name=settings.langsmith_project, limit=1))
        print(f"OK -- API key is valid, project {settings.langsmith_project!r} is reachable.")
        print(f"({len(recent)} run(s) fetched in this check.)")
    except Exception as exc:
        print(f"FAILED: {exc!r}")
        print(
            "\nThis usually means an invalid/expired API key, or the project name doesn't\n"
            "exist under the account this key belongs to. Check your .env's\n"
            "LANGSMITH_API_KEY and LANGSMITH_PROJECT against your actual LangSmith account."
        )
        return

    print("\n" + "=" * 72)
    print(f"TEST 2: Submit one tiny traced call (run_name={RUN_NAME!r})")
    print("=" * 72)
    _tiny_traced_call()
    print("Called. Waiting 8s for LangSmith to ingest it (usually near-instant)...")
    time.sleep(8)

    print("\nQuerying LangSmith for that specific run...")
    found = list(
        client.list_runs(
            project_name=settings.langsmith_project,
            filter=f'eq(name, "{RUN_NAME}")',
        )
    )
    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    if found:
        print(f"FOUND -- trace {found[0].id} landed successfully.")
        print(
            "Your tracing pipeline is healthy end-to-end. If your earlier real ticket/chat\n"
            "traces still aren't showing up, double check you're looking at the same\n"
            "project this script just confirmed, and that LANGSMITH_TRACING was true at\n"
            "the moment those specific calls ran (not just now)."
        )
    else:
        print(
            "NOT FOUND after 8s. The API key and project are reachable (Test 1 passed),\n"
            "but this specific trace was silently rejected on ingestion. The most likely\n"
            "cause is your LangSmith organization's usage/trace quota -- check the\n"
            "Usage & Billing page in your LangSmith account (not the Tracing project\n"
            "list, which won't show a quota warning) for a limit-reached notice."
        )


if __name__ == "__main__":
    main()
