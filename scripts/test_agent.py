"""Manually run the agent graph against a real ticket, using your real
OpenAI/Pinecone/LangSmith credentials from .env. Use this to sanity-check
the whole pipeline before wiring up the API endpoint.

Usage:
    uv run python scripts/test_agent.py
    uv run python scripts/test_agent.py --customer-id 3 --text "Does my policy cover a cracked windscreen?"

Prerequisites (in order):
    1. Postgres running: docker compose up -d postgres
    2. Tables created:    uv run python scripts/init_db.py
    3. Seed data loaded:  uv run python scripts/seed_customers.py
    4. Policies ingested: uv run python scripts/ingest.py
    5. .env filled in with real OPENAI_API_KEY / OPENAI_BASE_URL,
       PINECONE_API_KEY, and (optionally) LANGSMITH_API_KEY + LANGSMITH_TRACING=true
"""

import argparse

from customer_support_agent.core import get_logger, get_session, settings
from customer_support_agent.graph import build_graph
from customer_support_agent.models import MessageSender, TicketCategory
from customer_support_agent.repositories import (
    CustomerRepository,
    TicketMessageRepository,
    TicketRepository,
)

logger = get_logger(__name__)

DEFAULT_TICKET_TEXT = (
    "My car was hit while parked in a mall parking lot yesterday. "
    "Is this covered under my policy, and what's my excess?"
)


def _format_customer_context(context: dict | None) -> str:
    """Render customer_context as a readable block instead of raw JSON --
    a wall of JSON is hard to scan when you're eyeballing agent output."""
    if not context:
        return "  (none)"

    lines = [
        f"  Active policy:      {'Yes' if context.get('has_active_policy') else 'No'}",
        f"  Policy number:      {context.get('policy_number') or '-'}",
        f"  Policy status:      {context.get('policy_status') or '-'}",
        f"  Policy expiry:      {context.get('policy_expiry_date') or '-'}",
        f"  Tickets (30 days):  {context.get('tickets_last_30_days', 0)}",
    ]

    interactions = context.get("recent_interactions") or []
    if interactions:
        lines.append(f"  Recent interactions ({len(interactions)}):")
        header = f"    {'Ticket':<8} {'Date':<12} {'Escalated':<10} Summary"
        lines.append(header)
        lines.append(f"    {'-' * 8} {'-' * 12} {'-' * 10} {'-' * 45}")
        for item in interactions:
            ticket_id = str(item.get("ticket_id", "-"))
            created_at = (item.get("created_at") or "")[:10]  # date part only
            escalated = "Yes" if item.get("escalated") else "No"
            summary = item.get("summary") or ""
            if len(summary) > 55:
                summary = summary[:52] + "..."
            lines.append(f"    {ticket_id:<8} {created_at:<12} {escalated:<10} {summary}")
    else:
        lines.append("  Recent interactions: (none)")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--customer-id", type=int, default=1,
        help="Existing customer id to open the ticket under (default: 1, the first seeded customer)",
    )
    parser.add_argument("--text", type=str, default=DEFAULT_TICKET_TEXT, help="Ticket text")
    args = parser.parse_args()

    print(f"Chat model:       {settings.chat_model}")
    print(f"Embedding model:  {settings.embedding_model}")
    print(f"LangSmith:        tracing={settings.langsmith_tracing} project={settings.langsmith_project}\n")

    with get_session() as session:
        customers = CustomerRepository(session)
        customer = customers.get(args.customer_id)
        if customer is None:
            print(f"No customer with id={args.customer_id}. Run scripts/seed_customers.py first.")
            return

        tickets = TicketRepository(session)
        ticket = tickets.create(customer_id=customer.id, category=TicketCategory.OTHER)
        session.flush()

        # Store the text as a message too, matching what POST /tickets does --
        # keeps CLI-created and API-created tickets consistent, so any
        # ticket this script creates could also be re-processed later via
        # POST /tickets/{id}/process without a "no message text" error.
        messages = TicketMessageRepository(session)
        messages.add_message(ticket.id, MessageSender.CUSTOMER, args.text)
        session.commit()
        print(f"Created ticket id={ticket.id} for customer '{customer.name}' (id={customer.id})")
        print(f"Ticket text: {args.text}\n")

        graph = build_graph(session)
        try:
            result = graph.invoke(
                {
                    "ticket_id": ticket.id,
                    "customer_id": customer.id,
                    "ticket_text": args.text,
                },
                config={
                    # Names the LangSmith trace so it's findable in the UI,
                    # instead of showing up as an unlabeled run.
                    "run_name": f"ticket-{ticket.id}",
                    "tags": ["manual-test"],
                    "metadata": {"ticket_id": ticket.id, "customer_id": customer.id},
                },
            )
        except Exception as exc:
            print("\n" + "=" * 72)
            print(f"Agent run failed: {exc}")
            message = str(exc).lower()
            if "model" in message and ("not exist" in message or "not found" in message):
                print(
                    "\nHint: this usually means CHAT_MODEL (or EMBEDDING_MODEL) in your "
                    ".env doesn't exactly match a model name your OpenAI-compatible "
                    "endpoint supports. Model names are case-sensitive (e.g. "
                    "'gpt-4o-mini', not 'GPT-4o-mini') -- double-check the exact "
                    "spelling/casing against your provider's supported model list."
                )
            print("=" * 72)
            return
        session.commit()

    print("=" * 72)
    print(f"Category:          {result.get('category')}")
    print(f"Tool calls made:   {result.get('tool_calls_made')}")
    print(f"Retrieved clauses: {len(result.get('retrieved_clauses', []))}")
    print("Customer context:")
    print(_format_customer_context(result.get("customer_context")))
    print(f"Faithfulness pass: {result.get('faithfulness_pass')}  ({result.get('faithfulness_reason')})")
    print(f"Escalated:         {result.get('escalated')}  ({result.get('escalation_reason')})")
    print("-" * 72)
    print(f"Summary:\n{result.get('summary')}\n")
    print(f"Draft response:\n{result.get('draft_response')}")
    print("=" * 72)

    if settings.langsmith_tracing and settings.langsmith_api_key:
        print(f"\nView the trace at https://smith.langchain.com/ -> project '{settings.langsmith_project}'")
    else:
        print("\n(LangSmith tracing is off -- set LANGSMITH_TRACING=true in .env to see traces.)")


if __name__ == "__main__":
    main()
