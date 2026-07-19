"""Tools for the customer-facing chat assistant (graph/chat_agent.py).

Kept separate from graph/tools.py (the ticket-processing agent's tools):
this is a genuinely different assistant with a different job -- answering
account/administrative questions and helping create tickets, not
processing a ticket through the full classify/faithfulness/escalation
pipeline. It deliberately does NOT give live coverage/claim determinations
itself (see the system prompt in chat_agent.py) -- those still go through
a real ticket + human review, preserving the human-in-the-loop design.

Same security principle as the ticket-processing tools: customer_id is
bound via closure, never an argument the LLM controls -- the model must
never be able to guess or choose whose data it reads.
"""

from __future__ import annotations

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from customer_support_agent.core import get_logger, log_transaction
from customer_support_agent.graph.build_graph import build_graph
from customer_support_agent.graph.tools import make_policy_lookup_tool
from customer_support_agent.models import MessageSender, TicketCategory
from customer_support_agent.repositories import (
    CustomerPolicyRepository,
    TicketMessageRepository,
    TicketRepository,
)

logger = get_logger(__name__)


def make_chat_tools(session: Session, customer_id: int, embed_fn=None, index=None) -> list:
    @tool
    def get_my_policy_info() -> dict:
        """Look up the customer's own most recent policy: number, status,
        expiry date, and vehicle registration number."""
        policies = CustomerPolicyRepository(session).list_for_customer(customer_id)
        if not policies:
            return {"has_policy": False}
        latest = policies[0]  # list_for_customer already orders most-recent-first
        return {
            "has_policy": True,
            "policy_number": latest.policy_number,
            "status": latest.status.value,
            "expiry_date": latest.expiry_date.isoformat(),
            "vehicle_reg_no": latest.vehicle_reg_no,
        }

    @tool
    def list_my_tickets() -> list[dict]:
        """List the customer's own support tickets with their status."""
        tickets = TicketRepository(session).list_for_customer(customer_id)
        return [
            {
                "ticket_id": t.id,
                "category": t.category.value,
                "status": t.status.value,
                "resolution": t.resolution.value if t.resolution else None,
                "opened_at": t.opened_at.isoformat(),
            }
            for t in tickets
        ]

    @tool
    def create_support_ticket(issue_description: str) -> dict:
        """Create a new support ticket describing the customer's issue.
        Only call this after the customer has explicitly asked you to
        open/file/create a ticket, or has confirmed "yes" after you offered
        to open one -- never on your own judgment alone, even if their
        situation clearly needs a real coverage/claim determination or
        human review. If it seems like they need one, tell them so and ask
        first; only call this tool once they've actually agreed."""
        ticket_repo = TicketRepository(session)
        message_repo = TicketMessageRepository(session)

        ticket = ticket_repo.create(customer_id=customer_id, category=TicketCategory.OTHER)
        session.flush()
        message_repo.add_message(ticket.id, MessageSender.CUSTOMER, issue_description)
        session.commit()

        try:
            graph = build_graph(session)
            with log_transaction("process_ticket", ticket_id=ticket.id, customer_id=customer_id):
                result = graph.invoke(
                    {
                        "ticket_id": ticket.id,
                        "customer_id": customer_id,
                        "ticket_text": issue_description,
                    },
                    config={
                        "run_name": f"ticket-{ticket.id}",
                        "tags": ["chat"],
                        "metadata": {"ticket_id": ticket.id, "customer_id": customer_id},
                    },
                )

            # Mirror api/tickets.py's process_ticket() post-processing: the
            # graph's own nodes persist the Interaction row, but the draft
            # and classified category are only ever saved here -- this was
            # the bug (see initial_analysis.txt), these two steps must run
            # on every code path that invokes the graph, not just the
            # /tickets/{id}/process endpoint.
            draft_response = result.get("draft_response")
            if draft_response:
                message_repo.add_message(ticket.id, MessageSender.AI_DRAFT, draft_response)

            classified_category = result.get("category")
            if classified_category:
                try:
                    ticket.category = TicketCategory(classified_category)
                except ValueError:
                    logger.warning(
                        "chat create_support_ticket: model returned unrecognized "
                        "category %r for ticket_id=%s, leaving existing category unchanged",
                        classified_category,
                        ticket.id,
                    )

            session.commit()
        except Exception as exc:  # noqa: BLE001 -- ticket creation must still succeed
            session.rollback()
            logger.warning(
                "chat create_support_ticket: auto-processing failed for ticket_id=%s: %r",
                ticket.id,
                exc,
            )

        return {"ticket_id": ticket.id, "status": "open"}

    policy_lookup = make_policy_lookup_tool(embed_fn=embed_fn, index=index)

    return [get_my_policy_info, list_my_tickets, create_support_ticket, policy_lookup]
