"""Ticket endpoints.

POST /tickets              -- create a new ticket (customer's raw text)
GET  /tickets              -- list tickets (agent console queue)
POST /tickets/{id}/process -- run the agent graph on it, persisting the
                              resulting Interaction/Escalation via the
                              graph's own terminal nodes, and the draft
                              itself as an AI_DRAFT ticket_messages row
GET  /tickets/{id}         -- fetch the ticket + its interaction/message history
POST /tickets/{id}/approve -- human agent sends the (possibly edited) draft,
                              closing the ticket

Deliberately separate create/process/approve steps rather than one combined
endpoint: ticket intake, the AI's processing of it, and a human agent's
final sign-off are three different events in time, and keeping them
separate means a ticket can exist and be inspected at any stage.
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from customer_support_agent.api.deps import get_db
from customer_support_agent.core import get_logger
from customer_support_agent.core.retry import llm_call_retry
from customer_support_agent.graph import build_graph
from customer_support_agent.models import (
    MessageSender,
    TicketCategory,
    TicketResolution,
    TicketStatus,
)
from customer_support_agent.repositories import (
    CustomerRepository,
    FeedbackRepository,
    InteractionRepository,
    TicketMessageRepository,
    TicketRepository,
)
from customer_support_agent.schemas.api import (
    InteractionSummary,
    MessageOut,
    TicketApproveRequest,
    TicketApproveResponse,
    TicketCreateRequest,
    TicketCreateResponse,
    TicketDetailResponse,
    TicketProcessResponse,
    TicketSummary,
)

router = APIRouter(prefix="/tickets", tags=["tickets"])
logger = get_logger(__name__)


@llm_call_retry
def _invoke_graph_with_retry(graph, initial_state: dict, config: dict) -> dict:
    return graph.invoke(initial_state, config=config)


def _levenshtein(a: str, b: str) -> int:
    """Plain DP edit distance -- no extra dependency needed for strings
    this short. Backs Feedback.edit_distance, the Phase 7 'how much did the
    agent have to change the draft' evidence, which was sitting unused
    until now."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


@router.post("", response_model=TicketCreateResponse, status_code=201)
def create_ticket(payload: TicketCreateRequest, db: Session = Depends(get_db)) -> TicketCreateResponse:
    customers = CustomerRepository(db)
    customer = customers.get(payload.customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer {payload.customer_id} not found")

    category = TicketCategory(payload.category) if payload.category else TicketCategory.OTHER

    tickets = TicketRepository(db)
    ticket = tickets.create(customer_id=customer.id, category=category)
    db.flush()

    messages = TicketMessageRepository(db)
    messages.add_message(ticket.id, MessageSender.CUSTOMER, payload.ticket_text)

    return TicketCreateResponse(
        ticket_id=ticket.id,
        customer_id=customer.id,
        status=ticket.status.value,
        opened_at=ticket.opened_at,
    )


@router.get("", response_model=list[TicketSummary])
def list_tickets(
    status: str | None = None,
    resolution: str | None = None,
    customer_id: int | None = None,
    db: Session = Depends(get_db),
) -> list[TicketSummary]:
    """Agent console queue view (all customers) when customer_id is
    omitted; a specific customer's own ticket history when it's provided.
    resolution filters to 'approved' or 'rejected' -- only meaningful
    alongside closed tickets, since open tickets have no resolution yet."""
    tickets_repo = TicketRepository(db)
    customers_repo = CustomerRepository(db)
    interactions_repo = InteractionRepository(db)

    status_enum = TicketStatus(status) if status else None
    try:
        resolution_enum = TicketResolution(resolution) if resolution else None
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"resolution must be 'approved' or 'rejected', got {resolution!r}",
        )

    if customer_id is not None:
        tickets = tickets_repo.list_for_customer(
            customer_id, status=status_enum, resolution=resolution_enum
        )
    else:
        tickets = tickets_repo.list_all(status=status_enum, resolution=resolution_enum)

    summaries = []
    for ticket in tickets:
        customer = customers_repo.get(ticket.customer_id)
        interactions = interactions_repo.get_for_ticket(ticket.id)
        latest_escalated = interactions[-1].escalated if interactions else None
        latest_summary = interactions[-1].summary if interactions else None
        summaries.append(
            TicketSummary(
                ticket_id=ticket.id,
                customer_id=ticket.customer_id,
                customer_name=customer.name if customer else None,
                category=ticket.category.value,
                status=ticket.status.value,
                resolution=ticket.resolution.value if ticket.resolution else None,
                opened_at=ticket.opened_at,
                escalated=latest_escalated,
                latest_summary=latest_summary,
            )
        )
    return summaries


@router.post("/{ticket_id}/process", response_model=TicketProcessResponse)
def process_ticket(ticket_id: int, db: Session = Depends(get_db)) -> TicketProcessResponse:
    tickets = TicketRepository(db)
    ticket = tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    messages = TicketMessageRepository(db)
    thread = messages.get_thread(ticket_id)
    if not thread:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Ticket {ticket_id} has no message text to process. "
                "This usually means it was created before this endpoint existed, or by "
                "a script/CLI path that didn't store a ticket_messages row -- create a "
                "new ticket via POST /tickets (which does store the text) and process that one."
            ),
        )
    ticket_text = thread[0].text  # the original customer submission

    graph = build_graph(db)
    try:
        result = _invoke_graph_with_retry(
            graph,
            {
                "ticket_id": ticket.id,
                "customer_id": ticket.customer_id,
                "ticket_text": ticket_text,
            },
            config={
                "run_name": f"ticket-{ticket.id}",
                "tags": ["api"],
                "metadata": {"ticket_id": ticket.id, "customer_id": ticket.customer_id},
            },
        )
    except Exception as exc:
        logger.error(
            "process_ticket: graph invocation failed after retries for ticket_id=%s: %r",
            ticket_id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "The AI service returned an invalid or incomplete response after several "
                "attempts. This is usually a transient issue with the upstream model "
                "provider -- please try again in a moment."
            ),
        ) from exc

    draft_response = result.get("draft_response")
    if draft_response:
        # Persist the draft so it's retrievable later (GET /tickets/{id} and
        # the agent console) -- graph.invoke()'s return value alone is
        # ephemeral, this is the only place it's actually saved.
        messages.add_message(ticket.id, MessageSender.AI_DRAFT, draft_response)

    classified_category = result.get("category")
    if classified_category:
        # This was the actual bug behind tickets showing 'Other' even when
        # classify_ticket correctly determined the real category: that
        # value was only ever included in this endpoint's response, never
        # written back to the Ticket row itself. Every ticket is created
        # with category=OTHER by default (the customer never specifies one
        # at submission time), so without this, GET /tickets and GET
        # /tickets/{id} -- which read the persisted column, not the
        # ephemeral graph result -- would show OTHER regardless of what
        # the AI actually classified it as.
        try:
            ticket.category = TicketCategory(classified_category)
        except ValueError:
            logger.warning(
                "process_ticket: model returned unrecognized category %r for ticket_id=%s, "
                "leaving existing category unchanged",
                classified_category,
                ticket_id,
            )

    return TicketProcessResponse(
        ticket_id=ticket.id,
        category=result.get("category"),
        tool_calls_made=result.get("tool_calls_made", []),
        retrieved_clauses_count=len(result.get("retrieved_clauses", [])),
        customer_context=result.get("customer_context"),
        claimed_amount=result.get("claimed_amount"),
        faithfulness_pass=result.get("faithfulness_pass", True),
        faithfulness_reason=result.get("faithfulness_reason"),
        escalated=result.get("escalated", False),
        escalation_reason=result.get("escalation_reason"),
        summary=result.get("summary"),
        draft_response=result.get("draft_response"),
    )


@router.get("/{ticket_id}", response_model=TicketDetailResponse)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)) -> TicketDetailResponse:
    tickets = TicketRepository(db)
    ticket = tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    customers = CustomerRepository(db)
    customer = customers.get(ticket.customer_id)

    interactions_repo = InteractionRepository(db)
    interactions = interactions_repo.get_for_ticket(ticket_id)

    message_repo = TicketMessageRepository(db)
    thread = message_repo.get_thread(ticket_id)

    return TicketDetailResponse(
        ticket_id=ticket.id,
        customer_id=ticket.customer_id,
        customer_name=customer.name if customer else None,
        category=ticket.category.value,
        status=ticket.status.value,
        resolution=ticket.resolution.value if ticket.resolution else None,
        opened_at=ticket.opened_at,
        closed_at=ticket.closed_at,
        interactions=[
            InteractionSummary(
                interaction_id=i.id,
                summary=i.summary,
                escalated=i.escalated,
                escalation_reason=i.escalation_reason,
                faithfulness_pass=i.faithfulness_pass,
                created_at=i.created_at,
            )
            for i in interactions
        ],
        messages=[
            MessageOut(sender=m.sender.value, text=m.text, created_at=m.created_at) for m in thread
        ],
    )


@router.post("/{ticket_id}/approve", response_model=TicketApproveResponse)
def approve_ticket(
    ticket_id: int, payload: TicketApproveRequest, db: Session = Depends(get_db)
) -> TicketApproveResponse:
    """Human agent sends the draft (as-is, edited, or a rejection) to the
    customer. Records a HUMAN_AGENT message and closes the ticket -- this
    is the only place a response actually leaves the system, matching the
    safety requirement that the AI never sends anything autonomously.

    Also records Feedback.edit_distance (vs the original AI draft, if one
    existed) -- the Phase 7 'how much did the agent have to change it'
    evidence."""
    tickets = TicketRepository(db)
    ticket = tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    try:
        resolution = TicketResolution(payload.resolution)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"resolution must be 'approved' or 'rejected', got {payload.resolution!r}",
        )

    message_repo = TicketMessageRepository(db)
    thread = message_repo.get_thread(ticket_id)
    drafts = [m for m in thread if m.sender == MessageSender.AI_DRAFT]
    original_draft_text = drafts[-1].text if drafts else None

    final_text = payload.final_response
    if final_text is None:
        if original_draft_text is None:
            raise HTTPException(
                status_code=400,
                detail=f"Ticket {ticket_id} has no AI draft to send -- process it first, "
                "or provide final_response explicitly.",
            )
        final_text = original_draft_text

    message_repo.add_message(ticket_id, MessageSender.HUMAN_AGENT, final_text)

    ticket.resolution = resolution
    updated_ticket = tickets.update_status(ticket_id, TicketStatus.CLOSED)

    edit_distance = None
    if original_draft_text is not None:
        edit_distance = _levenshtein(original_draft_text, final_text)
        interactions = InteractionRepository(db).get_for_ticket(ticket_id)
        if interactions:
            FeedbackRepository(db).create(
                interaction_id=interactions[-1].id,
                edit_distance=edit_distance,
                notes=f"Resolution: {resolution.value}",
            )

    return TicketApproveResponse(
        ticket_id=ticket_id,
        status=updated_ticket.status.value,
        resolution=resolution.value,
        sent_text=final_text,
        edit_distance=edit_distance,
    )
