"""Chat endpoint -- the customer-facing conversational assistant.

Stateless on the server side: the caller (the Streamlit UI, holding
short-term memory in session state) sends the full conversation history
with each request, and gets back the updated history plus the new reply.
See graph/chat_agent.py for the actual reasoning/memory design notes.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from customer_support_agent.api.deps import get_db
from customer_support_agent.core import get_logger
from customer_support_agent.core.retry import llm_call_retry
from customer_support_agent.graph.chat_agent import run_chat_turn
from customer_support_agent.repositories import CustomerRepository
from customer_support_agent.schemas.api import ChatMessage, ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


@llm_call_retry
def _run_chat_turn_with_retry(db, customer_id, customer_name, history, message):
    return run_chat_turn(db, customer_id, customer_name, history, message)


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    customer = CustomerRepository(db).get(payload.customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer {payload.customer_id} not found")

    history_dicts = [m.model_dump() for m in payload.history]
    try:
        reply, updated_history = _run_chat_turn_with_retry(
            db, customer.id, customer.name, history_dicts, payload.message
        )
    except Exception as exc:
        logger.error(
            "chat: run_chat_turn failed after retries for customer_id=%s: %r",
            customer.id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "The assistant returned an invalid or incomplete response after several "
                "attempts. This is usually a transient issue -- please try again in a moment."
            ),
        ) from exc

    return ChatResponse(reply=reply, history=[ChatMessage(**h) for h in updated_history])
