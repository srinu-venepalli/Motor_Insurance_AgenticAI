"""Shared state schema for the customer support agent's LangGraph graph.

A TypedDict, not a Pydantic model -- LangGraph nodes return partial dict
updates that get merged into this state, which is the standard pattern for
StateGraph. total=False since most fields are populated progressively as
the ticket moves through the graph, not all at once.
"""

from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- Input ---
    ticket_id: int
    customer_id: int
    ticket_text: str

    # --- classify_ticket ---
    category: Optional[str]

    # --- fetch_customer_context (deterministic, always runs) ---
    customer_name: Optional[str]
    tool_calls_made: list[str]
    retrieved_clauses: list[dict]
    customer_context: Optional[dict]
    # Phase 7 adaptive behaviour signal: average historical edit_distance
    # for this ticket's category (from Feedback), or None if there isn't
    # enough data yet -- see _evaluate_escalation() in graph/nodes.py.
    category_avg_edit_distance: Optional[float]

    # --- summarize_and_draft ---
    summary: Optional[str]
    draft_response: Optional[str]
    cited_clause_ids: list[str]
    claimed_amount: Optional[float]
    needs_escalation_soft_signal: bool
    escalation_soft_reason: Optional[str]

    # --- faithfulness_check (deterministic) ---
    faithfulness_pass: bool
    faithfulness_reason: Optional[str]

    # --- escalation_gate / escalate_to_agent / present_to_human (deterministic) ---
    escalated: bool
    escalation_reason: Optional[str]
    escalation_id: Optional[int]
