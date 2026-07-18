"""The two tools bound to the LLM in the agent_reasoning node.

escalate_to_agent is deliberately absent from this module: per the design
decision (see conversation / README), escalation is a rule-based gate the
LLM cannot trigger itself -- it's a plain function call in graph/nodes.py.

Each tool's actual logic is a plain function (policy_lookup_fn,
customer_history_lookup_fn) with injectable dependencies, kept separate
from the @tool-decorated wrapper so the logic is testable without
constructing a LangChain Tool object or a real LLM at all. The make_*_tool
factories bind runtime context (a DB session, or fake embed/index clients
for tests) that the LLM itself must never control.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from customer_support_agent.integrations.rag import (
    EmbedFn,
    VectorIndex,
    default_embed_fn,
    get_policies_index,
)
from customer_support_agent.repositories import CustomerPolicyRepository, InteractionRepository


def policy_lookup_fn(
    query: str,
    product_type: Optional[str] = None,
    top_k: int = 5,
    embed_fn: EmbedFn | None = None,
    index: VectorIndex | None = None,
) -> list[dict]:
    """Embed `query` and retrieve the top_k most relevant policy clauses from
    Pinecone, optionally filtered to one product_type."""
    embed_fn = embed_fn or default_embed_fn
    index = index or get_policies_index()

    query_vector = embed_fn([query])[0]
    filter_ = {"product_type": {"$eq": product_type}} if product_type else None
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True, filter=filter_)

    matches = _get(response, "matches", []) or []
    results = []
    for m in matches:
        metadata = _get(m, "metadata", {}) or {}
        results.append(
            {
                "clause_id": metadata.get("clause_id"),
                "clause_title": metadata.get("clause_title"),
                "section": metadata.get("section"),
                "text": metadata.get("text"),
                "product_type": metadata.get("product_type"),
                "policy_version": metadata.get("policy_version"),
                "score": _get(m, "score", None),
            }
        )
    return results


def _get(obj, key, default=None):
    """Read `key` whether obj is a dict (as in our fakes/tests) or a real
    Pinecone SDK object with attribute access."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def customer_history_lookup_fn(customer_id: int, session: Session) -> dict:
    """Policy validity + recent ticket history for a customer -- the actual
    'memory' behaviour, built on the repositories from Phase 2, not a
    separate memory store.

    Reports the *most recent* policy's details (number, status, expiry)
    regardless of whether it's currently active -- has_active_policy is a
    separate boolean check. Without this distinction, a lapsed/expired
    customer would get back nothing but "no active policy", and the agent
    could never construct a useful "your policy lapsed on <date>, here's how
    to renew" response -- exactly the failure mode flagged in the Problem
    Framing Document.
    """
    policy_repo = CustomerPolicyRepository(session)
    interaction_repo = InteractionRepository(session)

    all_policies = policy_repo.list_for_customer(customer_id)  # most recent first
    most_recent_policy = all_policies[0] if all_policies else None
    active_policy = policy_repo.get_active_policy_for_customer(customer_id)
    recent_interactions = interaction_repo.get_recent_for_customer(customer_id, limit=5)

    return {
        "has_active_policy": active_policy is not None,
        "policy_number": most_recent_policy.policy_number if most_recent_policy else None,
        "policy_status": most_recent_policy.status.value if most_recent_policy else None,
        "policy_expiry_date": (
            most_recent_policy.expiry_date.isoformat() if most_recent_policy else None
        ),
        "recent_interactions": [
            {
                "ticket_id": i.ticket_id,
                "summary": i.summary,
                "escalated": i.escalated,
                "created_at": i.created_at.isoformat(),
            }
            for i in recent_interactions
        ],
    }


def make_policy_lookup_tool(
    embed_fn: EmbedFn | None = None, index: VectorIndex | None = None
):
    """Factory so tests can bind fake embed_fn/index; production code (the
    graph) calls this with no args and gets the real Pinecone-backed tool."""

    @tool
    def policy_lookup(query: str, product_type: str | None = None) -> list[dict]:
        """Search the motor insurance policy knowledge base for relevant
        clauses. Use this for coverage questions, exclusions, claim
        eligibility, or anything requiring the actual policy wording.
        product_type, if known, should be 'motor_comprehensive' or
        'motor_third_party' to narrow the search."""
        return policy_lookup_fn(query, product_type=product_type, embed_fn=embed_fn, index=index)

    return policy_lookup


def make_customer_history_tool(session: Session):
    """Factory binding the current request's DB session -- the LLM never
    sees or controls which session is used, only customer_id."""

    @tool
    def customer_history_lookup(customer_id: int) -> dict:
        """Look up whether the customer has an active policy and their
        recent ticket history. Use this for claim status, renewal, policy
        validity, or repeat-customer questions."""
        return customer_history_lookup_fn(customer_id, session)

    return customer_history_lookup
