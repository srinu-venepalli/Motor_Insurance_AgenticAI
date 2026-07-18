"""Assembles the LangGraph graph:

    classify_ticket -> agent_reasoning -> summarize_and_draft
        -> faithfulness_check -> [escalation_gate] -> escalate_to_agent | present_to_human

llm/embed_fn/index are all injectable so this same function builds either
the real graph (production: build_graph(session)) or a fully-faked one for
tests (build_graph(session, llm=fake_llm, embed_fn=fake_embed, index=fake_index)).
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from customer_support_agent.core import configure_langsmith
from customer_support_agent.graph.llm import get_llm
from customer_support_agent.graph.nodes import (
    faithfulness_check,
    make_agent_reasoning_node,
    make_classify_node,
    make_escalate_node,
    make_escalation_gate,
    make_fetch_customer_context_node,
    make_present_node,
    make_summarize_node,
)
from customer_support_agent.graph.state import AgentState
from customer_support_agent.graph.tools import make_customer_history_tool, make_policy_lookup_tool


def build_graph(session: Session, llm=None, embed_fn=None, index=None):
    """Compile the agent graph bound to `session` (and, for tests, fake
    llm/embed_fn/index). Calling this per-ticket (rather than once globally)
    keeps the DB session and tool bindings scoped correctly per request."""
    configure_langsmith()
    llm = llm or get_llm()

    policy_tool = make_policy_lookup_tool(embed_fn=embed_fn, index=index)
    history_tool = make_customer_history_tool(session)

    graph = StateGraph(AgentState)
    graph.add_node("classify_ticket", make_classify_node(llm))
    graph.add_node("fetch_customer_context", make_fetch_customer_context_node(session))
    graph.add_node("agent_reasoning", make_agent_reasoning_node(llm, policy_tool, history_tool))
    graph.add_node("summarize_and_draft", make_summarize_node(llm))
    graph.add_node("faithfulness_check", faithfulness_check)
    graph.add_node("escalate_to_agent", make_escalate_node(session))
    graph.add_node("present_to_human", make_present_node(session))

    graph.set_entry_point("classify_ticket")
    graph.add_edge("classify_ticket", "fetch_customer_context")
    graph.add_edge("fetch_customer_context", "agent_reasoning")
    graph.add_edge("agent_reasoning", "summarize_and_draft")
    graph.add_edge("summarize_and_draft", "faithfulness_check")
    graph.add_conditional_edges(
        "faithfulness_check",
        make_escalation_gate(),
        {"escalate": "escalate_to_agent", "present": "present_to_human"},
    )
    graph.add_edge("escalate_to_agent", END)
    graph.add_edge("present_to_human", END)

    return graph.compile()
