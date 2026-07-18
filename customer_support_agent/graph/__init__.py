"""Graph package -- the LangGraph agent core.

build_graph(session) is the main entrypoint: compiles the full
classify -> agent_reasoning -> summarize_and_draft -> faithfulness_check ->
[escalate_to_agent | present_to_human] graph, bound to a DB session.
"""

from customer_support_agent.graph.build_graph import build_graph
from customer_support_agent.graph.llm import get_llm
from customer_support_agent.graph.state import AgentState

__all__ = ["build_graph", "get_llm", "AgentState"]
