"""Shared retry policy for upstream LLM calls.

Confirmed via scripts/diagnose_tool_calling.py: the IITM/Vocareum
OpenAI-compatible proxy has occasionally returned a 200 OK whose body
contains more than one JSON document concatenated together, specifically
during tool-calling -- intermittent proxy-side flakiness, not a structural
bug in how requests are shaped (four targeted reproduction attempts all
came back clean). Retrying with some runway is the right mitigation.

Used by both /tickets/{id}/process (graph.invoke) and /chat (run_chat_turn)
since both go through tool-calling and can hit the same issue -- kept here
once rather than duplicated in each router.
"""

import json

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

llm_call_retry = retry(
    retry=retry_if_exception_type((json.JSONDecodeError, openai.APIError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
