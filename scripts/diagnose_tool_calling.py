"""Diagnose whether tool-calling (the OpenAI 'tools' parameter) specifically
triggers malformed responses from your OpenAI-compatible endpoint.

Makes two raw HTTP calls (bypassing the OpenAI SDK's response parsing, so a
malformed body can actually be inspected instead of just raising an
exception):
    1. A plain chat completion, no tools -- the control.
    2. The same call, but with one simple tool bound -- the variable.

If (1) succeeds cleanly and (2) fails with a JSON parse error, that
confirms the proxy has a bug specifically in how it handles tool-calling
responses (not a general outage, not a credits/quota issue -- those would
fail both calls, usually with a 429/402, not a malformed 200).

Usage:
    uv run python scripts/diagnose_tool_calling.py
"""

import json

import requests

from customer_support_agent.core import settings

HEADERS = {
    "Authorization": f"Bearer {settings.openai_api_key}",
    "Content-Type": "application/json",
}
URL = f"{settings.openai_base_url.rstrip('/')}/chat/completions"


def _print_raw_around_failure(text: str, error: json.JSONDecodeError) -> None:
    start = max(0, error.pos - 150)
    end = min(len(text), error.pos + 150)
    print(f"--- Raw response body, 150 chars around the parse failure (pos={error.pos}) ---")
    print(text[start:end])
    print("--- end excerpt ---")


def test_plain_call() -> bool:
    print("=" * 72)
    print("TEST 1: Plain chat completion, NO tools (the control)")
    print("=" * 72)
    payload = {
        "model": settings.chat_model,
        "messages": [{"role": "user", "content": "Say hello in exactly one word."}],
    }
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
    print(f"HTTP status: {resp.status_code}")
    print(f"Response body length: {len(resp.text)} chars")
    try:
        data = resp.json()
        print(f"JSON parsed OK. Content: {data['choices'][0]['message']['content']!r}")
        return True
    except json.JSONDecodeError as exc:
        print(f"JSON PARSE FAILED: {exc}")
        _print_raw_around_failure(resp.text, exc)
        return False


def test_tool_call() -> bool:
    print("\n" + "=" * 72)
    print("TEST 2: Chat completion WITH a tool bound (the variable)")
    print("=" * 72)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    payload = {
        "model": settings.chat_model,
        "messages": [{"role": "user", "content": "What's the weather in Chennai right now?"}],
        "tools": tools,
    }
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
    print(f"HTTP status: {resp.status_code}")
    print(f"Response body length: {len(resp.text)} chars")
    try:
        data = resp.json()
        print("JSON parsed OK.")
        print(json.dumps(data["choices"][0], indent=2))
        return True
    except json.JSONDecodeError as exc:
        print(f"JSON PARSE FAILED: {exc}")
        _print_raw_around_failure(resp.text, exc)
        return False


def test_multi_turn_two_tools_with_large_result() -> bool:
    """Simulates the SECOND call in the real agent_reasoning loop: two tools
    bound (not one), with a realistic-size tool result already in the
    message history (a ToolMessage containing ~5 retrieved policy clauses,
    just like the real policy_lookup tool returns). This is the scenario
    the simple TEST 2 above doesn't cover -- if this is where it breaks,
    the bug is likely payload-size-related, not tool-calling-in-general."""
    print("\n" + "=" * 72)
    print("TEST 3: Two tools bound + a realistic-size tool result fed back in")
    print("=" * 72)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "policy_lookup",
                "description": "Search the motor insurance policy knowledge base for relevant clauses.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "product_type": {"type": ["string", "null"]},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "customer_history_lookup",
                "description": "Look up whether the customer has an active policy and their recent ticket history.",
                "parameters": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "integer"}},
                    "required": ["customer_id"],
                },
            },
        },
    ]

    # Roughly mimics real policy_lookup_fn output: 5 clauses, realistic
    # length, JSON-serialized -- this is exactly what a real ToolMessage's
    # content looks like in agent_reasoning's loop.
    fake_tool_result = json.dumps(
        [
            {
                "clause_id": f"OD-{i}.0",
                "clause_title": f"Sample Clause {i}",
                "section": "What This Policy Covers",
                "text": (
                    "Own damage cover applies to collision or impact damage sustained "
                    "while the vehicle is parked, even where the responsible party "
                    "cannot be identified, subject to the standard policy excess and a "
                    "mandatory claim inspection by an authorized surveyor. " * 2
                ),
                "product_type": "motor_comprehensive",
                "policy_version": "v3.2",
                "score": round(0.9 - i * 0.02, 2),
            }
            for i in range(5)
        ]
    )

    messages = [
        {
            "role": "system",
            "content": "You are an assistant helping a human insurance support agent.",
        },
        {
            "role": "user",
            "content": (
                "Ticket category: coverage_question\nCustomer ID: 3\n"
                "Ticket: Does my policy cover a cracked windscreen?"
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_test123",
                    "type": "function",
                    "function": {
                        "name": "policy_lookup",
                        "arguments": json.dumps({"query": "cracked windscreen cover"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_test123", "content": fake_tool_result},
    ]

    payload = {"model": settings.chat_model, "messages": messages, "tools": tools}
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
    print(f"HTTP status: {resp.status_code}")
    print(f"Response body length: {len(resp.text)} chars")
    try:
        data = resp.json()
        print("JSON parsed OK.")
        print(json.dumps(data["choices"][0], indent=2)[:1200])
        return True
    except json.JSONDecodeError as exc:
        print(f"JSON PARSE FAILED: {exc}")
        _print_raw_around_failure(resp.text, exc)
        return False


def test_final_response_after_both_tools() -> bool:
    """The one response shape not yet tested: BOTH tools already called and
    both results already in history, so the model should stop calling
    tools and return an actual natural-language answer (finish_reason
    'stop', a real paragraph of text in 'content') -- structurally very
    different from the compact tool_calls JSON shape TESTs 2 and 3 got back.
    If this is what corrupts, we've found the real trigger."""
    print("\n" + "=" * 72)
    print("TEST 4: Final natural-language response after both tools already ran")
    print("=" * 72)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "policy_lookup",
                "description": "Search the motor insurance policy knowledge base for relevant clauses.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "product_type": {"type": ["string", "null"]},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "customer_history_lookup",
                "description": "Look up whether the customer has an active policy and their recent ticket history.",
                "parameters": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "integer"}},
                    "required": ["customer_id"],
                },
            },
        },
    ]

    fake_policy_result = json.dumps(
        [
            {
                "clause_id": f"OD-{i}.0",
                "clause_title": f"Sample Clause {i}",
                "section": "What This Policy Covers",
                "text": (
                    "Own damage cover applies to collision or impact damage sustained "
                    "while the vehicle is parked, even where the responsible party "
                    "cannot be identified, subject to the standard policy excess and a "
                    "mandatory claim inspection by an authorized surveyor. " * 2
                ),
                "product_type": "motor_comprehensive",
                "policy_version": "v3.2",
                "score": round(0.9 - i * 0.02, 2),
            }
            for i in range(5)
        ]
    )
    fake_history_result = json.dumps(
        {
            "has_active_policy": True,
            "policy_number": "POL-MC-000102",
            "policy_status": "active",
            "policy_expiry_date": "2026-12-27",
            "recent_interactions": [
                {
                    "ticket_id": 5,
                    "summary": "Customer inquires about coverage for a cracked windscreen.",
                    "escalated": False,
                    "created_at": "2026-07-14T06:25:50.422167+00:00",
                }
            ],
            "tickets_last_30_days": 3,
        }
    )

    messages = [
        {"role": "system", "content": "You are an assistant helping a human insurance support agent."},
        {
            "role": "user",
            "content": (
                "Ticket category: coverage_question\nCustomer ID: 3\n"
                "Ticket: Does my policy cover a cracked windscreen?"
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_policy",
                    "type": "function",
                    "function": {
                        "name": "policy_lookup",
                        "arguments": json.dumps({"query": "cracked windscreen cover"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_policy", "content": fake_policy_result},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_history",
                    "type": "function",
                    "function": {
                        "name": "customer_history_lookup",
                        "arguments": json.dumps({"customer_id": 3}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_history", "content": fake_history_result},
    ]

    payload = {"model": settings.chat_model, "messages": messages, "tools": tools}
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
    print(f"HTTP status: {resp.status_code}")
    print(f"Response body length: {len(resp.text)} chars")
    try:
        data = resp.json()
        print("JSON parsed OK.")
        print(f"finish_reason: {data['choices'][0]['finish_reason']}")
        content = data["choices"][0]["message"].get("content")
        print(f"content: {content!r}")
        return True
    except json.JSONDecodeError as exc:
        print(f"JSON PARSE FAILED: {exc}")
        _print_raw_around_failure(resp.text, exc)
        return False


def main() -> None:
    print(f"Endpoint: {URL}")
    print(f"Model:    {settings.chat_model}\n")

    plain_ok = test_plain_call()
    tool_ok = test_tool_call()
    multi_turn_ok = test_multi_turn_two_tools_with_large_result()
    final_ok = test_final_response_after_both_tools()

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    if plain_ok and tool_ok and multi_turn_ok and not final_ok:
        print(
            "CONFIRMED: the failure is specifically in the FINAL natural-language\n"
            "response (finish_reason='stop', real text content) after tool results are\n"
            "already in history -- not tool-calling in general, and not just payload size.\n"
            "Recommended fix: after the last tool result comes back, make the final\n"
            "response call WITHOUT tools bound (a plain llm.invoke, like\n"
            "summarize_and_draft already does reliably) instead of keeping\n"
            "bind_tools() active for a call that no longer needs to offer any tools."
        )
    elif plain_ok and tool_ok and multi_turn_ok and final_ok:
        print(
            "All four succeeded here. At this point the failure may genuinely be\n"
            "intermittent (proxy-side flakiness under load/rate limits at certain times\n"
            "of day) rather than deterministically reproducible -- the retry logic\n"
            "already in place is the right mitigation, and it's worth just monitoring\n"
            "whether it recurs rather than changing the architecture further."
        )
    else:
        print("An earlier test failed -- see details above for exactly which one and why.")


if __name__ == "__main__":
    main()
