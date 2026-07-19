"""Customer Portal -- submit a new ticket, view past ticket history.

The customer never sees the AI's draft directly (only after a human agent
approves and sends it) -- that's the human-in-the-loop boundary this whole
project is built around, not an oversight.
"""

import streamlit as st

from customer_support_agent.core import get_logger
from ui import api_client
from ui.theme import render_header, status_pill

logger = get_logger(__name__)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _logout():
    st.session_state.clear()
    st.rerun()


def render() -> None:
    customer_id = st.session_state["customer_id"]
    customer_name = st.session_state.get("customer_name", f"Customer #{customer_id}")

    render_header(
        title="Motor Insurance Support",
        subtitle=f"Welcome back, {customer_name}",
        on_logout=_logout,
    )

    tab_new, tab_history, tab_chat = st.tabs(["New Ticket", "My Tickets", "AI Assistant"])

    with tab_new:
        st.write("Describe your question or issue below, and our team will get back to you.")
        ticket_text = st.chat_input("e.g. Does my policy cover a cracked windscreen?")

        if ticket_text:
            with st.spinner("Submitting your ticket..."):
                try:
                    created = api_client.create_ticket(customer_id, ticket_text)
                except api_client.ApiError as exc:
                    st.error(f"Something went wrong submitting your ticket: {exc}")
                else:
                    ticket_id = created["ticket_id"]
                    try:
                        # Kick off AI processing immediately so an agent has a
                        # draft ready to review -- but if this specific step
                        # fails (e.g. a transient upstream hiccup), the ticket
                        # itself was still created successfully, so don't
                        # scare the customer with an error. An agent can
                        # retry via "Process with AI" in the console.
                        api_client.process_ticket(ticket_id)
                    except api_client.ApiError:
                        logger.warning(
                            "Auto-processing failed for ticket %s right after submission; "
                            "an agent will need to trigger it manually.",
                            ticket_id,
                        )
                    # st.toast (not st.success) -- a transient notification
                    # that disappears on its own after a few seconds. A
                    # static st.success banner only clears on the next full
                    # rerun (switching tabs, submitting again, etc.), so it
                    # would otherwise just sit there indefinitely.
                    st.toast(f"Ticket #{ticket_id} submitted! Our support team is reviewing it.", icon="\u2705")
                    st.session_state["_just_submitted"] = ticket_id

    with tab_history:
        col_refresh, _ = st.columns([1, 4])
        with col_refresh:
            if st.button("Refresh", key="refresh_my_tickets", use_container_width=True):
                st.rerun()

        try:
            tickets = api_client.list_tickets(customer_id=customer_id)
        except api_client.ApiError as exc:
            st.error(f"Couldn't load your tickets: {exc}")
            return

        if not tickets:
            st.info("You haven't submitted any tickets yet.")
            return

        # Position within THIS customer's own history (1st, 2nd, ...) --
        # purely a friendly label shown alongside the real ticket ID, never
        # replacing it. The real ID stays the single source of truth (it's
        # what the Agent Console uses too), computed from this customer's
        # tickets in the order they were actually opened, independent of
        # whatever order the list happens to render in.
        chronological = sorted(tickets, key=lambda t: t["ticket_id"])
        sequence_number = {t["ticket_id"]: i + 1 for i, t in enumerate(chronological)}

        for t in tickets:
            if t["status"] == "closed":
                if t.get("resolution") == "rejected":
                    kind, label = "rejected", "Rejected"
                else:
                    kind, label = "closed", "Approved"
            elif t.get("escalated"):
                kind, label = "escalated", "With a specialist"
            else:
                kind, label = "open", "Under review"

            seq = sequence_number.get(t["ticket_id"])
            seq_label = f" (your {_ordinal(seq)} ticket)" if seq else ""

            with st.expander(
                f"Ticket #{t['ticket_id']}{seq_label} \u2014 {t['category'].replace('_', ' ').title()}",
                expanded=(t["ticket_id"] == st.session_state.get("_just_submitted")),
            ):
                st.markdown(status_pill(label, kind), unsafe_allow_html=True)
                st.write("")

                try:
                    detail = api_client.get_ticket(t["ticket_id"])
                except api_client.ApiError as exc:
                    st.error(f"Couldn't load ticket detail: {exc}")
                    continue

                for msg in detail["messages"]:
                    if msg["sender"] == "customer":
                        with st.chat_message("user"):
                            st.write(msg["text"])
                    elif msg["sender"] == "human_agent":
                        with st.chat_message("assistant"):
                            st.write(msg["text"])
                    # ai_draft messages are intentionally not shown here --
                    # unapproved drafts are agent-console-only.

                if t["status"] != "closed":
                    with st.chat_message("assistant"):
                        st.write(
                            "Thanks for reaching out \u2014 our support team is reviewing "
                            "your request and will respond shortly."
                        )

    with tab_chat:
        st.write(
            "Ask about your policy number, expiry date, ticket status, how to renew, "
            "or describe a new issue \u2014 I can open a ticket for our team if needed."
        )
        st.caption(
            "For real coverage or claim decisions, I'll open a support ticket for a human "
            "agent to review rather than deciding myself."
        )

        # Short-term memory: held in this browser session only, reset on
        # logout (session_state.clear() in _logout()) -- not persisted
        # server-side. Long-term memory (policy/ticket history) is read
        # fresh from Postgres on every turn via the chat tools.
        if "_chat_history" not in st.session_state:
            st.session_state["_chat_history"] = []

        for turn in st.session_state["_chat_history"]:
            role = "user" if turn["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.write(turn["content"])

        user_msg = st.chat_input("Ask me anything about your policy or tickets...")
        if user_msg:
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                # A placeholder inside the actual chat bubble (not a
                # generic spinner off to the side) -- reads like a real
                # "typing..." indicator, and makes it clear the assistant
                # is doing multi-step work (looking up your policy,
                # creating a ticket, etc.) that can genuinely take 10-20+
                # seconds through the LLM provider, not that the app has
                # frozen.
                placeholder = st.empty()
                placeholder.markdown("_Thinking..._")
                try:
                    result = api_client.chat(
                        customer_id, user_msg, st.session_state["_chat_history"]
                    )
                except api_client.ApiError as exc:
                    placeholder.error(f"Something went wrong: {exc}")
                else:
                    st.session_state["_chat_history"] = [
                        dict(turn) for turn in result["history"]
                    ]
                    placeholder.write(result["reply"])
