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

    tab_new, tab_history = st.tabs(["New Ticket", "My Tickets"])

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
                    st.success(f"Ticket #{ticket_id} submitted! Our support team is reviewing it.")
                    st.session_state["_just_submitted"] = ticket_id

    with tab_history:
        try:
            tickets = api_client.list_tickets(customer_id=customer_id)
        except api_client.ApiError as exc:
            st.error(f"Couldn't load your tickets: {exc}")
            return

        if not tickets:
            st.info("You haven't submitted any tickets yet.")
            return

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

            with st.expander(
                f"Ticket #{t['ticket_id']} \u2014 {t['category'].replace('_', ' ').title()}",
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
