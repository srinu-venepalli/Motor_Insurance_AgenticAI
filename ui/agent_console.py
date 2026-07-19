"""Agent Console -- queue of tickets, review the AI's draft + reasoning,
edit if needed, and approve & send or trigger (re)processing.

This is where the human-in-the-loop boundary actually lives: nothing here
lets an agent skip reviewing a draft before it's sent to a customer.
"""

import streamlit as st

from ui import api_client
from ui.theme import render_header, status_pill


def _logout():
    st.session_state.clear()
    st.rerun()


def _status_kind_and_label(ticket: dict) -> tuple[str, str]:
    if ticket["status"] == "closed":
        if ticket.get("resolution") == "rejected":
            return "rejected", "Rejected"
        return "closed", "Approved"
    if ticket.get("escalated"):
        return "escalated", "Escalated"
    if ticket.get("escalated") is None:
        return "open", "Not yet processed"
    return "open", "Awaiting review"


def _queue_sort_priority(ticket: dict) -> int:
    """Lower sorts first. Pending/actionable work should always appear
    above already-closed tickets, with escalated cases surfaced above
    plain awaiting-review ones -- an agent's queue should read like a
    to-do list, not just an ID-ordered log."""
    if ticket["status"] == "closed":
        return 3
    if ticket.get("escalated"):
        return 0
    if ticket.get("escalated") is None:
        return 2  # not yet processed
    return 1  # awaiting review


def _render_previous_tickets_accordion(customer_id: int, current_ticket_id: int) -> None:
    """Every other ticket from this same customer, newest first, one
    expander each -- this is the 'memory' the agent gets to see directly
    (the backend already uses this same data for the AI's own
    customer_history_lookup tool; this is just surfacing it to a human
    too), now with the full summary and the actual response sent, not just
    a truncated one-line table."""
    st.divider()
    st.markdown("**Previous tickets from this customer:**")
    try:
        all_tickets = api_client.list_tickets(customer_id=customer_id)
    except api_client.ApiError as exc:
        st.caption(f"Couldn't load ticket history: {exc}")
        return

    other_tickets = [t for t in all_tickets if t["ticket_id"] != current_ticket_id]
    other_tickets.sort(key=lambda t: t["ticket_id"], reverse=True)  # latest first

    if not other_tickets:
        st.caption("No previous tickets from this customer.")
        return

    for t in other_tickets:
        _, label = _status_kind_and_label(t)
        header = (
            f"#{t['ticket_id']} \u00b7 {t['opened_at'][:10]} \u00b7 "
            f"{t['category'].replace('_', ' ').title()} \u00b7 {label}"
        )
        with st.expander(header):
            try:
                other_detail = api_client.get_ticket(t["ticket_id"])
            except api_client.ApiError as exc:
                st.caption(f"Couldn't load detail: {exc}")
                continue

            other_customer_msgs = [m for m in other_detail["messages"] if m["sender"] == "customer"]
            other_human_msgs = [m for m in other_detail["messages"] if m["sender"] == "human_agent"]

            if other_customer_msgs:
                st.caption("Customer asked:")
                st.write(other_customer_msgs[0]["text"])

            if other_detail["interactions"]:
                st.caption("AI summary (internal):")
                st.write(other_detail["interactions"][-1]["summary"])

            if other_human_msgs:
                st.caption("Response sent:")
                st.success(other_human_msgs[-1]["text"])
            else:
                st.caption("(No response sent yet -- still open or escalated.)")


def render() -> None:
    agent_name = st.session_state.get("agent_name", "Agent")
    agent_role = st.session_state.get("agent_role", "support_agent")
    is_supervisor = agent_role == "supervisor"

    render_header(
        title="Agent Console",
        subtitle=f"Signed in as {agent_name} ({agent_role.replace('_', ' ')})",
        on_logout=_logout,
    )

    col_filter, col_refresh = st.columns([3, 1])
    with col_filter:
        status_filter = st.selectbox(
            "Filter",
            ["All", "Pending", "Approved", "Rejected"],
            index=0,
            label_visibility="collapsed",
        )
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            st.session_state.pop("_selected_ticket_id", None)
            st.rerun()

    # "Pending" filters by status (still open, regardless of escalation
    # state); "Approved"/"Rejected" filter by resolution (only meaningful
    # for closed tickets) -- these are two different underlying fields, so
    # map the single dropdown choice to whichever query param actually
    # applies.
    filter_status = "open" if status_filter == "Pending" else None
    filter_resolution = (
        status_filter.lower() if status_filter in ("Approved", "Rejected") else None
    )

    try:
        tickets = api_client.list_tickets(status=filter_status, resolution=filter_resolution)
    except api_client.ApiError as exc:
        st.error(f"Couldn't load the ticket queue: {exc}")
        return

    # Escalated tickets are assigned to the supervisor (see
    # make_escalate_node) -- a regular agent shouldn't see them in their
    # queue at all, not just be unable to act on them.
    if not is_supervisor:
        tickets = [t for t in tickets if not t.get("escalated")]

    if not tickets:
        st.info("No tickets in the queue.")
        return

    # Pending work first (escalated > awaiting review > not yet processed),
    # closed tickets last -- see _queue_sort_priority() for the rationale.
    # Within each tier, newest first (descending ticket_id) -- same
    # convention as _render_previous_tickets_accordion()'s history view, so the
    # two lists don't disagree about what "recent" means.
    tickets.sort(key=lambda t: (_queue_sort_priority(t), -t["ticket_id"]))

    col_list, col_detail = st.columns([2, 3])

    with col_list:
        st.caption(f"{len(tickets)} ticket(s)")
        for t in tickets:
            kind, label = _status_kind_and_label(t)
            is_selected = st.session_state.get("_selected_ticket_id") == t["ticket_id"]
            button_label = f"#{t['ticket_id']} \u00b7 {t['customer_name'] or 'Unknown'}"
            if st.button(
                button_label,
                key=f"select_{t['ticket_id']}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state["_selected_ticket_id"] = t["ticket_id"]
                st.rerun()
            st.markdown(status_pill(label, kind), unsafe_allow_html=True)
            st.write("")

    with col_detail:
        ticket_id = st.session_state.get("_selected_ticket_id")
        if ticket_id is None:
            st.info("Select a ticket from the list to review it.")
            return

        try:
            detail = api_client.get_ticket(ticket_id)
        except api_client.ApiError as exc:
            st.error(f"Couldn't load ticket #{ticket_id}: {exc}")
            return

        st.subheader(f"Ticket #{ticket_id} \u2014 {detail['customer_name'] or 'Unknown customer'}")
        st.caption(
            f"Category: {detail['category'].replace('_', ' ').title()} \u00b7 Status: {detail['status']}"
        )

        latest_interaction = detail["interactions"][-1] if detail["interactions"] else None
        is_escalated_and_open = (
            latest_interaction is not None
            and latest_interaction["escalated"]
            and detail["status"] != "closed"
        )

        if is_escalated_and_open and not is_supervisor:
            # Defense in depth: even if a regular agent somehow still has
            # this ticket selected (e.g. it was escalated after they opened
            # it, or a stale session), block the detail/action view too --
            # not just hide it from the queue list above.
            st.warning(
                "This ticket has been escalated and is assigned to a supervisor. "
                "You don't have access to review or act on it."
            )
            return

        customer_messages = [m for m in detail["messages"] if m["sender"] == "customer"]
        ai_drafts = [m for m in detail["messages"] if m["sender"] == "ai_draft"]
        human_messages = [m for m in detail["messages"] if m["sender"] == "human_agent"]

        if customer_messages:
            st.markdown("**Customer's message:**")
            st.info(customer_messages[0]["text"])

        if detail["interactions"]:
            latest = detail["interactions"][-1]
            st.markdown("**AI summary (internal):**")
            st.write(latest["summary"])
            badges = []
            badges.append(
                "\u2705 Faithfulness passed" if latest["faithfulness_pass"] else "\u26a0\ufe0f Faithfulness FAILED"
            )
            if latest["escalated"]:
                badges.append("\U0001f6a9 Escalated")
            st.caption(" \u00b7 ".join(badges))
            # Faithfulness and escalation are independent checks -- a ticket
            # can pass faithfulness and still escalate for an entirely
            # different reason (repeat customer, high-value claim, complaint
            # category, etc). Show the actual reason so that's never a
            # mystery.
            if latest["escalated"] and latest.get("escalation_reason"):
                st.caption(f"\U0001f4cb Escalation reason: {latest['escalation_reason']}")

        # Deliberately no early returns below this point -- the previous-
        # tickets table at the end should render regardless of which of
        # these three states the ticket is in.
        if detail["status"] == "closed":
            resolution_label = (
                "\u2705 Approved" if detail.get("resolution") == "approved" else "\u274c Rejected"
            )
            st.markdown(f"**Sent to customer** ({resolution_label}):")
            st.success(human_messages[-1]["text"] if human_messages else "(no record)")

        elif not ai_drafts:
            st.warning("This ticket hasn't been processed by the AI agent yet.")
            if st.button("Process with AI", type="primary"):
                with st.spinner("Running the agent..."):
                    try:
                        api_client.process_ticket(ticket_id)
                    except api_client.ApiError as exc:
                        st.error(f"Processing failed: {exc}")
                    else:
                        st.rerun()

        else:
            st.markdown("**AI draft response (edit before sending if needed):**")
            edited_text = st.text_area(
                "Draft response", value=ai_drafts[-1]["text"], height=220, label_visibility="collapsed"
            )
            st.caption(
                "Edits you make here are sent as-is -- the difference from the original AI "
                "draft is recorded for review (Phase 7 feedback)."
            )

            col_send, col_reject, col_reprocess = st.columns(3)
            with col_send:
                if st.button("Approve & Send", type="primary", use_container_width=True):
                    try:
                        api_client.approve_ticket(
                            ticket_id, final_response=edited_text, resolution="approved"
                        )
                    except api_client.ApiError as exc:
                        st.error(f"Couldn't send: {exc}")
                    else:
                        st.success("Sent to customer.")
                        st.rerun()
            with col_reject:
                if st.button("Reject & Send", use_container_width=True):
                    try:
                        api_client.approve_ticket(
                            ticket_id, final_response=edited_text, resolution="rejected"
                        )
                    except api_client.ApiError as exc:
                        st.error(f"Couldn't send: {exc}")
                    else:
                        st.success("Rejection sent to customer.")
                        st.rerun()
            with col_reprocess:
                if st.button("Re-run AI", use_container_width=True):
                    with st.spinner("Re-running the agent..."):
                        try:
                            api_client.process_ticket(ticket_id)
                        except api_client.ApiError as exc:
                            st.error(f"Processing failed: {exc}")
                        else:
                            st.rerun()

        _render_previous_tickets_accordion(detail["customer_id"], ticket_id)
