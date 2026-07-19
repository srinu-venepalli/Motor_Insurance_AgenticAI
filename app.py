"""Streamlit entrypoint: login gate, then role-based routing to either the
Customer Portal or the Agent Console.

Run with:
    uv run streamlit run app.py

Login is intentionally simplified for this demo: a single shared password
(settings.demo_shared_password, default "password1") for everyone --
customers pick their name from a dropdown (fetched from GET /customers),
agents pick theirs from the seeded roster. Customer IDs are resolved
internally from the selected name and never shown or typed by the user --
asking someone to know their own internal database ID isn't realistic UX.
This is NOT production authentication; a real deployment needs per-user
credentials and hashed passwords.
"""

from pathlib import Path

import streamlit as st

from customer_support_agent.core import settings
from customer_support_agent.services import AGENT_NAMES, AGENT_ROLES
from ui import api_client
from ui.theme import inject_base_style

st.set_page_config(page_title="Motor Insurance Support", page_icon="\U0001f6e1\ufe0f", layout="centered")

_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.svg"


def _render_login_header() -> None:
    logo_svg = _LOGO_PATH.read_text().replace(
        "<svg ", '<svg style="width:64px;height:64px;display:block;margin:0 auto;" '
    )
    st.markdown(logo_svg, unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align:center;margin-top:0.5rem;'>Motor Insurance Support</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#6B7280;'>Sign in to continue</p>",
        unsafe_allow_html=True,
    )


def _login_form(placeholder) -> None:
    with placeholder.container():
        _render_login_header()

        role = st.radio("I am a...", ["Customer", "Support Agent"], horizontal=True)

        customer_options: dict[str, int] = {}
        if role == "Customer":
            try:
                customers = api_client.list_customers()
            except api_client.ApiError as exc:
                st.error(f"Couldn't load the customer list: {exc}")
                return
            customer_options = {c["name"]: c["id"] for c in customers}

        with st.form("login_form"):
            selected_customer_name = None
            agent_name = None
            if role == "Customer":
                if not customer_options:
                    st.warning("No customers found -- run scripts/seed_customers.py first.")
                selected_customer_name = st.selectbox("Customer", list(customer_options.keys()))
            else:
                agent_name = st.selectbox("Agent name", AGENT_NAMES)
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if not submitted:
            return

        if password != settings.demo_shared_password:
            st.error("Incorrect password.")
            return

        if role == "Customer":
            customer_id = customer_options.get(selected_customer_name)
            if customer_id is None:
                st.error("Please select a valid customer.")
                return
            st.session_state["role"] = "customer"
            st.session_state["customer_id"] = customer_id
            st.session_state["customer_name"] = selected_customer_name
        else:
            st.session_state["role"] = "agent"
            st.session_state["agent_name"] = agent_name
            st.session_state["agent_role"] = AGENT_ROLES.get(agent_name, "support_agent")

    # Explicitly clear the placeholder's contents BEFORE rerunning, rather
    # than relying on Streamlit's automatic diffing between runs -- that's
    # what was causing the login form to visibly linger/fade while the
    # portal was already rendering underneath it.
    placeholder.empty()
    st.rerun()


def main() -> None:
    inject_base_style()

    if "role" not in st.session_state:
        placeholder = st.empty()
        _login_form(placeholder)
        return

    if st.session_state["role"] == "customer":
        from ui import customer_portal

        customer_portal.render()
    else:
        from ui import agent_console

        agent_console.render()


if __name__ == "__main__":
    main()
