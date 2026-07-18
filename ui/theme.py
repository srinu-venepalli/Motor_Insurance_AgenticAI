"""Shared visual chrome: CSS injection, logo, and a top header bar.

Kept separate from page logic so both the Customer Portal and Agent Console
render identically without duplicating markup.
"""

from pathlib import Path

import streamlit as st

_LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logo.svg"

_CUSTOM_CSS = """
<style>
    /* Hide Streamlit's default chrome for a cleaner, branded look */
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}

    .block-container {padding-top: 1.5rem; max-width: 900px;}

    .app-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E5E9F0;
    }
    .app-header img {width: 40px; height: 40px;}
    .app-header .title {font-size: 1.35rem; font-weight: 700; color: #1A1D23;}
    .app-header .subtitle {font-size: 0.85rem; color: #6B7280; margin-top: -2px;}

    .status-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-open {background: #FEF3C7; color: #92400E;}
    .status-closed {background: #DCFCE7; color: #166534;}
    .status-escalated {background: #FEE2E2; color: #991B1B;}
    .status-rejected {background: #FEE2E2; color: #7F1D1D; font-weight: 700;}
</style>
"""


def inject_base_style() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, on_logout=None) -> None:
    inject_base_style()
    raw_svg = _LOGO_PATH.read_text().replace("<svg ", '<svg style="width:40px;height:40px;" ')
    # Flatten to a single line -- the SVG file itself is nicely indented for
    # readability, but embedding that multi-line content re-introduces the
    # same "Markdown sees leading whitespace, treats it as a code block"
    # issue this function is specifically written to avoid.
    logo_svg = " ".join(line.strip() for line in raw_svg.splitlines())

    col_header, col_logout = st.columns([5, 1])
    with col_header:
        html = (
            f'<div class="app-header">{logo_svg}'
            f'<div><div class="title">{title}</div>'
            f'<div class="subtitle">{subtitle}</div></div></div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    with col_logout:
        if on_logout is not None:
            st.write("")
            if st.button("Log out", use_container_width=True):
                on_logout()


def status_pill(label: str, kind: str) -> str:
    """kind: 'open' | 'closed' | 'escalated' | 'rejected'"""
    return f'<span class="status-pill status-{kind}">{label}</span>'
