"""Sneddon in Sport branding, applied to every Streamlit page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"
LOGO = ASSETS / "logo.svg"
LOGO_ICON = ASSETS / "logo_icon.svg"

SIDEBAR_LOGO_HEIGHT = "5.5rem"  # st.logo(size="large") caps out far smaller than this

_CSS = f"""
<style>
[data-testid="stSidebarHeader"] {{ height: auto; min-height: {SIDEBAR_LOGO_HEIGHT}; padding-top: 1rem; }}
[data-testid="stSidebarHeader"] img, img[data-testid="stLogo"] {{
    height: {SIDEBAR_LOGO_HEIGHT} !important; max-width: 100%; width: auto;
}}
</style>
"""


def apply_branding(*, show_header_logo: bool = False) -> None:
    """Show a large logo at the top of the sidebar. Call right after set_page_config."""
    st.logo(str(LOGO), icon_image=str(LOGO_ICON), size="large")
    st.markdown(_CSS, unsafe_allow_html=True)
    if show_header_logo:
        st.image(str(LOGO), width=460)
