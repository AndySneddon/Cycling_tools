"""Sneddon in Sport branding, applied to every Streamlit page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"
LOGO = ASSETS / "logo.svg"
LOGO_ICON = ASSETS / "logo_icon.svg"


def apply_branding(*, show_header_logo: bool = False) -> None:
    """Show the logo at the top of the sidebar (and collapsed-sidebar icon). Call right after set_page_config."""
    st.logo(str(LOGO), icon_image=str(LOGO_ICON), size="large")
    if show_header_logo:
        st.image(str(LOGO), width=300)
