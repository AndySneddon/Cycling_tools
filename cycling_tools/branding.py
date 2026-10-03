"""Sneddon in Sport branding and shared page styling for every Streamlit page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"
LOGO = ASSETS / "logo.svg"
LOGO_ICON = ASSETS / "logo_icon.svg"
PAGE_ICON = str(LOGO_ICON)

SIDEBAR_LOGO_HEIGHT = "5.5rem"  # st.logo(size="large") caps out far smaller than this
ACCENT = "#F25C2A"
BLUE = "#4FC3F7"

# Colours use currentColor / translucent greys so cards read correctly in both light and dark themes.
_CSS = f"""
<style>
.block-container {{ max-width: 1280px; padding-top: 2.2rem; padding-bottom: 4rem; }}

/* sidebar logo */
[data-testid="stSidebarHeader"] {{ height: auto; min-height: {SIDEBAR_LOGO_HEIGHT}; padding-top: 1rem; }}
[data-testid="stSidebarHeader"] img, img[data-testid="stLogo"] {{
    height: {SIDEBAR_LOGO_HEIGHT} !important; max-width: 100%; width: auto;
}}
[data-testid="stSidebar"] h2 {{
    font-size: 0.78rem; letter-spacing: .12em; text-transform: uppercase; opacity: .7; margin-top: .6rem;
}}

/* page header */
.sis-eyebrow {{ color: {ACCENT}; font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }}
.sis-title {{ font-size: 2.3rem; font-weight: 800; line-height: 1.15; margin: .15rem 0 .35rem 0; }}
.sis-sub {{ opacity: .75; font-size: 1.02rem; max-width: 62rem; margin-bottom: .6rem; }}
.sis-rule {{ height: 4px; width: 72px; border-radius: 4px; background: linear-gradient(90deg, {ACCENT}, {BLUE}); margin: .5rem 0 1.4rem 0; }}

/* section headings */
h2, h3 {{ letter-spacing: -.01em; }}
[data-testid="stMain"] h3 {{ margin-top: 1.4rem; padding-bottom: .25rem; border-bottom: 1px solid rgba(128,128,128,.25); }}

/* metric cards */
[data-testid="stMetric"] {{
    background: color-mix(in srgb, currentColor 5%, transparent);
    border: 1px solid rgba(128,128,128,.28); border-left: 4px solid {ACCENT};
    border-radius: 14px; padding: 12px 14px;
}}
[data-testid="stMetricLabel"] {{ opacity: .75; font-size: .82rem; }}
[data-testid="stMetricLabel"] p {{ white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.2; }}
/* never truncate the headline numbers: shrink and wrap instead of showing "0.2..." */
[data-testid="stMetricValue"] {{ font-weight: 750; font-size: 1.4rem; line-height: 1.2; overflow-wrap: anywhere; }}
[data-testid="stMetricValue"] div {{ white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}
[data-testid="stMetricDelta"] {{ font-size: .75rem; }}

/* bordered containers (cards) */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px; }}

/* tabs */
button[data-baseweb="tab"] {{ font-weight: 600; padding: .6rem 1rem; }}

/* buttons */
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; font-weight: 650; padding: .45rem 1.1rem; }}

/* alerts and expanders */
[data-testid="stAlert"] {{ border-radius: 12px; }}
[data-testid="stExpander"] {{ border-radius: 12px; }}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
.stPlotlyChart {{ border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 6px; }}
</style>
"""


def apply_branding(*, show_header_logo: bool = False) -> None:
    """Sidebar logo plus shared styling. Call right after set_page_config."""
    st.logo(str(LOGO), icon_image=str(LOGO_ICON), size="large")
    st.markdown(_CSS, unsafe_allow_html=True)
    if show_header_logo:
        st.image(str(LOGO), width=460)


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    """Consistent page title block."""
    parts = []
    if eyebrow:
        parts.append(f'<div class="sis-eyebrow">{eyebrow}</div>')
    parts.append(f'<div class="sis-title">{title}</div>')
    if subtitle:
        parts.append(f'<div class="sis-sub">{subtitle}</div>')
    parts.append('<div class="sis-rule"></div>')
    st.markdown("\n".join(parts), unsafe_allow_html=True)
