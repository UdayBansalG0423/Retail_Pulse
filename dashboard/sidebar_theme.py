"""
sidebar_theme.py
----------------
Shared sidebar styling, brand banner, and theme injection for RetailPulse.
Provides premium modern dark aesthetics, customized navigation buttons,
status pill, and styled filter controls across all dashboard pages.
"""

import streamlit as st

SIDEBAR_CSS = """
<style>
/* -------------------------------------------------------------------------
   RetailPulse Executive Sidebar Theme
   ------------------------------------------------------------------------- */

/* Overall Sidebar Container */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f19 0%, #0f172a 40%, #090d16 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.35) !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

/* Sidebar Navigation Container */
[data-testid="stSidebarNav"] {
    padding: 0.5rem 0 1rem 0 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    margin-bottom: 1rem !important;
}

/* Category header before navigation items */
[data-testid="stSidebarNav"]::before {
    content: "🧭 DASHBOARD PAGES";
    display: block;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    color: #64748b;
    margin-left: 1.25rem;
    margin-bottom: 0.5rem;
}

/* Navigation List & Items */
[data-testid="stSidebarNav"] ul {
    gap: 0.3rem !important;
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
}

[data-testid="stSidebarNav"] li a {
    border-radius: 9px !important;
    padding: 0.55rem 0.95rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #cbd5e1 !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: 1px solid transparent !important;
    background: rgba(255, 255, 255, 0.015) !important;
    display: flex !important;
    align-items: center !important;
}

/* Replace default 'app' text label with '🏠 Executive Overview' */
[data-testid="stSidebarNav"] li:first-child a span {
    visibility: hidden;
    position: relative;
}
[data-testid="stSidebarNav"] li:first-child a span::after {
    visibility: visible;
    position: absolute;
    left: 0;
    top: 0;
    content: "🏠 Executive Overview";
    font-weight: 600;
    color: inherit;
}

/* Hover state for nav links */
[data-testid="stSidebarNav"] li a:hover {
    background: rgba(255, 255, 255, 0.07) !important;
    color: #ffffff !important;
    transform: translateX(4px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
}

/* Active Page Highlight */
[data-testid="stSidebarNav"] li a[aria-current="page"],
[data-testid="stSidebarNav"] li a.st-emotion-cache-1yak103 {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.24) 0%, rgba(139, 92, 246, 0.15) 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-left: 3.5px solid #3b82f6 !important;
    border-top: 1px solid rgba(59, 130, 246, 0.35) !important;
    border-right: 1px solid rgba(59, 130, 246, 0.25) !important;
    border-bottom: 1px solid rgba(59, 130, 246, 0.25) !important;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.2) !important;
}

/* Sidebar Headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.6rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
}

/* Control labels & inputs */
section[data-testid="stSidebar"] label {
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"],
section[data-testid="stSidebar"] div[data-baseweb="input"] {
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background-color: rgba(15, 23, 42, 0.75) !important;
    transition: border-color 0.2s ease !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"]:hover,
section[data-testid="stSidebar"] div[data-baseweb="input"]:hover {
    border-color: rgba(59, 130, 246, 0.4) !important;
}

/* Sliders */
section[data-testid="stSidebar"] div[data-testid="stSlider"] {
    padding-top: 2px !important;
    padding-bottom: 6px !important;
}

/* Styled expanders in sidebar */
section[data-testid="stSidebar"] div[data-testid="stExpander"] {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    background: rgba(15, 23, 42, 0.4) !important;
}

/* Custom scrollbar */
section[data-testid="stSidebar"]::-webkit-scrollbar {
    width: 5px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar-track {
    background: transparent;
}
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 4px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.25);
}
</style>
"""


def apply_sidebar_theme():
    """Injects premium styling and brand card into the Streamlit sidebar."""
    # Inject CSS
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    
    # Render Brand Header Card at the top of the sidebar
    st.sidebar.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(10px);
    ">
        <div style="display: flex; align-items: center; gap: 11px;">
            <div style="
                width: 36px;
                height: 36px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                border-radius: 9px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                box-shadow: 0 2px 10px rgba(59, 130, 246, 0.45);
            ">🛍️</div>
            <div style="flex-grow: 1;">
                <div style="font-weight: 700; font-size: 15px; color: #ffffff; letter-spacing: -0.02em; line-height: 1.2;">RetailPulse</div>
                <div style="font-size: 10.5px; color: #94a3b8; font-weight: 500;">AI Decision Platform</div>
            </div>
            <span style="
                background: rgba(59, 130, 246, 0.16);
                color: #60a5fa;
                padding: 2px 7px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 10px;
                border: 1px solid rgba(59, 130, 246, 0.35);
                letter-spacing: 0.02em;
            ">v2.4</span>
        </div>
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 9px;
            padding-top: 8px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            font-size: 10.5px;
        ">
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="width: 6.5px; height: 6.5px; background: #22c55e; border-radius: 50%; display: inline-block; box-shadow: 0 0 6px #22c55e;"></span>
                <span style="color: #cbd5e1; font-weight: 500;">Models Active</span>
            </div>
            <span style="color: #64748b; font-size: 10px;">Zidio Internship</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_footer():
    """Renders a sleek system telemetry card at the bottom of the sidebar."""
    st.sidebar.markdown("""
    <div style="
        margin-top: 22px;
        padding: 11px 13px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        font-size: 10.5px;
    ">
        <div style="color: #94a3b8; font-weight: 700; font-size: 10px; letter-spacing: 0.06em; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
            <span>SYSTEM METRICS</span>
            <span style="color: #38bdf8; font-weight: 600;">HEALTHY</span>
        </div>
        <div style="color: #64748b; line-height: 1.55;">
            <div>📦 524,231 Transactions</div>
            <div>👥 4,338 Customer Cohorts</div>
            <div>⚡ Prophet + XGBoost + SHAP</div>
        </div>
        <div style="margin-top: 7px; padding-top: 7px; border-top: 1px solid rgba(255, 255, 255, 0.06); color: #475569; font-size: 9.5px; text-align: center;">
            RetailPulse Analytics Engine
        </div>
    </div>
    """, unsafe_allow_html=True)
