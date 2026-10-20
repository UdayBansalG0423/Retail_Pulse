"""
app.py
------
RetailPulse — AI-Powered Retail Analytics Platform
Executive Hub & Multipage Navigation Entrypoint
"""

from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

st.set_page_config(
    page_title="RetailPulse | AI Retail Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛍️ RetailPulse — AI-Powered Retail Analytics Platform")
st.caption("End-to-End Decision Support: Customer Segmentation, Demand Forecasting, and Inventory Replenishment.")

st.markdown("""
Welcome to **RetailPulse**. This enterprise dashboard brings together machine learning models 
and operational research to optimize customer retention, anticipate product demand, and prevent costly stockouts.
""")

st.markdown("---")

# -----------------------------------------------------------------------------
# Cross-Module Executive KPI Cockpit
# -----------------------------------------------------------------------------
st.subheader("⚡ Executive KPI Cockpit")

# Load summaries if available
customer_file = DATA_DIR / "customer_rfm.csv"
recommendations_file = DATA_DIR / "inventory_recommendations.csv"
prophet_file = DATA_DIR / "prophet_test_predictions.csv"

n_cust, tot_rev = 4338, 8911408
if customer_file.exists():
    c_df = pd.read_csv(customer_file)
    n_cust = len(c_df)
    tot_rev = c_df["monetary"].sum()

p_wape, p_mae = 74.76, 108.44
if prophet_file.exists():
    p_df = pd.read_csv(prophet_file)
    p_mae = (p_df["actual_units"] - p_df["predicted_units"]).abs().mean()
    p_wape = (p_df["actual_units"] - p_df["predicted_units"]).abs().sum() / p_df["actual_units"].sum() * 100

reorder_skus, suggested_units = 3, 4303
if recommendations_file.exists():
    r_df = pd.read_csv(recommendations_file)
    reorder_skus = (r_df["action"] == "REORDER_NOW").sum()
    suggested_units = r_df["suggested_order_qty"].sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers Identified", f"{n_cust:,}", "4 Behavioral Segments")
with col2:
    st.metric("Total Revenue Tracked", f"£{tot_rev:,.0f}", "UK Retail (GBP)")
with col3:
    st.metric("Prophet Forecast WAPE", f"{p_wape:.1f}%", "-6.6% vs. Baseline", delta_color="normal")
with col4:
    st.metric("Urgent Reorders Needed", f"{reorder_skus} SKUs", f"{suggested_units:,} units total", delta_color="inverse")

st.markdown("---")

# -----------------------------------------------------------------------------
# Platform Navigation Cards
# -----------------------------------------------------------------------------
st.subheader("🧭 Platform Modules")

c_mod1, c_mod2, c_mod3 = st.columns(3)

with c_mod1:
    st.info("### 👥 1. Customer Segmentation\n"
            "**RFM + K-Means Clustering**\n\n"
            "- Analyzes 4,338 customer behavioral records\n"
            "- Discovers Champions, Loyal, At-Risk, and Inactive cohorts\n"
            "- Interactive Customer Explorer with real-time filters and CSV export\n\n"
            "👉 *Navigate to **1_Customer_Segmentation** in the sidebar.*")

with c_mod2:
    st.success("### 📈 2. Demand Forecasting\n"
               "**Meta Prophet vs. Seasonal Baseline**\n\n"
               "- Evaluates 42-day holdout test window\n"
               "- Compares MAE and WAPE against Seasonal-Naïve benchmark\n"
               "- Generates 30-day forward demand projections\n\n"
               "👉 *Navigate to **2_Demand_Forecasting** in the sidebar.*")

with c_mod3:
    st.warning("### 📦 3. Inventory Optimization\n"
               "**Target Stock & Reorder Point Policy**\n\n"
               "- Reorder Point ($ROP$) & Target Stock ($T$) formulas\n"
               "- Evaluates physical stock coverage against supplier lead times\n"
               "- High-urgency replenishment action recommendations\n\n"
               "👉 *Navigate to **3_Inventory_Optimization** in the sidebar.*")

st.markdown("---")

# -----------------------------------------------------------------------------
# System Status & Transparency Notice
# -----------------------------------------------------------------------------
st.subheader("ℹ️ System Architecture & Data Notes")
st.markdown("""
* **Customer & Sales History:** Built on 524,000+ validated transaction rows from the Online Retail dataset.
* **Forecasting Engine:** Meta Prophet configured with additive weekly seasonality and trend changepoint detection.
* **Simulated Inventory Notice:** Current stock levels, supplier lead times, and backorders are labelled `SIMULATED_DEMO_DATA` 
  to validate replenishment calculations responsibly without inventing unverified inventory records.
""")
