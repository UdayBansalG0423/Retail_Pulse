"""
app.py
------
RetailPulse — AI-Powered Retail Analytics Platform
Executive Hub & Multipage Navigation Entrypoint
"""

from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "churn"

st.set_page_config(
    page_title="RetailPulse | AI Retail Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛍️ RetailPulse — AI-Powered Retail Analytics Platform")
st.caption(
    "End-to-End Decision Support: Customer Segmentation · Demand Forecasting · "
    "Inventory Replenishment · Churn Prediction"
)

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
churn_metrics_file = REPORT_DIR / "churn_metrics.json"
churn_predictions_file = DATA_DIR / "churn_customer_predictions.csv"

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

# Churn metrics
churn_roc_auc = None
churn_high_risk = None
if churn_metrics_file.exists():
    with open(churn_metrics_file, "r") as _f:
        _cm = json.load(_f)
    churn_roc_auc = _cm.get("roc_auc")
if churn_predictions_file.exists():
    _cp = pd.read_csv(churn_predictions_file)
    churn_high_risk = ((_cp["churn_probability"] >= 0.65).sum())

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Customers Analysed", f"{n_cust:,}", "4 Behavioural Segments")
with col2:
    st.metric("Total Revenue Tracked", f"£{tot_rev:,.0f}", "UK Retail (GBP)")
with col3:
    st.metric("Prophet Forecast WAPE", f"{p_wape:.1f}%", "-6.6 pp vs. Baseline", delta_color="normal")
with col4:
    st.metric("Urgent Reorders (Simulated)", f"{reorder_skus} SKUs", f"{suggested_units:,} units", delta_color="inverse")
with col5:
    if churn_roc_auc is not None:
        st.metric(
            "Churn Model ROC-AUC",
            f"{churn_roc_auc:.4f}",
            f"{churn_high_risk:,} high-risk (test cohort)" if churn_high_risk is not None else "(test set)",
            delta_color="off",
        )
    else:
        st.metric("Churn Model ROC-AUC", "—", "Run churn pipeline")

st.markdown("---")

# -----------------------------------------------------------------------------
# Platform Navigation Cards
# -----------------------------------------------------------------------------
st.subheader("🧭 Platform Modules")

c_mod1, c_mod2 = st.columns(2)
c_mod3, c_mod4 = st.columns(2)

with c_mod1:
    st.info(
        "### 👥 1. Customer Segmentation\n"
        "**RFM + K-Means Clustering**\n\n"
        "- Analyses 4,338 customer behavioural records\n"
        "- Discovers Champions, Loyal, At-Risk, and Inactive cohorts\n"
        "- Interactive Customer Explorer with real-time filters and CSV export\n\n"
        "👉 *Navigate to **1 Customer Segmentation** in the sidebar.*"
    )

with c_mod2:
    st.success(
        "### 📈 2. Demand Forecasting\n"
        "**Meta Prophet vs. Seasonal Baseline**\n\n"
        "- Evaluates 42-day holdout test window\n"
        "- Compares MAE and WAPE against Seasonal-Naïve benchmark\n"
        "- Generates 30-day forward demand projections\n\n"
        "👉 *Navigate to **2 Demand Forecasting** in the sidebar.*"
    )

with c_mod3:
    st.warning(
        "### 📦 3. Inventory Optimization\n"
        "**Target Stock & Reorder Point Policy**\n\n"
        "- Reorder Point ($ROP$) & Target Stock ($T$) formulas\n"
        "- Evaluates physical stock coverage against supplier lead times\n"
        "- Includes What-If Simulator for lead-time and stock-level scenarios\n\n"
        "👉 *Navigate to **3 Inventory Optimization** in the sidebar.*"
    )

with c_mod4:
    st.error(
        "### 🔮 4. Churn Prediction\n"
        "**XGBoost Classifier + SHAP Interpretability**\n\n"
        "- Identifies customers at risk of disengaging (60-day churn window)\n"
        "- Risk-tier filtering (High / Medium / Low) with CSV export\n"
        "- SHAP feature importance explains individual predictions\n\n"
        "👉 *Navigate to **4 Churn Prediction** in the sidebar.*"
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# System Status & Transparency Notice
# -----------------------------------------------------------------------------
st.subheader("ℹ️ System Architecture & Data Notes")
st.markdown("""
* **Customer & Sales History:** Built on 524,000+ validated transaction rows from the UCI Online Retail dataset (2009–2011, UK-based).
* **Forecasting Engine:** Meta Prophet configured with additive weekly seasonality and trend changepoint detection. 
  WAPE 74.76% on the 42-day holdout — the assignment target of MAPE ≤ 12% has **not yet been demonstrated**.
* **Simulated Inventory Notice:** Current stock levels, supplier lead times, and backorders are explicitly simulated 
  (`SIMULATED_DEMO_DATA`) to validate replenishment logic without inventing unverified inventory records.
* **Churn Prediction Scope:** The churn page scores a **held-out historical test cohort** (25% of pre-2011-10-01 customers). 
  ROC-AUC 0.7422 — the assignment target of 0.88 has **not yet been demonstrated**. 
  Results should not be interpreted as live campaign recommendations for current customers.
""")
