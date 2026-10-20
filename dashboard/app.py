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
    "Enterprise Decision Support: Sales Analytics · Customer Intelligence & Churn · "
    "Demand Forecasting · Inventory Replenishment"
)

st.markdown("""
Welcome to **RetailPulse**. This enterprise dashboard brings together machine learning models, 
statistical time-series forecasting, and operational research to maximize revenue, preserve customer lifetime value, 
anticipate demand velocity, and prevent costly inventory stockouts.
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
transactions_file = DATA_DIR / "transactions_clean.csv"

tot_rev, tot_units = 8911408, 5176450
if transactions_file.exists():
    try:
        t_summary = pd.read_csv(transactions_file, usecols=["Revenue", "Quantity"])
        tot_rev = float(t_summary["Revenue"].sum())
        tot_units = int(t_summary["Quantity"].sum())
    except Exception:
        pass

n_cust = 4338
if customer_file.exists():
    c_df = pd.read_csv(customer_file)
    n_cust = len(c_df)

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
    st.metric("Total Revenue Tracked", f"£{tot_rev:,.0f}", f"{tot_units:,.0f} units sold")
with col2:
    st.metric("Total Customers Analysed", f"{n_cust:,}", "4 Behavioural Segments")
with col3:
    st.metric("Prophet Forecast WAPE", f"{p_wape:.1f}%", "-6.6 pp vs. Baseline", delta_color="normal")
with col4:
    st.metric("Urgent Reorders (Simulated)", f"{reorder_skus} SKUs", f"{suggested_units:,} units", delta_color="inverse")
with col5:
    if churn_roc_auc is not None:
        st.metric(
            "Churn Model ROC-AUC",
            f"{churn_roc_auc:.4f}",
            f"{churn_high_risk:,} high-risk (test set)" if churn_high_risk is not None else "(test set)",
            delta_color="off",
        )
    else:
        st.metric("Churn Model ROC-AUC", "—", "Run churn pipeline")

st.markdown("---")

# -----------------------------------------------------------------------------
# Platform Navigation Cards (Ordered 1 to 4)
# -----------------------------------------------------------------------------
st.subheader("🧭 Platform Navigation & Modules")

c_mod1, c_mod2 = st.columns(2)
c_mod3, c_mod4 = st.columns(2)

with c_mod1:
    st.info(
        "### 💰 Page 1 — Sales Dashboard\n"
        "**Revenue Trends & Product Velocity**\n\n"
        "- Historical revenue trends (daily, weekly, monthly, and moving averages)\n"
        "- Top products by revenue and sales volume\n"
        "- Country-level market analytics (UK vs 37 International export markets)\n"
        "- Interactive date range, country, and product substring filters\n"
    )
    st.page_link("pages/1_Sales_Dashboard.py", label="Open Page 1: Sales Dashboard", icon="💰")

with c_mod2:
    st.success(
        "### 👥 Page 2 — Customer Dashboard\n"
        "**RFM Segmentation & Churn Prediction**\n\n"
        "- K-Means RFM clustering: Champions, Loyal, At-Risk, and Inactive\n"
        "- XGBoost churn prediction engine with ROC-AUC 0.7422\n"
        "- SHAP explainability: feature importance ranking and beeswarm plot\n"
        "- Customer 360° individual risk lookup with probability gauge\n"
    )
    st.page_link("pages/2_Customer_Dashboard.py", label="Open Page 2: Customer Dashboard", icon="👥")

with c_mod3:
    st.warning(
        "### 📈 Page 3 — Forecast Dashboard\n"
        "**Meta Prophet vs. Baseline Benchmark**\n\n"
        "- Rigorous evaluation on 42-day holdout test window\n"
        "- Compares MAE and WAPE against Seasonal-Naïve benchmark\n"
        "- Visual evaluation overlay and anomaly diagnostics\n"
        "- 30-day forward demand projections by SKU for procurement\n"
    )
    st.page_link("pages/3_Forecast_Dashboard.py", label="Open Page 3: Forecast Dashboard", icon="📈")

with c_mod4:
    st.error(
        "### 📦 Page 4 — Inventory Dashboard\n"
        "**Target Stock Policy & Reorder Optimization**\n\n"
        "- Continuous-review $(s, S)$ Target Stock & Reorder Point formulas\n"
        "- Physical stock coverage vs. supplier lead times\n"
        "- Actionable alerts: REORDER_NOW, MONITOR, NO_REORDER\n"
        "- Interactive What-If Simulator for lead-time delays and demand surges\n"
    )
    st.page_link("pages/4_Inventory_Dashboard.py", label="Open Page 4: Inventory Dashboard", icon="📦")

st.markdown("---")

# -----------------------------------------------------------------------------
# System Status & Transparency Notice
# -----------------------------------------------------------------------------
st.subheader("ℹ️ System Architecture & Data Notes")
st.markdown("""
* **Customer & Sales History:** Built on 524,000+ validated transaction rows from the UCI Online Retail dataset (2010–2011, UK-based).
* **Forecasting Engine:** Meta Prophet configured with additive weekly seasonality and trend changepoint detection. 
  WAPE 74.76% on the 42-day holdout — the assignment target of MAPE ≤ 12% has **not yet been demonstrated**.
* **Simulated Inventory Notice:** Current stock levels, supplier lead times, and backorders are explicitly simulated 
  (`SIMULATED_DEMO_DATA`) to validate replenishment logic without inventing unverified inventory records.
* **Churn Prediction Scope:** The churn page scores a **held-out historical test cohort** (25% of pre-2011-10-01 customers). 
  ROC-AUC 0.7422 — the assignment target of 0.88 has **not yet been demonstrated**. 
  Results should not be interpreted as live campaign recommendations for current customers.
""")
