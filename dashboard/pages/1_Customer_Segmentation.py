"""
1_Customer_Segmentation.py
--------------------------
Streamlit dashboard page for RFM customer segmentation.
Provides:
  - Total customer, revenue, and recency KPI cards
  - Segment analytics table and profile charts
  - Dynamic multi-filter Customer Explorer
  - Filtered customer CSV export
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("dark_background")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "segmentation"

CUSTOMER_FILE = DATA_DIR / "customer_rfm.csv"
SUMMARY_FILE = DATA_DIR / "segment_summary.csv"

st.set_page_config(
    page_title="RetailPulse | Customer Segmentation",
    page_icon="👥",
    layout="wide",
)

st.title("👥 Customer Segmentation (RFM + K-Means)")
st.caption("AI-powered customer behavioral clustering on UK retail transaction data.")


import math

@st.cache_data(ttl=60, show_spinner=False)
def load_data():
    if not CUSTOMER_FILE.exists() or not SUMMARY_FILE.exists():
        raise FileNotFoundError("Processed segmentation CSVs not found.")
    customers = pd.read_csv(CUSTOMER_FILE)
    segments = pd.read_csv(SUMMARY_FILE)
    return customers, segments


try:
    customers_df, segment_summary_df = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Customer Filters")

segment_options = segment_summary_df["segment"].tolist()
selected_segments = st.sidebar.multiselect(
    "Segment",
    options=segment_options,
    default=segment_options,
)

recency_min = int(math.floor(customers_df["recency"].min()))
recency_max = int(math.ceil(customers_df["recency"].max()))
recency_range = st.sidebar.slider(
    "Recency (days)",
    min_value=recency_min,
    max_value=recency_max,
    value=(recency_min, recency_max),
    step=1,
)

freq_min = int(math.floor(customers_df["frequency"].min()))
freq_max = int(math.ceil(customers_df["frequency"].max()))
frequency_range = st.sidebar.slider(
    "Frequency (purchase dates)",
    min_value=freq_min,
    max_value=freq_max,
    value=(freq_min, freq_max),
    step=1,
)

monetary_min = int(math.floor(customers_df["monetary"].min()))
monetary_max = int(math.ceil(customers_df["monetary"].max()))
monetary_range = st.sidebar.slider(
    "Monetary (Spend £)",
    min_value=monetary_min,
    max_value=monetary_max,
    value=(monetary_min, monetary_max),
    step=10,
)

search_id = st.sidebar.text_input(
    "Search Customer ID",
    placeholder="e.g. 12346",
)

# ---------------------------------------------------------------------------
# Overview KPI Cards
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Identified Customers", value=f"{customers_df.shape[0]:,}", help="Full dataset scope: 4,338 unique non-null Customer IDs")
with col2:
    total_rev = customers_df["monetary"].sum()
    st.metric(label="Total Database Revenue", value=f"£{total_rev:,.0f}", help="Total spending across all 4,338 identified customers")
with col3:
    avg_recency = customers_df["recency"].mean()
    st.metric(label="Avg Customer Recency", value=f"{avg_recency:,.1f} days", help="Average days since last recorded purchase")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📊 Segment Analytics & Distributions",
    "🔎 Customer Explorer Table",
    "📖 Cluster Profiles & Insights",
])

# ---------------------------------------------------------------------------
# Tab 1: Segment Analytics
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Segment Summary Statistics")
    st.dataframe(
        segment_summary_df.style.format({
            "customer_count": "{:,}",
            "median_recency_days": "{:.1f}",
            "median_purchase_dates": "{:.1f}",
            "median_spend_gbp": "£{:,.2f}",
            "average_spend_gbp": "£{:,.2f}",
            "customer_share_pct": "{:.2f}%",
        }),
        use_container_width=True,
    )

    c_img1, c_img2 = st.columns(2)
    with c_img1:
        seg_sizes = REPORT_DIR / "segment_sizes.png"
        if seg_sizes.exists():
            st.image(str(seg_sizes), caption="Segment Size Distribution")
    with c_img2:
        rfm_prof = REPORT_DIR / "rfm_profiles.png"
        if rfm_prof.exists():
            st.image(str(rfm_prof), caption="Median RFM Profile per Segment")

# ---------------------------------------------------------------------------
# Tab 2: Customer Explorer
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Filtered Customer Database")
    filtered = customers_df.copy()
    filtered = filtered[filtered["segment"].isin(selected_segments)]
    filtered = filtered[(filtered["recency"].between(*recency_range))]
    filtered = filtered[(filtered["frequency"].between(*frequency_range))]
    filtered = filtered[(filtered["monetary"].between(*monetary_range))]
    if search_id.strip():
        try:
            search_val = float(search_id)
            filtered = filtered[filtered["CustomerID"] == search_val]
        except ValueError:
            st.warning("CustomerID search must be numeric.")

    max_rows = 200
    st.caption(f"Showing **{min(len(filtered), max_rows):,}** of **{len(filtered):,}** matching customers filtered from the 4,338 database total.")
    display_df = filtered.head(max_rows).copy()
    display_df["CustomerID"] = display_df["CustomerID"].astype(int).astype(str)
    display_df["monetary"] = display_df["monetary"].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(display_df, use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Export Filtered Customers as CSV",
        data=csv_bytes,
        file_name="filtered_customers.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Tab 3: Insights & Recency vs Spend
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Customer Value Distribution (Recency vs. Spend)")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sc = ax.scatter(customers_df["recency"], customers_df["monetary"], alpha=0.5, c="#3498DB", edgecolors="none")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary Spend (£, log scale)")
    ax.set_yscale("log")
    ax.set_title("Recency vs. Spend (Log-Scale)")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)
    st.caption("Lower recency and higher monetary spend represent high-value retention opportunities.")
