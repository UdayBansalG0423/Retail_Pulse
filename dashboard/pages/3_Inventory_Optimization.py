"""
3_Inventory_Optimization.py
---------------------------
Streamlit decision-support page for RetailPulse inventory optimization.
Provides:
  - Reorder alerts and urgency metrics
  - Inventory position vs. Reorder Point (ROP) and Target Stock comparisons
  - Daily demand forecast curve drill-down
  - What-if simulation parameters
  - CSV export for procurement planning
"""

from pathlib import Path
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Apply dark theme styling to matplotlib to match dashboard aesthetics
plt.style.use("dark_background")

# Path setup: dashboard/pages -> dashboard -> RetailPulse root
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RECOMMENDATIONS_FILE = DATA_DIR / "inventory_recommendations.csv"
FORECAST_FILE = DATA_DIR / "forecast_30_day.csv"
INVENTORY_FILE = DATA_DIR / "sample_inventory.csv"

st.set_page_config(
    page_title="RetailPulse | Inventory Optimization",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Inventory Optimization & Replenishment")
st.caption("AI-powered demand-driven replenishment recommendations for retail procurement.")

# Demo data banner
st.warning(
    "⚠️ **Notice: Simulated Demo Inventory** — Transaction history and demand forecasts are derived "
    "from real retail sales, while stock on hand, supplier lead times, and backorders are "
    "explicitly simulated parameters (`SIMULATED_DEMO_DATA`) for algorithmic evaluation.",
    icon="ℹ️",
)


@st.cache_data(show_spinner=False)
def load_inventory_data():
    if not RECOMMENDATIONS_FILE.exists():
        return None, None
    rec_df = pd.read_csv(RECOMMENDATIONS_FILE)
    forecast_df = pd.read_csv(FORECAST_FILE, parse_dates=["date"]) if FORECAST_FILE.exists() else None
    return rec_df, forecast_df


rec_df, forecast_df = load_inventory_data()

if rec_df is None or rec_df.empty:
    st.error(
        f"Inventory recommendations not found at `{RECOMMENDATIONS_FILE}`. "
        "Please run `src/inventory/inventory_optimizer.py` first."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Inventory Filters")

# Urgency Filter
urgency_options = ["All"] + sorted(rec_df["urgency"].unique().tolist())
selected_urgency = st.sidebar.selectbox("Filter by Urgency", urgency_options)

# Action Filter
action_options = ["All"] + sorted(rec_df["action"].unique().tolist())
selected_action = st.sidebar.selectbox("Filter by Action", action_options)

# Product Multiselect
product_options = rec_df["product"].tolist()
selected_products = st.sidebar.multiselect(
    "Select Products",
    options=product_options,
    default=product_options,
)

# Apply filters
filtered_df = rec_df.copy()
if selected_urgency != "All":
    filtered_df = filtered_df[filtered_df["urgency"] == selected_urgency]
if selected_action != "All":
    filtered_df = filtered_df[filtered_df["action"] == selected_action]
if selected_products:
    filtered_df = filtered_df[filtered_df["product"].isin(selected_products)]

# -----------------------------------------------------------------------------
# Top Metric KPI Cards
# -----------------------------------------------------------------------------
total_products = len(rec_df)
reorder_now_count = (rec_df["action"] == "REORDER_NOW").sum()
total_suggested_units = rec_df["suggested_order_qty"].sum()
high_urgency_count = (rec_df["urgency"] == "HIGH").sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Products Monitored", f"{total_products}")
with col2:
    st.metric("Need Reorder Now", f"{reorder_now_count}", delta=f"{reorder_now_count} Urgent", delta_color="inverse")
with col3:
    st.metric("Total Suggested Order", f"{total_suggested_units:,} units")
with col4:
    st.metric("High Urgency SKUs", f"{high_urgency_count}", delta=f"{high_urgency_count} Actionable", delta_color="inverse")

st.markdown("---")

# -----------------------------------------------------------------------------
# Tabbed Interface
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Recommendations & Orders",
    "📊 Stock vs ROP Comparison",
    "📈 30-Day Demand Drilldown",
    "📐 Methodology & Formulas",
])

# -----------------------------------------------------------------------------
# Tab 1: Recommendations Table & Details
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Replenishment Action Table")
    
    display_cols = [
        "product", "action", "urgency", "current_stock", "on_order", 
        "backorders", "inventory_position", "reorder_point", 
        "target_stock", "suggested_order_qty", "days_of_stock"
    ]
    
    styled_df = filtered_df[display_cols].copy()
    styled_df.columns = [
        "Product", "Action", "Urgency", "Current Stock", "On Order", 
        "Backorders", "Inv Position", "ROP", "Target Stock", "Suggested Order", "Days Cover"
    ]
    
    st.dataframe(
        styled_df.style.format({
            "Current Stock": "{:,}",
            "On Order": "{:,}",
            "Backorders": "{:,}",
            "Inv Position": "{:,}",
            "ROP": "{:.1f}",
            "Target Stock": "{:.1f}",
            "Suggested Order": "{:,}",
            "Days Cover": "{:.1f} days",
        }),
        use_container_width=True,
    )

    # Detailed product explanation cards
    st.markdown("#### Decision Rationale & SKU Details")
    for _, row in filtered_df.iterrows():
        status_color = "🔴" if row["action"] == "REORDER_NOW" else ("🟡" if row["action"] == "MONITOR" else "🟢")
        with st.expander(f"{status_color} {row['product']} — Action: {row['action']} ({row['suggested_order_qty']} units)"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Inventory Position Breakdown**")
                st.write(f"- Physical Stock: `{row['current_stock']:,}` units")
                st.write(f"- On Order (In Transit): `{row['on_order']:,}` units")
                st.write(f"- Backorders: `{row['backorders']:,}` units")
                st.write(f"- **Net Position ($IP$):** `{row['inventory_position']:,}` units")
            with c2:
                st.markdown("**Replenishment Parameters**")
                st.write(f"- Supplier Lead Time: `{row['lead_time_days']}` days")
                st.write(f"- Review Period: `{row['review_period_days']}` days")
                st.write(f"- Safety Stock ($SS$): `{row['safety_stock']:,}` units")
                st.write(f"- Avg Daily Demand ($d$): `{row['avg_daily_demand']:.2f}` units/day")
            with c3:
                st.markdown("**Target & Suggestion**")
                st.write(f"- Reorder Point ($ROP$): `{row['reorder_point']:.1f}` units")
                st.write(f"- Target Stock ($T$): `{row['target_stock']:.1f}` units")
                st.write(f"- **Suggested Order ($Q$):** `{row['suggested_order_qty']:,}` units")
                st.write(f"- Stock Runout: `{row['days_of_stock']:.1f}` days")
            st.info(f"💡 **Recommendation Rationale:** {row['rationale']}")

    # CSV Download
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Export Inventory Recommendations (CSV)",
        data=csv_bytes,
        file_name="inventory_recommendations.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Tab 2: Visual Comparison (Stock vs ROP vs Target)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Stock Position vs. Safety & Reorder Thresholds")
    
    if not filtered_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Chart 1: Inventory Position vs ROP vs Target Stock
        products_short = [p[:22] + "..." if len(p) > 25 else p for p in filtered_df["product"]]
        y_pos = range(len(products_short))
        bar_width = 0.25

        ax1.barh([y - bar_width for y in y_pos], filtered_df["inventory_position"], height=bar_width, label="Inv Position (IP)", color="#3498DB")
        ax1.barh(y_pos, filtered_df["reorder_point"], height=bar_width, label="Reorder Point (ROP)", color="#E74C3C")
        ax1.barh([y + bar_width for y in y_pos], filtered_df["target_stock"], height=bar_width, label="Target Stock (T)", color="#2ECC71")

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(products_short, fontsize=9)
        ax1.set_xlabel("Units", fontsize=10)
        ax1.set_title("Inventory Position vs ROP & Target Stock", fontsize=11, fontweight="bold")
        ax1.grid(axis="x", linestyle="--", alpha=0.3)
        ax1.legend(loc="lower right", fontsize=9)

        # Chart 2: Days of Stock Cover vs Supplier Lead Time
        ax2.barh([y - 0.15 for y in y_pos], filtered_df["days_of_stock"], height=0.3, label="Days of Stock on Hand", color="#F39C12")
        ax2.barh([y + 0.15 for y in y_pos], filtered_df["lead_time_days"], height=0.3, label="Supplier Lead Time (Days)", color="#9B59B6")

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(products_short, fontsize=9)
        ax2.set_xlabel("Days", fontsize=10)
        ax2.set_title("Stock Coverage vs. Supplier Lead Time (Days)", fontsize=11, fontweight="bold")
        ax2.grid(axis="x", linestyle="--", alpha=0.3)
        ax2.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        st.caption("🔴 When Days of Stock on Hand < Supplier Lead Time, stockout risk is severe before new shipments can arrive.")
    else:
        st.info("No products match the selected filters.")

# -----------------------------------------------------------------------------
# Tab 3: Demand Forecast Drilldown
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("30-Day Forward Demand Curve (Meta Prophet)")
    
    if forecast_df is not None and not forecast_df.empty:
        selected_drill_prod = st.selectbox("Select product to inspect forecast curve:", rec_df["product"].tolist())
        
        prod_forecast = forecast_df[forecast_df["product"] == selected_drill_prod].sort_values("date")
        
        if not prod_forecast.empty:
            c_left, c_right = st.columns([3, 1])
            with c_left:
                fig_f, ax_f = plt.subplots(figsize=(11, 4.5))
                ax_f.plot(prod_forecast["date"], prod_forecast["yhat"], color="#1ABC9C", marker="o", markersize=4, lw=2, label="Forecast Demand (yhat)")
                avg_d = prod_forecast["yhat"].mean()
                ax_f.axhline(avg_d, color="#F1C40F", linestyle="--", label=f"Average Demand ({avg_d:.1f} units/day)")
                
                ax_f.set_title(f"30-Day Forward Demand Curve — {selected_drill_prod}", fontsize=11, fontweight="bold")
                ax_f.set_xlabel("Date", fontsize=9)
                ax_f.set_ylabel("Predicted Units", fontsize=9)
                ax_f.grid(True, linestyle="--", alpha=0.3)
                ax_f.legend(loc="upper right", fontsize=9)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_f)
            
            with c_right:
                st.markdown("#### Forecast Summary")
                st.metric("30-Day Total Demand", f"{prod_forecast['yhat'].sum():,.0f} units")
                st.metric("Average Daily Demand", f"{avg_d:.1f} units/day")
                st.metric("Peak Forecast Day", f"{prod_forecast['yhat'].max():,.0f} units")
                st.caption(f"Dates: {prod_forecast['date'].min().date()} to {prod_forecast['date'].max().date()}")
        else:
            st.info("No forecast series available for this product.")
    else:
        st.info("Forecast data not loaded.")

# -----------------------------------------------------------------------------
# Tab 4: Methodology & Formulas
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Replenishment Methodology & Policy Specifications")
    
    st.markdown("""
    The RetailPulse inventory optimizer applies a continuous review **Target Stock Policy** $(s, S)$ parameterized with 
    machine learning demand forecasts:

    #### 1. Average Daily Demand ($d$)
    Forecasted demand is aggregated across the 30-day Prophet forward horizon:
    """)
    st.latex(r"d = \frac{1}{30} \sum_{t=1}^{30} \hat{y}_t")

    st.markdown("""
    #### 2. Inventory Position ($IP$)
    Total effective inventory available or scheduled:
    """)
    st.latex(r"IP = \text{Current Stock} + \text{On Order} - \text{Backorders}")

    st.markdown("""
    #### 3. Reorder Point ($ROP$)
    The minimum inventory threshold required to avoid stockouts during the supplier replenishment cycle:
    """)
    st.latex(r"ROP = (d \times L) + SS")
    st.caption("Where $L$ is supplier lead time (days) and $SS$ is safety stock (units).")

    st.markdown("""
    #### 4. Target Stock ($T$) & Order Quantity ($Q$)
    Targets sufficient stock to cover both lead time and the periodic review window ($R$), bringing net position to target:
    """)
    st.latex(r"T = d \times (L + R) + SS")
    st.latex(r"Q = \max\left(0, \; \lceil T - IP \rceil\right)")

    st.markdown("---")
    st.caption("RetailPulse Decision Engine • Module 3 of 4: Inventory Optimization")
