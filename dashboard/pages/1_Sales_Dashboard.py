"""
1_Sales_Dashboard.py
--------------------
RetailPulse — Page 1: Sales Analytics Dashboard

Provides:
  - Revenue trends (Daily, Weekly, Monthly, Cumulative, and Moving Averages)
  - Top products performance and sales volume analytics
  - Country-level market distribution (UK vs International comparisons)
  - Interactive multi-filter controls (Date range, Country, Product search)
  - Filtered transaction data preview and CSV export
"""

from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# Path setup: dashboard/pages -> dashboard -> RetailPulse root
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
TRANSACTIONS_FILE = DATA_DIR / "transactions_clean.csv"

st.set_page_config(
    page_title="RetailPulse | Sales Dashboard",
    page_icon="💰",
    layout="wide",
)

st.title("💰 Sales Analytics Dashboard")
st.caption(
    "Revenue trends, top products, sales volume velocity, and country-level analytics "
    "across 524,000+ verified transactions."
)


# ---------------------------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def load_sales_data():
    if not TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(f"Clean transactions file not found at: {TRANSACTIONS_FILE}")
    
    df = pd.read_csv(
        TRANSACTIONS_FILE,
        dtype={
            "Description": "string",
            "Quantity": "int32",
            "UnitPrice": "float32",
            "CustomerID": "float32",
            "Country": "category",
            "is_return": "bool",
            "Revenue": "float32",
        },
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Date"] = df["InvoiceDate"].dt.date
    return df


try:
    raw_df = load_sales_data()
except Exception as e:
    st.error(f"Error loading sales data: {e}")
    st.info("Run `python setup_pipeline.py` or `python src/data_preprocessing.py` to generate the clean transactions dataset.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Sales Filters")

min_date = raw_df["Date"].min()
max_date = raw_df["Date"].max()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Filter transactions by date window",
)

# Handle single date selection during interactive picker change
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Country filter
all_countries = sorted(raw_df["Country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=all_countries,
    default=[],
    placeholder="All Countries (leave blank for all)",
    help="Select one or more countries, or leave blank to include all 38 markets",
)

# Product search filter
product_query = st.sidebar.text_input(
    "Product Description Search",
    placeholder="e.g. HEART, BUNTING, MUG",
    help="Filter by substring match on product name",
)

# Top N display setting
top_n = st.sidebar.slider(
    "Top N Products to Display",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
)

# Apply filters
mask = (raw_df["Date"] >= start_date) & (raw_df["Date"] <= end_date)
if selected_countries:
    mask &= raw_df["Country"].isin(selected_countries)
if product_query.strip():
    mask &= raw_df["Description"].str.contains(product_query.strip(), case=False, na=False)

df_filtered = raw_df[mask]

if df_filtered.empty:
    st.warning("⚠️ No transactions match the selected filter criteria. Please broaden your selection.")
    st.stop()


# ---------------------------------------------------------------------------
# Executive KPI Cards
# ---------------------------------------------------------------------------
total_rev = float(df_filtered["Revenue"].sum())
total_units = int(df_filtered["Quantity"].sum())
total_tx = len(df_filtered)
avg_unit_price = float(df_filtered["UnitPrice"].mean())
num_countries = df_filtered["Country"].nunique()
num_customers = df_filtered["CustomerID"].dropna().nunique()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric(
        "Total Net Revenue",
        f"£{total_rev:,.0f}",
        help="Sum of all transaction revenues in selected scope",
    )
with kpi2:
    st.metric(
        "Sales Volume",
        f"{total_units:,} units",
        help="Total physical units sold",
    )
with kpi3:
    st.metric(
        "Transactions",
        f"{total_tx:,}",
        help="Total line-item sales records",
    )
with kpi4:
    st.metric(
        "Avg Unit Price",
        f"£{avg_unit_price:.2f}",
        help="Average price per individual unit",
    )
with kpi5:
    st.metric(
        "Active Markets",
        f"{num_countries} countries",
        f"{num_customers:,} customers",
        delta_color="off",
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# Tabbed Analytics Interface
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Revenue Trends & Seasonality",
    "🏆 Top Products & Sales Volume",
    "🌍 Country-Level Analytics",
    "🔍 Transaction Explorer",
])


# ---------------------------------------------------------------------------
# Tab 1: Revenue Trends & Seasonality
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Historical Revenue Trajectory")
    
    granularity = st.radio(
        "Time Aggregation:",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
    )
    
    # Aggregate by time period
    if granularity == "Daily":
        t_df = df_filtered.groupby("Date").agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
        ).reset_index()
        t_df["Date"] = pd.to_datetime(t_df["Date"])
        t_df = t_df.sort_values("Date")
        t_df["MA7"] = t_df["Revenue"].rolling(window=7, min_periods=1).mean()
        
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(t_df["Date"], t_df["Revenue"], color="#3b82f6", alpha=0.55, lw=1.2, label="Daily Revenue")
        ax.plot(t_df["Date"], t_df["MA7"], color="#60a5fa", lw=2.5, label="7-Day Moving Avg")
        ax.fill_between(t_df["Date"], 0, t_df["Revenue"], color="#3b82f6", alpha=0.15)
        ax.set_title("Daily Sales Revenue with 7-Day Moving Average", fontsize=11, fontweight="bold", color="white")
        ax.set_ylabel("Revenue (£)", color="white")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="upper left")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        plt.xticks(rotation=30, color="white")
        plt.yticks(color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    elif granularity == "Weekly":
        df_copy = df_filtered.copy()
        df_copy["Week"] = df_copy["InvoiceDate"].dt.to_period("W").apply(lambda r: r.start_time)
        t_df = df_copy.groupby("Week").agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
        ).reset_index().sort_values("Week")
        
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.bar(t_df["Week"], t_df["Revenue"], width=5, color="#8b5cf6", edgecolor="#6d28d9", alpha=0.85, label="Weekly Revenue")
        ax.plot(t_df["Week"], t_df["Revenue"], color="#c084fc", marker="o", lw=2)
        ax.set_title("Weekly Sales Revenue (£)", fontsize=11, fontweight="bold", color="white")
        ax.set_ylabel("Revenue (£)", color="white")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        plt.xticks(rotation=30, color="white")
        plt.yticks(color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    else:  # Monthly
        df_copy = df_filtered.copy()
        df_copy["Month"] = df_copy["InvoiceDate"].dt.to_period("M").astype(str)
        t_df = df_copy.groupby("Month").agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
        ).reset_index().sort_values("Month")
        
        fig, ax = plt.subplots(figsize=(12, 4.5))
        bars = ax.bar(t_df["Month"], t_df["Revenue"], color="#10b981", edgecolor="#047857", alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (max(t_df["Revenue"]) * 0.015),
                f"£{height/1e3:,.0f}k",
                ha="center",
                va="bottom",
                color="white",
                fontsize=8,
            )
        ax.set_title("Monthly Revenue Trajectory (£)", fontsize=11, fontweight="bold", color="white")
        ax.set_ylabel("Revenue (£)", color="white")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        plt.xticks(rotation=30, color="white")
        plt.yticks(color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Day-of-Week Seasonality Pattern")
    c_dow1, c_dow2 = st.columns(2)
    
    with c_dow1:
        df_dow = df_filtered.copy()
        df_dow["DayOfWeek"] = df_dow["InvoiceDate"].dt.day_name()
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
        dow_stats = df_dow.groupby("DayOfWeek")["Revenue"].sum().reindex(days_order).dropna()
        
        fig_dow, ax_dow = plt.subplots(figsize=(7, 3.8))
        ax_dow.bar(dow_stats.index, dow_stats.values / 1e3, color="#f59e0b", edgecolor="#b45309", alpha=0.85)
        ax_dow.set_title("Revenue by Day of Week (£ in Thousands)", fontsize=10, fontweight="bold", color="white")
        ax_dow.set_ylabel("Revenue (£k)", color="white")
        ax_dow.grid(axis="y", linestyle="--", alpha=0.25)
        ax_dow.set_facecolor("#0e1117")
        fig_dow.patch.set_facecolor("#0e1117")
        plt.xticks(rotation=20, color="white")
        plt.yticks(color="white")
        plt.tight_layout()
        st.pyplot(fig_dow)
        plt.close(fig_dow)
        
    with c_dow2:
        dow_qty = df_dow.groupby("DayOfWeek")["Quantity"].sum().reindex(days_order).dropna()
        fig_qty, ax_qty = plt.subplots(figsize=(7, 3.8))
        ax_qty.bar(dow_qty.index, dow_qty.values / 1e3, color="#ec4899", edgecolor="#be185d", alpha=0.85)
        ax_qty.set_title("Volume (Units Sold in Thousands)", fontsize=10, fontweight="bold", color="white")
        ax_qty.set_ylabel("Units (k)", color="white")
        ax_qty.grid(axis="y", linestyle="--", alpha=0.25)
        ax_qty.set_facecolor("#0e1117")
        fig_qty.patch.set_facecolor("#0e1117")
        plt.xticks(rotation=20, color="white")
        plt.yticks(color="white")
        plt.tight_layout()
        st.pyplot(fig_qty)
        plt.close(fig_qty)


# ---------------------------------------------------------------------------
# Tab 2: Top Products & Sales Volume
# ---------------------------------------------------------------------------
with tab2:
    st.subheader(f"Top {top_n} Products by Revenue & Sales Volume")
    
    prod_summary = df_filtered.groupby("Description").agg(
        Total_Revenue=("Revenue", "sum"),
        Units_Sold=("Quantity", "sum"),
        Avg_Unit_Price=("UnitPrice", "mean"),
        Transactions=("Revenue", "count"),
    ).reset_index()
    
    prod_summary["Revenue_Share_Pct"] = (prod_summary["Total_Revenue"] / total_rev * 100).round(2)
    top_rev_prods = prod_summary.sort_values("Total_Revenue", ascending=False).head(top_n)
    top_vol_prods = prod_summary.sort_values("Units_Sold", ascending=False).head(top_n)
    
    cp1, cp2 = st.columns(2)
    
    with cp1:
        st.markdown(f"#### Top {top_n} Revenue Drivers")
        names_rev = [n[:24] + "..." if len(n) > 26 else n for n in top_rev_prods["Description"]]
        fig_p1, ax_p1 = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
        bars = ax_p1.barh(names_rev[::-1], top_rev_prods["Total_Revenue"][::-1], color="#38bdf8", edgecolor="#0284c7")
        for bar in bars:
            w = bar.get_width()
            ax_p1.text(w + (max(top_rev_prods["Total_Revenue"]) * 0.01), bar.get_y() + bar.get_height() / 2,
                       f"£{w/1e3:,.1f}k", va="center", ha="left", color="white", fontsize=8)
        ax_p1.set_xlabel("Revenue (£)", color="white")
        ax_p1.set_title("Products by Total Revenue (£)", color="white", fontsize=10, fontweight="bold")
        ax_p1.grid(axis="x", linestyle="--", alpha=0.25)
        ax_p1.set_facecolor("#0e1117")
        fig_p1.patch.set_facecolor("#0e1117")
        plt.yticks(color="white", fontsize=9)
        plt.xticks(color="white")
        plt.tight_layout()
        st.pyplot(fig_p1)
        plt.close(fig_p1)

    with cp2:
        st.markdown(f"#### Top {top_n} Volume Velocity")
        names_vol = [n[:24] + "..." if len(n) > 26 else n for n in top_vol_prods["Description"]]
        fig_p2, ax_p2 = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
        bars2 = ax_p2.barh(names_vol[::-1], top_vol_prods["Units_Sold"][::-1], color="#34d399", edgecolor="#059669")
        for bar in bars2:
            w = bar.get_width()
            ax_p2.text(w + (max(top_vol_prods["Units_Sold"]) * 0.01), bar.get_y() + bar.get_height() / 2,
                       f"{w:,.0f} u", va="center", ha="left", color="white", fontsize=8)
        ax_p2.set_xlabel("Units Sold", color="white")
        ax_p2.set_title("Products by Physical Units Sold", color="white", fontsize=10, fontweight="bold")
        ax_p2.grid(axis="x", linestyle="--", alpha=0.25)
        ax_p2.set_facecolor("#0e1117")
        fig_p2.patch.set_facecolor("#0e1117")
        plt.yticks(color="white", fontsize=9)
        plt.xticks(color="white")
        plt.tight_layout()
        st.pyplot(fig_p2)
        plt.close(fig_p2)

    st.markdown("---")
    st.subheader("Searchable Product Catalog & Margin Analysis")
    
    prod_table = prod_summary.sort_values("Total_Revenue", ascending=False).head(100).copy()
    st.dataframe(
        prod_table.style.format({
            "Total_Revenue": "£{:,.2f}",
            "Units_Sold": "{:,}",
            "Avg_Unit_Price": "£{:.2f}",
            "Transactions": "{:,}",
            "Revenue_Share_Pct": "{:.2f}%",
        }),
        use_container_width=True,
        height=350,
    )


# ---------------------------------------------------------------------------
# Tab 3: Country-Level Analytics
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Geographic Revenue & Market Distribution")
    
    country_summary = df_filtered.groupby("Country").agg(
        Revenue=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        Transactions=("Revenue", "count"),
        Customers=("CustomerID", "nunique"),
    ).reset_index()
    country_summary["Share_Pct"] = (country_summary["Revenue"] / total_rev * 100).round(2)
    country_summary = country_summary.sort_values("Revenue", ascending=False).reset_index(drop=True)
    
    # Split UK vs International
    uk_rev = country_summary.loc[country_summary["Country"] == "United Kingdom", "Revenue"].sum()
    intl_rev = total_rev - uk_rev
    intl_share = (intl_rev / total_rev * 100) if total_rev > 0 else 0
    
    cg1, cg2, cg3 = st.columns(3)
    cg1.metric("United Kingdom Revenue", f"£{uk_rev:,.0f}", f"{100-intl_share:.1f}% Market Share", delta_color="off")
    cg2.metric("International Revenue", f"£{intl_rev:,.0f}", f"{intl_share:.1f}% International", delta_color="normal")
    cg3.metric("Total Export Markets", f"{country_summary[country_summary['Country'] != 'United Kingdom']['Country'].nunique()} Countries")
    
    c_geo1, c_geo2 = st.columns(2)
    
    with c_geo1:
        st.markdown("#### Top 10 International Markets (Excl. UK)")
        intl_top = country_summary[country_summary["Country"] != "United Kingdom"].head(10)
        
        fig_c, ax_c = plt.subplots(figsize=(7, 4.2))
        bars_c = ax_c.barh(intl_top["Country"][::-1].astype(str), intl_top["Revenue"][::-1], color="#f43f5e", edgecolor="#be123c")
        for bar in bars_c:
            w = bar.get_width()
            ax_c.text(w + (max(intl_top["Revenue"]) * 0.015), bar.get_y() + bar.get_height() / 2,
                      f"£{w/1e3:,.1f}k", va="center", ha="left", color="white", fontsize=8)
        ax_c.set_xlabel("Revenue (£)", color="white")
        ax_c.set_title("Top 10 Export Markets by Revenue", color="white", fontsize=10, fontweight="bold")
        ax_c.grid(axis="x", linestyle="--", alpha=0.25)
        ax_c.set_facecolor("#0e1117")
        fig_c.patch.set_facecolor("#0e1117")
        plt.yticks(color="white", fontsize=9)
        plt.xticks(color="white")
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close(fig_c)

    with c_geo2:
        st.markdown("#### Market Concentration (UK vs Top 5 International)")
        top6 = country_summary.head(6).copy()
        other_rev = total_rev - top6["Revenue"].sum()
        if other_rev > 0:
            pie_df = pd.concat([
                top6[["Country", "Revenue"]],
                pd.DataFrame([{"Country": "All Other Markets", "Revenue": other_rev}])
            ])
        else:
            pie_df = top6[["Country", "Revenue"]]
            
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4.2))
        colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#64748b"]
        wedges, texts, autotexts = ax_pie.pie(
            pie_df["Revenue"],
            labels=pie_df["Country"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors[:len(pie_df)],
            textprops=dict(color="white", fontsize=8),
        )
        for at in autotexts:
            at.set_fontsize(7.5)
            at.set_weight("bold")
        ax_pie.set_title("Market Revenue Distribution", color="white", fontsize=10, fontweight="bold")
        fig_pie.patch.set_facecolor("#0e1117")
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    st.markdown("---")
    st.markdown("#### Complete Market Breakdown Table")
    st.dataframe(
        country_summary.style.format({
            "Revenue": "£{:,.2f}",
            "Units": "{:,}",
            "Transactions": "{:,}",
            "Customers": "{:,}",
            "Share_Pct": "{:.2f}%",
        }),
        use_container_width=True,
        height=320,
    )


# ---------------------------------------------------------------------------
# Tab 4: Transaction Explorer & CSV Export
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Filtered Transaction Data Explorer")
    preview_limit = 250
    st.caption(
        f"Displaying first **{min(preview_limit, len(df_filtered)):,}** of **{len(df_filtered):,}** "
        "matching transactions for the selected filters."
    )
    
    display_sample = df_filtered.head(preview_limit).copy()
    display_sample["CustomerID"] = display_sample["CustomerID"].fillna(0).astype(int).replace(0, np.nan)
    display_sample["InvoiceDate"] = display_sample["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")
    
    st.dataframe(
        display_sample[[
            "InvoiceDate", "Description", "Quantity", "UnitPrice", "Revenue", "Country", "CustomerID"
        ]].style.format({
            "UnitPrice": "£{:.2f}",
            "Revenue": "£{:.2f}",
            "Quantity": "{:,}",
        }),
        use_container_width=True,
        height=400,
    )
    
    # CSV Download Button
    csv_sample = df_filtered.head(5000)[[
        "InvoiceDate", "Description", "Quantity", "UnitPrice", "Revenue", "Country", "CustomerID"
    ]].to_csv(index=False).encode("utf-8")
    
    st.download_button(
        label="💾 Download Filtered Sales Records (CSV)",
        data=csv_sample,
        file_name="sales_transactions_filtered.csv",
        mime="text/csv",
        help="Export top filtered transactions as CSV",
    )
