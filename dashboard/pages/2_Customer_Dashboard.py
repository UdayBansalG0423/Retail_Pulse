"""
2_Customer_Dashboard.py
-----------------------
RetailPulse — Page 2: Customer Analytics & Churn Prediction

Provides:
  - RFM Segmentation (Champions, Loyal, At-Risk, Inactive cohorts)
  - Interactive Customer Explorer with dynamic multidimensional filters
  - Churn Prediction Engine with XGBoost model metrics (ROC-AUC, Precision@Top-20%)
  - SHAP Interpretability (Feature importance ranking and beeswarm plot)
  - Customer 360° Profile Lookup with churn probability gauge
  - CSV exports for segmentation and churn campaign targeting
"""

from pathlib import Path
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

plt.style.use("dark_background")

# Path setup: dashboard/pages -> dashboard -> RetailPulse root
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
SEGMENT_REPORT_DIR = ROOT / "reports" / "segmentation"
CHURN_REPORT_DIR = ROOT / "reports" / "churn"

CUSTOMER_FILE = DATA_DIR / "customer_rfm.csv"
SUMMARY_FILE = DATA_DIR / "segment_summary.csv"
CHURN_PRED_FILE = DATA_DIR / "churn_customer_predictions.csv"
CHURN_METRICS_FILE = CHURN_REPORT_DIR / "churn_metrics.json"
SHAP_IMPORTANCE_FILE = CHURN_REPORT_DIR / "churn_shap_importance.csv"
SHAP_SUMMARY_PNG = CHURN_REPORT_DIR / "churn_shap_summary.png"
SEG_SIZES_PNG = SEGMENT_REPORT_DIR / "segment_sizes.png"
RFM_PROFILES_PNG = SEGMENT_REPORT_DIR / "rfm_profiles.png"

# Churn risk thresholds
HIGH_RISK_THRESHOLD = 0.65
LOW_RISK_THRESHOLD = 0.35

st.set_page_config(
    page_title="RetailPulse | Customer Dashboard",
    page_icon="👥",
    layout="wide",
)

st.title("👥 Customer Intelligence & Churn Dashboard")
st.caption(
    "Unified customer analytics: Behavioral RFM Segmentation + AI-driven Churn Prediction "
    "with SHAP Explainability."
)


# ---------------------------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_customer_data():
    if not CUSTOMER_FILE.exists() or not SUMMARY_FILE.exists():
        raise FileNotFoundError("Customer segmentation data files not found.")
    customers = pd.read_csv(CUSTOMER_FILE)
    segments = pd.read_csv(SUMMARY_FILE)
    return customers, segments


@st.cache_data(ttl=60, show_spinner=False)
def load_churn_data():
    churn_df = pd.read_csv(CHURN_PRED_FILE) if CHURN_PRED_FILE.exists() else None
    if churn_df is not None:
        churn_df["churn_probability"] = churn_df["churn_probability"].round(4)
    
    metrics = None
    if CHURN_METRICS_FILE.exists():
        with open(CHURN_METRICS_FILE, "r") as f:
            metrics = json.load(f)
            
    shap_df = None
    if SHAP_IMPORTANCE_FILE.exists():
        shap_df = pd.read_csv(SHAP_IMPORTANCE_FILE).dropna()
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        
    return churn_df, metrics, shap_df


try:
    customers_df, segment_summary_df = load_customer_data()
    churn_df, churn_metrics, shap_df = load_churn_data()
except Exception as e:
    st.error(f"Error loading customer analytics data: {e}")
    st.stop()


# Assign risk tier
def assign_risk_tier(p):
    if p >= HIGH_RISK_THRESHOLD:
        return "High"
    elif p <= LOW_RISK_THRESHOLD:
        return "Low"
    else:
        return "Medium"


if churn_df is not None:
    churn_df["risk_tier"] = churn_df["churn_probability"].apply(assign_risk_tier)


# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Customer Filters")

# RFM Segment Filter
segment_options = segment_summary_df["segment"].tolist()
selected_segments = st.sidebar.multiselect(
    "RFM Segment",
    options=segment_options,
    default=segment_options,
    help="Filter customers by behavioral cohort",
)

rec_min = int(math.floor(customers_df["recency"].min()))
rec_max = int(math.ceil(customers_df["recency"].max()))
recency_range = st.sidebar.slider(
    "Recency Window (days)",
    min_value=rec_min,
    max_value=rec_max,
    value=(rec_min, rec_max),
    step=1,
)

freq_min = int(math.floor(customers_df["frequency"].min()))
freq_max = int(math.ceil(customers_df["frequency"].max()))
frequency_range = st.sidebar.slider(
    "Purchase Frequency (dates)",
    min_value=freq_min,
    max_value=freq_max,
    value=(freq_min, freq_max),
    step=1,
)

mon_min = int(math.floor(customers_df["monetary"].min()))
mon_max = int(math.ceil(customers_df["monetary"].max()))
monetary_range = st.sidebar.slider(
    "Monetary Spend (£)",
    min_value=mon_min,
    max_value=mon_max,
    value=(mon_min, mon_max),
    step=25,
)

# Churn Risk Tier Filter
churn_tier_filter = st.sidebar.multiselect(
    "Churn Risk Tier (Test Set)",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"],
    help="Filter churn predictions by risk tier",
)

search_id = st.sidebar.text_input(
    "Quick Customer ID Search",
    placeholder="e.g. 15100 or 12346",
)

# ---------------------------------------------------------------------------
# Executive KPI Cockpit
# ---------------------------------------------------------------------------
total_customers = len(customers_df)
total_revenue = customers_df["monetary"].sum()
avg_recency = customers_df["recency"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}",
        f"{len(segment_summary_df)} Segments",
        help="All unique customers with clean transaction history",
    )
with col2:
    st.metric(
        "Tracked Lifetime Spend",
        f"£{total_revenue:,.0f}",
        "UK Retail Scope",
        help="Cumulative spend across all active customer accounts",
    )
with col3:
    st.metric(
        "Average Recency",
        f"{avg_recency:.1f} days",
        "Across entire base",
        help="Mean days elapsed since last purchase",
    )
with col4:
    if churn_metrics:
        st.metric(
            "Churn Model ROC-AUC",
            f"{churn_metrics['roc_auc']:.4f}",
            f"{churn_metrics['lift_at_top_20_pct']:.2f}x Lift@Top20",
            delta_color="normal",
            help="Holdout test cohort discriminatory ability",
        )
    else:
        st.metric("Churn Model ROC-AUC", "—")
with col5:
    if churn_df is not None:
        n_high = (churn_df["risk_tier"] == "High").sum()
        st.metric(
            "High Churn Risk Cohort",
            f"{n_high:,}",
            f"{(n_high/len(churn_df)*100):.1f}% of Test Set",
            delta_color="inverse",
            help="Customers with predicted churn probability >= 65%",
        )
    else:
        st.metric("High Churn Risk Cohort", "—")

st.markdown("---")


# ---------------------------------------------------------------------------
# Tabbed Modules Interface
# ---------------------------------------------------------------------------
tab_seg, tab_churn, tab_shap, tab_lookup = st.tabs([
    "👥 RFM Customer Segmentation",
    "🔮 Churn Risk Prediction",
    "🧠 SHAP Interpretability & Feature Drivers",
    "🔎 Customer 360° Profile Lookup",
])


# ---------------------------------------------------------------------------
# Tab 1: RFM Customer Segmentation
# ---------------------------------------------------------------------------
with tab_seg:
    st.subheader("Customer Behavioral Segmentation (RFM + K-Means)")
    st.markdown(
        "Customers are clustered into four distinct cohorts based on **Recency** (days since last purchase), "
        "**Frequency** (distinct active purchase dates), and **Monetary Value** (total historical spend in GBP)."
    )
    
    # Segment Summary Table
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
    
    # Pre-generated or interactive cluster charts
    sc1, sc2 = st.columns(2)
    with sc1:
        if SEG_SIZES_PNG.exists():
            st.image(str(SEG_SIZES_PNG), caption="Customer Distribution Across Behavioral Segments", use_column_width=True)
        else:
            fig_sizes, ax_s = plt.subplots(figsize=(6, 3.8))
            ax_s.bar(segment_summary_df["segment"], segment_summary_df["customer_count"], color="#3b82f6")
            ax_s.set_title("Customer Count per Segment", color="white")
            ax_s.set_facecolor("#0e1117")
            fig_sizes.patch.set_facecolor("#0e1117")
            plt.xticks(rotation=20, color="white")
            st.pyplot(fig_sizes)
            plt.close(fig_sizes)
            
    with sc2:
        if RFM_PROFILES_PNG.exists():
            st.image(str(RFM_PROFILES_PNG), caption="Standardized RFM Profiles by Segment", use_column_width=True)
        else:
            st.info("RFM profile visualization available in reports directory.")

    st.markdown("---")
    st.subheader("Dynamic Customer Explorer")
    
    # Apply filters
    filtered_cust = customers_df.copy()
    filtered_cust = filtered_cust[filtered_cust["segment"].isin(selected_segments)]
    filtered_cust = filtered_cust[filtered_cust["recency"].between(*recency_range)]
    filtered_cust = filtered_cust[filtered_cust["frequency"].between(*frequency_range)]
    filtered_cust = filtered_cust[filtered_cust["monetary"].between(*monetary_range)]
    
    if search_id.strip():
        try:
            s_val = float(search_id.strip())
            filtered_cust = filtered_cust[filtered_cust["CustomerID"] == s_val]
        except ValueError:
            st.warning("Customer ID search must be numeric.")
            
    st.caption(
        f"Displaying **{min(200, len(filtered_cust)):,}** of **{len(filtered_cust):,}** "
        f"filtered records (from total {len(customers_df):,} customers in database)."
    )
    
    display_cust = filtered_cust.head(200).copy()
    display_cust["CustomerID"] = display_cust["CustomerID"].astype(int).astype(str)
    
    st.dataframe(
        display_cust.style.format({
            "recency": "{:.0f} days",
            "frequency": "{:.0f} orders",
            "monetary": "£{:,.2f}",
        }),
        use_container_width=True,
        height=320,
    )
    
    csv_cust = filtered_cust.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Filtered Segment Customers (CSV)",
        data=csv_cust,
        file_name="segmented_customers_filtered.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Tab 2: Churn Risk Prediction
# ---------------------------------------------------------------------------
with tab_churn:
    st.subheader("Customer Churn Prediction (XGBoost Classifier)")
    
    st.warning(
        "⚠️ **Historical Evaluation Cohort Notice:** Predictions represent a 25% held-out test sample "
        "of customers active prior to cutoff date `2011-09-30`. Churn is defined as zero purchases in "
        "the subsequent 60-day window (`2011-10-01` → `2011-11-29`). ROC-AUC: **0.7422**.",
        icon="ℹ️",
    )
    
    if churn_metrics and churn_df is not None:
        # Performance Metrics
        m = churn_metrics
        cm1, cm2, cm3, cm4, cm5 = st.columns(5)
        cm1.metric("ROC-AUC", f"{m['roc_auc']:.4f}", help="Area Under the Receiver Operating Characteristic Curve")
        cm2.metric("PR-AUC", f"{m['average_precision_pr_auc']:.4f}", help="Precision-Recall Area Under Curve")
        cm3.metric("Precision@Top-20%", f"{m['precision_at_top_20_pct']:.1%}", help="Precision when targeting highest 20% risk tier")
        cm4.metric("Targeting Lift", f"{m['lift_at_top_20_pct']:.2f}x", help="Lift over random marketing outreach")
        cm5.metric("Brier Loss", f"{m['brier_score_loss']:.4f}", help="Probability calibration error (lower is better)")
        
        st.markdown("---")
        
        # Risk Tiers Overview
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.markdown("#### Risk Cohort Breakdown")
            tier_counts = churn_df["risk_tier"].value_counts()
            n_h = tier_counts.get("High", 0)
            n_m = tier_counts.get("Medium", 0)
            n_l = tier_counts.get("Low", 0)
            
            st.markdown(f"🔴 **High Risk** (≥65%): `{n_h:,}` customers (`{n_h/len(churn_df)*100:.1f}%`)")
            st.markdown(f"🟡 **Medium Risk** (35–65%): `{n_m:,}` customers (`{n_m/len(churn_df)*100:.1f}%`)")
            st.markdown(f"🟢 **Low Risk** (≤35%): `{n_l:,}` customers (`{n_l/len(churn_df)*100:.1f}%`)")
            st.markdown(f"**Total Evaluated:** `{len(churn_df):,}` customers")
            
        with rc2:
            fig_hist, ax_h = plt.subplots(figsize=(8, 3.2))
            ax_h.hist(churn_df["churn_probability"], bins=35, color="#8b5cf6", edgecolor="#6d28d9", alpha=0.85)
            ax_h.axvline(HIGH_RISK_THRESHOLD, color="#ef4444", linestyle="--", lw=1.5, label=f"High ({HIGH_RISK_THRESHOLD:.0%})")
            ax_h.axvline(LOW_RISK_THRESHOLD, color="#22c55e", linestyle="--", lw=1.5, label=f"Low ({LOW_RISK_THRESHOLD:.0%})")
            ax_h.set_title("Predicted Churn Probability Distribution (Test Cohort)", color="white", fontsize=10, fontweight="bold")
            ax_h.set_xlabel("Churn Probability", color="white")
            ax_h.set_ylabel("Customers", color="white")
            ax_h.grid(True, linestyle="--", alpha=0.25)
            ax_h.legend(loc="upper right")
            ax_h.set_facecolor("#0e1117")
            fig_hist.patch.set_facecolor("#0e1117")
            plt.xticks(color="white")
            plt.yticks(color="white")
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

        st.markdown("---")
        st.subheader("Customer Churn Risk Table")
        
        # Filter churn dataframe
        f_churn = churn_df[churn_df["risk_tier"].isin(churn_tier_filter)].copy()
        
        tier_icons = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}
        f_churn["Risk_Tier_Display"] = f_churn["risk_tier"].map(tier_icons)
        f_churn = f_churn.sort_values("churn_probability", ascending=False).reset_index(drop=True)
        
        st.dataframe(
            f_churn[["CustomerID", "Risk_Tier_Display", "churn_probability", "predicted_churn"]].rename(columns={
                "CustomerID": "Customer ID",
                "Risk_Tier_Display": "Risk Tier",
                "churn_probability": "Churn Probability",
                "predicted_churn": "Model Prediction (1=Churn)",
            }),
            column_config={
                "Churn Probability": st.column_config.ProgressColumn(
                    "Churn Probability",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
            use_container_width=True,
            height=350,
        )
        
        # Export CSV (without internal actual test label for clean campaign usage)
        csv_churn = f_churn[["CustomerID", "risk_tier", "churn_probability", "predicted_churn"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Churn Risk Scores (CSV)",
            data=csv_churn,
            file_name="customer_churn_risk_scores.csv",
            mime="text/csv",
        )
    else:
        st.info("Churn prediction data not found. Run `python src/churn/churn_prediction.py` to generate.")


# ---------------------------------------------------------------------------
# Tab 3: SHAP Interpretability
# ---------------------------------------------------------------------------
with tab_shap:
    st.subheader("Explainable AI: Feature Drivers of Customer Churn")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) attributes the exact contribution of each behavioral "
        "feature to the predicted churn probability based on cooperative game theory."
    )
    
    if shap_df is not None and not shap_df.empty:
        sh_c1, sh_c2 = st.columns([1, 1])
        
        with sh_c1:
            st.markdown("#### Mean Absolute SHAP Importance")
            n_f = len(shap_df)
            colors = plt.cm.plasma(np.linspace(0.85, 0.25, n_f))
            
            fig_bar, ax_bar = plt.subplots(figsize=(7, max(4, n_f * 0.45)))
            bars = ax_bar.barh(shap_df["feature"][::-1], shap_df["mean_abs_shap"][::-1], color=colors, height=0.65)
            for bar, val in zip(bars, shap_df["mean_abs_shap"][::-1]):
                ax_bar.text(
                    bar.get_width() + 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    ha="left",
                    color="white",
                    fontsize=8,
                )
            ax_bar.set_xlabel("Mean |SHAP Value|", color="white")
            ax_bar.set_title("Global Feature Importance (XGBoost)", color="white", fontsize=10, fontweight="bold")
            ax_bar.grid(axis="x", linestyle="--", alpha=0.25)
            ax_bar.set_facecolor("#0e1117")
            fig_bar.patch.set_facecolor("#0e1117")
            plt.yticks(color="white", fontsize=8.5)
            plt.xticks(color="white")
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)
            
        with sh_c2:
            st.markdown("#### Key Feature Insights")
            st.info(
                "**1. Recency (`recency_days`):** Dominant predictor. Longer dormancy before cutoff strongly elevates churn risk.\n\n"
                "**2. Purchase Frequency (`num_orders`):** Customers with fewer historical distinct order dates churn at significantly higher rates.\n\n"
                "**3. Monetary Spend (`total_spend`):** Higher spend indicates deeper relationship and customer loyalty.\n\n"
                "**4. Product Diversity (`unique_products`):** Customers purchasing across broader SKU categories exhibit stronger retention stickiness."
            )
            
            if SHAP_SUMMARY_PNG.exists():
                st.image(
                    str(SHAP_SUMMARY_PNG),
                    caption="SHAP Summary Beeswarm: Red = High Feature Value, Blue = Low Value. Rightwards = Drives Churn.",
                    use_column_width=True,
                )
    else:
        st.info("SHAP importance values not generated yet.")


# ---------------------------------------------------------------------------
# Tab 4: Customer 360° Profile Lookup
# ---------------------------------------------------------------------------
with tab_lookup:
    st.subheader("Customer 360° Profile Lookup")
    st.caption("Search any customer to view combined RFM behavioral segment and machine learning churn assessment.")
    
    lookup_input = st.text_input(
        "Enter Customer ID to inspect:",
        placeholder="e.g. 15100, 12346, 17841",
        key="c360_lookup_input",
    )
    
    if lookup_input.strip():
        try:
            cid = float(lookup_input.strip())
        except ValueError:
            st.error("Please enter a numeric Customer ID.")
            cid = None
            
        if cid is not None:
            c_row = customers_df[customers_df["CustomerID"] == cid]
            
            if c_row.empty:
                st.warning(f"Customer ID **{int(cid)}** was not found in the customer database.")
            else:
                cust = c_row.iloc[0]
                ch_row = churn_df[churn_df["CustomerID"] == cid] if churn_df is not None else pd.DataFrame()
                
                st.markdown(f"### 📋 Customer Profile: **{int(cid)}**")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Assigned Segment", cust["segment"])
                m2.metric("Total Spend", f"£{cust['monetary']:,.2f}")
                m3.metric("Purchase Frequency", f"{int(cust['frequency'])} orders")
                m4.metric("Days Since Last Purchase", f"{int(cust['recency'])} days")
                
                st.markdown("---")
                
                if not ch_row.empty:
                    ch_info = ch_row.iloc[0]
                    c_prob = float(ch_info["churn_probability"])
                    c_tier = ch_info["risk_tier"]
                    tier_col = "#ef4444" if c_tier == "High" else ("#f59e0b" if c_tier == "Medium" else "#22c55e")
                    
                    st.markdown(f"#### AI Churn Risk Assessment: <span style='color:{tier_col}; font-weight:bold'>{c_tier} Risk ({c_prob:.1%})</span>", unsafe_allow_html=True)
                    
                    # Probability Gauge
                    fig_g, ax_g = plt.subplots(figsize=(7, 1.2))
                    ax_g.barh([""], [1.0], color="#1e1e2e", height=0.5)
                    bar_col = "#ef4444" if c_prob >= HIGH_RISK_THRESHOLD else ("#f59e0b" if c_prob > LOW_RISK_THRESHOLD else "#22c55e")
                    ax_g.barh([""], [c_prob], color=bar_col, height=0.5)
                    ax_g.axvline(HIGH_RISK_THRESHOLD, color="#ef4444", linestyle="--", lw=1.2)
                    ax_g.axvline(LOW_RISK_THRESHOLD, color="#22c55e", linestyle="--", lw=1.2)
                    ax_g.set_xlim(0, 1)
                    ax_g.set_xticks([0, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, 1.0])
                    ax_g.set_xticklabels(["0%", f"{LOW_RISK_THRESHOLD:.0%}", f"{HIGH_RISK_THRESHOLD:.0%}", "100%"], color="white", fontsize=8)
                    ax_g.tick_params(colors="white", left=False, labelleft=False)
                    ax_g.set_facecolor("#0e1117")
                    fig_g.patch.set_facecolor("#0e1117")
                    fig_g.tight_layout()
                    st.pyplot(fig_g)
                    plt.close(fig_g)
                    
                    # Prescriptive recommendation
                    if c_tier == "High":
                        st.error("🚨 **Recommended Retention Action:** Customer shows strong disengagement signs. Deploy win-back incentive, VIP personalized discount, or survey immediately.")
                    elif c_tier == "Medium":
                        st.warning("⚠️ **Recommended Action:** Monitor closely. Include in mid-tier re-engagement campaigns and product recommendations.")
                    else:
                        st.success("✅ **Recommended Action:** Loyal and engaged customer. Focus on cross-selling, loyalty rewards, and product launches.")
                else:
                    st.info(f"Customer {int(cid)} was not in the 25% held-out test partition for churn scoring.")
