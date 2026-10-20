"""
3_Forecast_Dashboard.py
-----------------------
RetailPulse — Page 3: Demand Forecast Dashboard

Provides:
  - Meta Prophet vs. Seasonal-Naïve Baseline benchmark evaluation (MAE & WAPE)
  - Pre-generated holdout visual diagnostic evaluation chart
  - Interactive 30-day forward demand projections by SKU with confidence bounds
  - Aligned holdout test predictions table with error analysis
  - CSV export for procurement planning
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("dark_background")

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard"))
from sidebar_theme import apply_sidebar_theme, render_sidebar_footer

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "forecasting"

BASELINE_PRED_FILE = DATA_DIR / "baseline_test_predictions.csv"
PROPHET_PRED_FILE = DATA_DIR / "prophet_test_predictions.csv"
FORECAST_30D_FILE = DATA_DIR / "forecast_30_day.csv"
DIAGNOSTICS_FILE = DATA_DIR / "forecast_diagnostics.csv"
PROPHET_IMG = REPORT_DIR / "prophet_forecast.png"
BASELINE_IMG = REPORT_DIR / "baseline_forecast.png"

st.set_page_config(
    page_title="RetailPulse | Forecast Dashboard",
    page_icon="📈",
    layout="wide",
)

apply_sidebar_theme()

st.title("📈 Demand Forecast Dashboard")
st.caption(
    "Meta Prophet vs. Seasonal-Naïve Baseline benchmark and 30-day forward demand projections."
)


@st.cache_data(show_spinner=False)
def load_forecast_data():
    b_df = pd.read_csv(BASELINE_PRED_FILE, parse_dates=["date"]) if BASELINE_PRED_FILE.exists() else None
    p_df = pd.read_csv(PROPHET_PRED_FILE, parse_dates=["date"]) if PROPHET_PRED_FILE.exists() else None
    f30_df = pd.read_csv(FORECAST_30D_FILE, parse_dates=["date"]) if FORECAST_30D_FILE.exists() else None
    diag_df = pd.read_csv(DIAGNOSTICS_FILE) if DIAGNOSTICS_FILE.exists() else None
    return b_df, p_df, f30_df, diag_df


b_df, p_df, f30_df, diag_df = load_forecast_data()

# -----------------------------------------------------------------------------
# Top Metric KPI Cards
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

if p_df is not None and b_df is not None:
    p_mae = float((p_df["actual_units"] - p_df["predicted_units"]).abs().mean())
    p_wape = float((p_df["actual_units"] - p_df["predicted_units"]).abs().sum() / p_df["actual_units"].sum() * 100)
    b_mae = float((b_df["actual_units"] - b_df["predicted_units"]).abs().mean())
    b_wape = float((b_df["actual_units"] - b_df["predicted_units"]).abs().sum() / b_df["actual_units"].sum() * 100)

    with col1:
        st.metric("Baseline MAE", f"{b_mae:.2f} u/day", help="Seasonal-Naïve (Lag-7) Mean Absolute Error")
    with col2:
        st.metric(
            "Prophet MAE",
            f"{p_mae:.2f} u/day",
            delta=f"{b_mae - p_mae:.2f} u/day better",
            delta_color="normal",
            help="Meta Prophet Mean Absolute Error on common 42-day holdout",
        )
    with col3:
        st.metric("Baseline WAPE", f"{b_wape:.2f}%", help="Weighted Absolute Percentage Error on test window")
    with col4:
        st.metric(
            "Prophet WAPE",
            f"{p_wape:.2f}%",
            delta=f"-{(b_wape - p_wape):.2f} pp",
            delta_color="normal",
            help="Prophet achieves lower error across holdout sales volume",
        )

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison & Diagnostics",
    "🔮 30-Day Forward Forecast",
    "📄 Holdout Test Predictions Table",
    "🔬 Forecast Diagnostics",
])

# -----------------------------------------------------------------------------
# Tab 1: Comparison & Visuals
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Holdout Benchmark (42 Days: Oct 28 – Dec 08, 2011)")
    
    comp_data = {
        "Model": ["Seasonal-Naïve Baseline (Lag-7)", "Meta Prophet (Weekly Seasonality)"],
        "MAE (units/day)": [f"{b_mae:.2f}", f"{p_mae:.2f}"],
        "WAPE (%)": [f"{b_wape:.2f}%", f"{p_wape:.2f}%"],
        "Seasonality": ["Weekly lag", "Additive Fourier series + trend changepoints"],
        "Outlier Robustness": ["Low (echoes spikes verbatim)", "High (smoothes anomalies via regularization)"],
        "Zero-Demand Handling": ["Can repeat zero blindly", "Continuous non-negative rate prediction"],
    }
    st.table(pd.DataFrame(comp_data))

    st.subheader("Holdout Visual Evaluation Chart")
    if PROPHET_IMG.exists():
        st.image(str(PROPHET_IMG), caption="Prophet vs. Baseline Holdout Evaluation (Pilot SKU: WHITE HANGING HEART T-LIGHT HOLDER)")
    elif BASELINE_IMG.exists():
        st.image(str(BASELINE_IMG), caption="Baseline Forecast Chart")
    else:
        st.info("Evaluation charts available under `reports/forecasting/`.")

# -----------------------------------------------------------------------------
# Tab 2: 30-Day Forward Forecast
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("30-Day Forward Demand Projections (Dec 2011 – Jan 2012)")
    
    if f30_df is not None:
        products = f30_df["product"].unique().tolist()
        sel_prod = st.selectbox("Select product to inspect forecast curve:", products)
        
        subset = f30_df[f30_df["product"] == sel_prod].sort_values("date")
        
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(subset["date"], subset["yhat"], color="#38bdf8", marker="o", markersize=4, lw=2, label="Prophet Point Forecast (yhat)")
        
        # Check if confidence intervals are available
        if "yhat_lower" in subset.columns and "yhat_upper" in subset.columns:
            ax.fill_between(
                subset["date"],
                subset["yhat_lower"].clip(lower=0),
                subset["yhat_upper"],
                color="#38bdf8",
                alpha=0.2,
                label="Uncertainty Interval [yhat_lower, yhat_upper]",
            )
        else:
            ax.fill_between(subset["date"], 0, subset["yhat"], color="#38bdf8", alpha=0.15)
            
        ax.set_title(f"30-Day Daily Forecast: {sel_prod}", fontsize=11, fontweight="bold", color="white")
        ax.set_xlabel("Date", fontsize=9, color="white")
        ax.set_ylabel("Predicted Units Sold", fontsize=9, color="white")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="upper right")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        plt.xticks(rotation=30, color="white")
        plt.yticks(color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Metrics summary
        fm1, fm2, fm3 = st.columns(3)
        fm1.metric("30-Day Total Units", f"{subset['yhat'].sum():,.0f} units")
        fm2.metric("Mean Daily Demand", f"{subset['yhat'].mean():.1f} units/day")
        fm3.metric("Peak Forecast Day", f"{subset['yhat'].max():,.0f} units")
        
        st.dataframe(subset, use_container_width=True)
        
        f_csv = subset.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"💾 Download {sel_prod[:20]} 30-Day Forecast (CSV)",
            data=f_csv,
            file_name=f"forecast_30_day_{sel_prod[:15].replace(' ', '_')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No 30-day forecast file found.")

# -----------------------------------------------------------------------------
# Tab 3: Holdout Prediction Details
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Holdout Predictions vs Actuals (42 Days)")
    if p_df is not None:
        p_df_display = p_df.copy()
        p_df_display["error"] = (p_df_display["actual_units"] - p_df_display["predicted_units"]).round(2)
        p_df_display["abs_error"] = p_df_display["error"].abs()
        
        st.dataframe(
            p_df_display.style.format({
                "actual_units": "{:,.1f}",
                "predicted_units": "{:,.1f}",
                "error": "{:+,.1f}",
                "abs_error": "{:,.1f}",
            }),
            use_container_width=True,
        )
        
        p_csv = p_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Holdout Test Predictions CSV",
            data=p_csv,
            file_name="prophet_test_predictions.csv",
            mime="text/csv",
        )

# -----------------------------------------------------------------------------
# Tab 4: Diagnostics
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Forecast Diagnostics & Anomaly Analysis")
    st.markdown(
        "Diagnostic logs track zero-demand days, demand spikes, and residuals to explain "
        "model accuracy and provide transparent audit trails."
    )
    if diag_df is not None and not diag_df.empty:
        st.dataframe(diag_df, use_container_width=True)
    else:
        st.info("Diagnostics summary generated during pipeline execution.")

render_sidebar_footer()
