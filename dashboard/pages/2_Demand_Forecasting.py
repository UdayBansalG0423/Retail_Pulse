"""
2_Demand_Forecasting.py
-----------------------
Streamlit page for RetailPulse Demand Forecasting.
Provides:
  - Model evaluation comparison: Seasonal-Naive Baseline vs. Meta Prophet
  - Test set holdout prediction analysis (MAE & WAPE)
  - Pre-generated visual diagnostic plots
  - 30-day forward demand forecast curves for procurement
  - Prediction CSV export
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("dark_background")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "forecasting"

BASELINE_PRED_FILE = DATA_DIR / "baseline_test_predictions.csv"
PROPHET_PRED_FILE = DATA_DIR / "prophet_test_predictions.csv"
FORECAST_30D_FILE = DATA_DIR / "forecast_30_day.csv"
PROPHET_IMG = REPORT_DIR / "prophet_forecast.png"
BASELINE_IMG = REPORT_DIR / "baseline_forecast.png"

st.set_page_config(
    page_title="RetailPulse | Demand Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Demand Forecasting Engine")
st.caption("AI-driven retail sales forecasting: Meta Prophet vs. Seasonal-Naïve Baseline benchmark.")


@st.cache_data(show_spinner=False)
def load_forecast_data():
    b_df = pd.read_csv(BASELINE_PRED_FILE, parse_dates=["date"]) if BASELINE_PRED_FILE.exists() else None
    p_df = pd.read_csv(PROPHET_PRED_FILE, parse_dates=["date"]) if PROPHET_PRED_FILE.exists() else None
    f30_df = pd.read_csv(FORECAST_30D_FILE, parse_dates=["date"]) if FORECAST_30D_FILE.exists() else None
    return b_df, p_df, f30_df


b_df, p_df, f30_df = load_forecast_data()

# -----------------------------------------------------------------------------
# Top Metric KPI Cards
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

if p_df is not None and b_df is not None:
    p_mae = (p_df["actual_units"] - p_df["predicted_units"]).abs().mean()
    p_wape = (p_df["actual_units"] - p_df["predicted_units"]).abs().sum() / p_df["actual_units"].sum() * 100
    b_mae = (b_df["actual_units"] - b_df["predicted_units"]).abs().mean()
    b_wape = (b_df["actual_units"] - b_df["predicted_units"]).abs().sum() / b_df["actual_units"].sum() * 100

    with col1:
        st.metric("Baseline MAE", f"{b_mae:.2f} u/day")
    with col2:
        st.metric("Prophet MAE", f"{p_mae:.2f} u/day", delta=f"{b_mae - p_mae:.2f} u/day", delta_color="normal")
    with col3:
        st.metric("Baseline WAPE", f"{b_wape:.2f}%")
    with col4:
        st.metric("Prophet WAPE", f"{p_wape:.2f}%", delta=f"{- (b_wape - p_wape):.2f}%", delta_color="normal")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📊 Model Comparison & Diagnostics",
    "🔮 30-Day Forward Forecast",
    "📄 Holdout Test Predictions Table",
])

# -----------------------------------------------------------------------------
# Tab 1: Comparison & Visuals
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Model Benchmark (42-Day Holdout: Oct 28 – Dec 08, 2011)")
    
    comp_data = {
        "Model": ["Seasonal-Naïve Baseline (Lag-7)", "Meta Prophet (Weekly Seasonality)"],
        "MAE (units/day)": [f"{b_mae:.2f}", f"{p_mae:.2f}"],
        "WAPE (%)": [f"{b_wape:.2f}%", f"{p_wape:.2f}%"],
        "Seasonality": ["Weekly lag", "Additive Fourier series + trend"],
        "Outlier Robustness": ["Low (echoes spikes)", "High (smoothes anomalies)"],
    }
    st.table(pd.DataFrame(comp_data))

    st.subheader("Visual Evaluation Chart")
    if PROPHET_IMG.exists():
        st.image(str(PROPHET_IMG), caption="Prophet vs. Baseline Holdout Evaluation (Pilot SKU: WHITE HANGING HEART T-LIGHT HOLDER)")
    elif BASELINE_IMG.exists():
        st.image(str(BASELINE_IMG), caption="Baseline Forecast Chart")

# -----------------------------------------------------------------------------
# Tab 2: 30-Day Forward Forecast
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("30-Day Forward Demand Estimates (Dec 2011 – Jan 2012)")
    
    if f30_df is not None:
        products = f30_df["product"].unique().tolist()
        sel_prod = st.selectbox("Select product to view forecast horizon:", products)
        
        subset = f30_df[f30_df["product"] == sel_prod].sort_values("date")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(subset["date"], subset["yhat"], color="#3498DB", marker="o", lw=2, label="Prophet Future yhat")
        ax.fill_between(subset["date"], 0, subset["yhat"], color="#3498DB", alpha=0.2)
        ax.set_title(f"30-Day Daily Forecast: {sel_prod}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Forecast Units Sold", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.dataframe(subset, use_container_width=True)
        
        f_csv = subset.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"💾 Download {sel_prod[:20]} 30-Day Forecast CSV",
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
    st.subheader("Holdout Predictions vs Actuals")
    if p_df is not None:
        p_df_display = p_df.copy()
        p_df_display["error"] = (p_df_display["actual_units"] - p_df_display["predicted_units"]).round(2)
        st.dataframe(p_df_display, use_container_width=True)
        
        p_csv = p_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Holdout Test Predictions CSV",
            data=p_csv,
            file_name="prophet_test_predictions.csv",
            mime="text/csv",
        )
