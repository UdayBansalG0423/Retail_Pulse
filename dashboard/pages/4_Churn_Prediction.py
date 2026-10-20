"""
4_Churn_Prediction.py
---------------------
Streamlit dashboard page for RetailPulse Customer Churn Prediction.

Provides:
  - Model performance summary (ROC-AUC, PR-AUC, Precision@top-20%)
  - Interactive risk-tier filter (High / Medium / Low churn probability)
  - Customer-level churn probability table with sorting and CSV export
  - SHAP feature importance bar chart (mean |SHAP|)
  - SHAP summary beeswarm plot (pre-generated PNG)
  - Customer lookup: enter any CustomerID to see their churn probability
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

plt.style.use("dark_background")

# ---------------------------------------------------------------------------
# Path setup: dashboard/pages -> dashboard -> RetailPulse root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "churn"

PREDICTIONS_FILE = DATA_DIR / "churn_customer_predictions.csv"
METRICS_FILE = REPORT_DIR / "churn_metrics.json"
SHAP_IMPORTANCE_FILE = REPORT_DIR / "churn_shap_importance.csv"
SHAP_SUMMARY_PNG = REPORT_DIR / "churn_shap_summary.png"

# Risk-tier thresholds
HIGH_THRESHOLD = 0.65
LOW_THRESHOLD = 0.35

st.set_page_config(
    page_title="RetailPulse | Churn Prediction",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Customer Churn Prediction")
st.caption(
    "XGBoost classifier with SHAP interpretability — identifies customers at risk of "
    "disengaging so retention campaigns can be prioritised."
)

st.warning(
    "**Historical test cohort — not a live risk score.**  "
    "Predictions below represent a 25% held-out sample of customers who were active "
    "**before 2011-09-30**. They are provided to evaluate model quality, not to drive "
    "production retention campaigns. ROC-AUC 0.7422 \u2014 the assignment target of 0.88 "
    "has **not yet been demonstrated** on this dataset.",
    icon="⚠️",
)

st.info(
    "**Methodology:** Churn = no transaction in the 60-day window "
    "2011-10-01 → 2011-11-29 after observation cutoff 2011-09-30. "
    "Features are engineered strictly from pre-cutoff history. "
    "Actual churn labels are shown in the table for evaluation purposes only "
    "and are excluded from the downloadable CSV.",
    icon="ℹ️",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def load_predictions():
    if not PREDICTIONS_FILE.exists():
        return None
    df = pd.read_csv(PREDICTIONS_FILE)
    df["churn_probability"] = df["churn_probability"].round(4)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_metrics():
    if not METRICS_FILE.exists():
        return None
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


@st.cache_data(ttl=60, show_spinner=False)
def load_shap_importance():
    if not SHAP_IMPORTANCE_FILE.exists():
        return None
    df = pd.read_csv(SHAP_IMPORTANCE_FILE).dropna()
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


predictions_df = load_predictions()
metrics = load_metrics()
shap_df = load_shap_importance()

if predictions_df is None or metrics is None:
    st.error(
        "Churn prediction outputs not found. "
        "Run `python src/churn/churn_prediction.py` from the project root to generate them."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Assign risk tiers
# ---------------------------------------------------------------------------
def assign_tier(p):
    if p >= HIGH_THRESHOLD:
        return "High"
    elif p <= LOW_THRESHOLD:
        return "Low"
    else:
        return "Medium"


predictions_df["risk_tier"] = predictions_df["churn_probability"].apply(assign_tier)


# ---------------------------------------------------------------------------
# Section 1: Model Performance Metrics
# ---------------------------------------------------------------------------
st.subheader("Model Performance")

m = metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("ROC-AUC", f"{m['roc_auc']:.4f}", help="Area under the ROC curve (test set)")
col2.metric("PR-AUC", f"{m['average_precision_pr_auc']:.4f}", help="Average Precision on the test set")
col3.metric(
    "Precision@Top-20%",
    f"{m['precision_at_top_20_pct']:.1%}",
    help="Precision among the top 20% highest-probability customers",
)
col4.metric(
    "Lift@Top-20%",
    f"{m['lift_at_top_20_pct']:.2f}x",
    help="Lift over random targeting among the top 20%",
)
col5.metric("Brier Score", f"{m['brier_score_loss']:.4f}", help="Calibration quality — lower is better")

st.caption(
    f"Test set: {m['test_holdout_size']:,} customers  |  "
    f"Overall churn rate: {m['overall_churn_rate']:.1%}  |  "
    f"Observation cutoff: {m['observation_cutoff']}  |  "
    f"Label window: {m['label_window_days']} days"
)

st.markdown("---")


# ---------------------------------------------------------------------------
# Section 2: Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filter Options")

tier_options = ["High", "Medium", "Low"]
selected_tiers = st.sidebar.multiselect(
    "Risk Tier",
    options=tier_options,
    default=tier_options,
    help="Filter customers by churn probability tier",
)

prob_range = st.sidebar.slider(
    "Churn Probability Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.01,
    help="Show only customers within this probability window",
)

filtered_df = predictions_df[
    (predictions_df["risk_tier"].isin(selected_tiers))
    & (predictions_df["churn_probability"] >= prob_range[0])
    & (predictions_df["churn_probability"] <= prob_range[1])
].copy()

n_high = (filtered_df["risk_tier"] == "High").sum()
n_med = (filtered_df["risk_tier"] == "Medium").sum()
n_low = (filtered_df["risk_tier"] == "Low").sum()

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtered Summary")
st.sidebar.markdown(f"🔴 **High risk:** {n_high:,}")
st.sidebar.markdown(f"🟡 **Medium risk:** {n_med:,}")
st.sidebar.markdown(f"🟢 **Low risk:** {n_low:,}")
st.sidebar.markdown(f"**Total:** {len(filtered_df):,}")


# ---------------------------------------------------------------------------
# Section 3: Risk Distribution Overview
# ---------------------------------------------------------------------------
st.subheader("Churn Risk Distribution")

oc1, oc2, oc3 = st.columns(3)
oc1.metric(
    "🔴 High Risk",
    f"{(predictions_df['risk_tier'] == 'High').sum():,}",
    help=f"Churn probability >= {HIGH_THRESHOLD:.0%}",
)
oc2.metric(
    "🟡 Medium Risk",
    f"{(predictions_df['risk_tier'] == 'Medium').sum():,}",
    help=f"{LOW_THRESHOLD:.0%} < probability < {HIGH_THRESHOLD:.0%}",
)
oc3.metric(
    "🟢 Low Risk",
    f"{(predictions_df['risk_tier'] == 'Low').sum():,}",
    help=f"Churn probability <= {LOW_THRESHOLD:.0%}",
)

# Probability histogram
fig_hist, ax_hist = plt.subplots(figsize=(10, 3))
ax_hist.hist(
    predictions_df["churn_probability"],
    bins=40,
    color="#7c3aed",
    edgecolor="#4c1d95",
    alpha=0.85,
)
ax_hist.axvline(
    HIGH_THRESHOLD, color="#ef4444", linestyle="--", linewidth=1.5,
    label=f"High threshold ({HIGH_THRESHOLD:.0%})",
)
ax_hist.axvline(
    LOW_THRESHOLD, color="#22c55e", linestyle="--", linewidth=1.5,
    label=f"Low threshold ({LOW_THRESHOLD:.0%})",
)
ax_hist.set_xlabel("Churn Probability", color="white")
ax_hist.set_ylabel("Number of Customers", color="white")
ax_hist.set_title("Distribution of Predicted Churn Probability (Test Set)", color="white")
ax_hist.tick_params(colors="white")
ax_hist.legend(framealpha=0.3)
fig_hist.tight_layout()
st.pyplot(fig_hist)
plt.close(fig_hist)

st.markdown("---")


# ---------------------------------------------------------------------------
# Section 4: Customer Churn Table
# ---------------------------------------------------------------------------
st.subheader("Customer Churn Probability Table")

if filtered_df.empty:
    st.warning("No customers match the selected filters. Adjust the sidebar controls.")
else:
    tier_emoji = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}
    display_df = (
        filtered_df[["CustomerID", "risk_tier", "churn_probability", "predicted_churn", "actual_churn"]]
        .assign(Risk_Tier=filtered_df["risk_tier"].map(tier_emoji))
        .rename(columns={
            "CustomerID": "Customer ID",
            "Risk_Tier": "Risk Tier",
            "churn_probability": "Churn Probability",
            "predicted_churn": "Predicted Churn",
            "actual_churn": "Actual Churn (Test)",
        })
        .drop(columns=["risk_tier"])
        .sort_values("Churn Probability", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Churn Probability": st.column_config.ProgressColumn(
                "Churn Probability",
                format="%.4f",
                min_value=0.0,
                max_value=1.0,
            ),
            "Risk Tier": st.column_config.TextColumn("Risk Tier"),
        },
        height=400,
    )

    st.caption(
        f"Showing {len(display_df):,} customers — sorted by churn probability (descending)"
    )

    csv_bytes = display_df.drop(columns=["Actual Churn (Test)"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered predictions (CSV)",
        data=csv_bytes,
        file_name="churn_predictions_filtered.csv",
        mime="text/csv",
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# Section 5: SHAP Feature Importance
# ---------------------------------------------------------------------------
st.subheader("SHAP Feature Importance")
st.caption(
    "Mean absolute SHAP value per feature — measures the average magnitude "
    "of each feature's contribution to the predicted churn probability."
)

tab_bar, tab_beeswarm = st.tabs(["Importance Bar Chart", "SHAP Beeswarm Plot"])

with tab_bar:
    if shap_df is None or shap_df.empty:
        st.warning("SHAP importance CSV not found. Run `src/churn/churn_prediction.py` to regenerate.")
    else:
        n = len(shap_df)
        colors = plt.cm.plasma(np.linspace(0.85, 0.25, n))

        fig_shap, ax_shap = plt.subplots(figsize=(9, max(4, n * 0.55)))
        bars = ax_shap.barh(
            shap_df["feature"][::-1],
            shap_df["mean_abs_shap"][::-1],
            color=colors,
            edgecolor="none",
            height=0.65,
        )
        for bar, val in zip(bars, shap_df["mean_abs_shap"][::-1]):
            ax_shap.text(
                bar.get_width() + 0.004,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                color="white",
                fontsize=8,
            )
        ax_shap.set_xlabel("Mean |SHAP Value|", color="white")
        ax_shap.set_title("Feature Importance — XGBoost Churn Model", color="white", pad=12)
        ax_shap.tick_params(colors="white")
        ax_shap.set_facecolor("#0e1117")
        fig_shap.patch.set_facecolor("#0e1117")
        fig_shap.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)

        st.markdown("**Top drivers of churn prediction:**")
        for _, row in shap_df.head(5).iterrows():
            st.markdown(f"- **`{row['feature']}`** — mean |SHAP| = `{row['mean_abs_shap']:.4f}`")

with tab_beeswarm:
    if SHAP_SUMMARY_PNG.exists():
        st.image(
            str(SHAP_SUMMARY_PNG),
            caption=(
                "SHAP beeswarm summary — each dot is one customer; "
                "colour indicates feature value (red = high, blue = low); "
                "position shows impact on prediction."
            ),
            use_column_width=True,
        )
    else:
        st.info(
            "The SHAP beeswarm PNG was not found at `reports/churn/churn_shap_summary.png`. "
            "Run `python src/churn/churn_prediction.py` to regenerate it."
        )

st.markdown("---")


# ---------------------------------------------------------------------------
# Section 6: Customer Lookup
# ---------------------------------------------------------------------------
st.subheader("Customer Lookup")
st.caption(
    "Enter a Customer ID (from the test set) to retrieve their individual churn probability, "
    "risk tier, and model prediction."
)

lookup_input = st.text_input(
    "Customer ID",
    placeholder="e.g. 15100",
    key="churn_lookup",
)

if lookup_input.strip():
    try:
        lookup_id = int(lookup_input.strip())
    except ValueError:
        st.error("Please enter a valid numeric Customer ID.")
        lookup_id = None

    if lookup_id is not None:
        row = predictions_df[predictions_df["CustomerID"] == lookup_id]

        if row.empty:
            st.warning(
                f"Customer **{lookup_id}** was not found in the test set. "
                "Predictions are only available for the 25% held-out test partition (~904 customers)."
            )
        else:
            r = row.iloc[0]
            prob = float(r["churn_probability"])
            tier = r["risk_tier"]
            pred_label = "Churned" if int(r["predicted_churn"]) == 1 else "Retained"
            actual_label = "Churned" if int(r["actual_churn"]) == 1 else "Retained"

            tier_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(tier, "white")
            tier_display = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}.get(tier, tier)

            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Customer ID", lookup_id)
            lc2.metric("Churn Probability", f"{prob:.2%}")
            lc3.metric("Model Prediction", pred_label)
            lc4.metric("Actual Outcome (Test)", actual_label)

            st.markdown(
                f"**Risk Tier:** <span style='color:{tier_color}; font-weight:bold'>{tier_display}</span>",
                unsafe_allow_html=True,
            )

            # Probability gauge bar
            fig_g, ax_g = plt.subplots(figsize=(6, 1.2))
            ax_g.barh([""], [1.0], color="#1e1e2e", height=0.6)
            bar_color = "#ef4444" if prob >= HIGH_THRESHOLD else ("#f59e0b" if prob > LOW_THRESHOLD else "#22c55e")
            ax_g.barh([""], [prob], color=bar_color, height=0.6)
            ax_g.axvline(HIGH_THRESHOLD, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7)
            ax_g.axvline(LOW_THRESHOLD, color="#22c55e", linestyle="--", linewidth=1.2, alpha=0.7)
            ax_g.set_xlim(0, 1)
            ax_g.set_xticks([0, LOW_THRESHOLD, HIGH_THRESHOLD, 1.0])
            ax_g.set_xticklabels(
                ["0%", f"{LOW_THRESHOLD:.0%}", f"{HIGH_THRESHOLD:.0%}", "100%"],
                color="white",
                fontsize=8,
            )
            ax_g.tick_params(colors="white", left=False, labelleft=False)
            ax_g.set_facecolor("#0e1117")
            fig_g.patch.set_facecolor("#0e1117")
            ax_g.set_title(
                f"Churn Probability Gauge — Customer {lookup_id}",
                color="white",
                fontsize=9,
            )
            fig_g.tight_layout()
            st.pyplot(fig_g)
            plt.close(fig_g)
