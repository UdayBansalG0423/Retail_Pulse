"""
rfm_segmentation.py
--------------------
Builds an RFM (Recency, Frequency, Monetary) table for each identified
customer and clusters them with K-Means.

LIMITATIONS (documented):
- InvoiceNo is absent from the ucimlrepo download, so Frequency is the count
  of distinct calendar purchase dates, NOT the number of distinct orders.
- The dataset covers Dec 2010 - Dec 2011 only.
- N_CLUSTERS = 4 is an initial choice; we will evaluate it in a later step.
- Segmentation is not churn prediction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "transactions_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_CLUSTERS = 4
RANDOM_STATE = 42


def build_rfm(df: pd.DataFrame):
    """
    Compute one RFM record per identified customer.

    Returns
    -------
    rfm : pd.DataFrame
        One row per CustomerID with columns:
        recency   – days since last purchase (lower = more recent)
        frequency – distinct purchase dates (proxy for order count)
        monetary  – total positive revenue
    analysis_date : pd.Timestamp
        The reference date used to compute recency.
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Use the day after the last transaction as the analysis reference point.
    analysis_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Drop rows without a CustomerID — kept for sales analysis but excluded
    # from customer-level models to avoid merging unrelated anonymous purchases.
    df_customers = df.dropna(subset=["CustomerID"])

    rfm = (
        df_customers
        .groupby("CustomerID")
        .agg(
            recency=("InvoiceDate", lambda x: (analysis_date - x.max()).days),
            frequency=("InvoiceDate", lambda x: x.dt.date.nunique()),
            monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    # Guard: exclude customers with zero or negative spend.
    rfm = rfm[
        (rfm["recency"] >= 0)
        & (rfm["frequency"] > 0)
        & (rfm["monetary"] > 0)
    ].copy()

    return rfm, analysis_date


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned transactions not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"CustomerID", "InvoiceDate", "Revenue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    rfm, analysis_date = build_rfm(df)

    if len(rfm) <= N_CLUSTERS:
        raise ValueError("Not enough customers for the chosen cluster count.")

    print(f"Analysis date            : {analysis_date.date()}")
    print(f"Customers for clustering : {len(rfm):,}")

    # ── Feature engineering ───────────────────────────────────────────────────
    # RFM features are right-skewed. Log1p reduces influence of extreme values.
    features = ["recency", "frequency", "monetary"]
    X_log = np.log1p(rfm[features])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # ── K-Means clustering ────────────────────────────────────────────────────
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    rfm["cluster"] = model.fit_predict(X_scaled)

    # ── Save customer-level RFM + cluster ─────────────────────────────────────
    rfm_file = OUTPUT_DIR / "customer_rfm.csv"
    rfm.to_csv(rfm_file, index=False)
    print(f"\nCustomer RFM saved   : {rfm_file}")

    # ── Cluster summary in original interpretable units ───────────────────────
    summary = (
        rfm.groupby("cluster")
        .agg(
            customer_count=("CustomerID", "nunique"),
            median_recency_days=("recency", "median"),
            median_purchase_dates=("frequency", "median"),
            median_spend_gbp=("monetary", "median"),
            average_spend_gbp=("monetary", "mean"),
        )
        .reset_index()
    )
    summary["customer_share_pct"] = (
        summary["customer_count"] / len(rfm) * 100
    ).round(2)
    summary = summary.sort_values("cluster").reset_index(drop=True)

    summary_file = OUTPUT_DIR / "segment_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Segment summary saved: {summary_file}")

    # ── Console output ────────────────────────────────────────────────────────
    print("\n--- RFM SAMPLE (first 5 rows) ---")
    print(rfm.head().to_string(index=False))

    print("\n--- CLUSTER SUMMARY ---")
    pd.set_option("display.float_format", "{:,.2f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))

    print("\n--- INTERPRETATION GUIDE ---")
    print("  Low recency + high freq + high spend  => likely loyal/high-value")
    print("  Low recency + low freq  + low spend   => likely new/recent")
    print("  High recency + high spend (historical) => at-risk / lapsed")
    print("  High recency + low freq + low spend   => inactive/low-engagement")
    print("\nNOTE: Cluster IDs are arbitrary labels; inspect median values above.")
    print("NOTE: Frequency = distinct purchase dates, not invoice count.")


if __name__ == "__main__":
    main()
