"""
label_and_visualize.py
-----------------------
1. Assigns business-friendly segment names to the cluster IDs in
   customer_rfm.csv and writes an updated customer_rfm.csv.
2. Recalculates segment_summary.csv with the named segments.
3. Saves three charts to reports/segmentation/:
   - segment_sizes.png        : bar chart of customer counts per segment
   - rfm_profiles.png         : grouped bar chart of median RFM values
   - recency_vs_spend.png     : scatter of recency vs. monetary, coloured by segment

CAUTION: SEGMENT_LABELS maps cluster IDs from the CURRENT four-cluster model.
If you retrain with different k or significantly changed data, re-inspect the
cluster medians and update this mapping before assigning names.
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "segmentation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RFM_FILE = DATA_DIR / "customer_rfm.csv"

# Mapping based on the Dec 2010 - Dec 2011 four-cluster results.
# Cluster IDs are arbitrary; always verify against actual medians.
SEGMENT_LABELS = {
    0: "Champions",
    3: "Promising / New",
    2: "At-Risk Regulars",
    1: "Inactive / Low Engagement",
}

# Colours consistent across all charts
PALETTE = {
    "Champions":                "#2563EB",   # blue
    "Promising / New":          "#16A34A",   # green
    "At-Risk Regulars":         "#D97706",   # amber
    "Inactive / Low Engagement": "#DC2626",  # red
}


def assign_labels(rfm: pd.DataFrame) -> pd.DataFrame:
    found = set(rfm["cluster"].unique())
    expected = set(SEGMENT_LABELS)
    if found != expected:
        raise ValueError(
            f"Cluster IDs changed. Found {found}; expected {expected}. "
            "Review SEGMENT_LABELS before assigning names."
        )
    rfm = rfm.copy()
    rfm["segment"] = rfm["cluster"].map(SEGMENT_LABELS)
    return rfm


def build_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rfm.groupby("segment")
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
        summary["customer_count"] / summary["customer_count"].sum() * 100
    ).round(2)

    # Preserve a meaningful display order
    order = list(SEGMENT_LABELS.values())
    summary["_order"] = summary["segment"].map({s: i for i, s in enumerate(order)})
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return summary


# ── Chart helpers ─────────────────────────────────────────────────────────────

def chart_segment_sizes(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE[s] for s in summary["segment"]]
    bars = ax.barh(summary["segment"], summary["customer_count"],
                   color=colors, edgecolor="white", height=0.55)

    for bar, count, pct in zip(
        bars, summary["customer_count"], summary["customer_share_pct"]
    ):
        ax.text(
            bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({pct}%)",
            va="center", fontsize=10,
        )

    ax.set_xlabel("Number of customers", fontsize=11)
    ax.set_title("Customer Segment Sizes", fontsize=13, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.set_xlim(0, summary["customer_count"].max() * 1.25)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = REPORT_DIR / "segment_sizes.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def chart_rfm_profiles(summary: pd.DataFrame):
    """Grouped bar chart of median Recency, Frequency, and Spend (normalised)."""
    segments = summary["segment"].tolist()
    x = np.arange(len(segments))
    width = 0.26

    # Normalise each metric to 0-1 so they share a common axis
    cols = {
        "Recency (days)":   "median_recency_days",
        "Frequency (dates)": "median_purchase_dates",
        "Spend (GBP)":       "median_spend_gbp",
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    bar_colors = ["#60A5FA", "#34D399", "#FBBF24"]

    for i, (label, col) in enumerate(cols.items()):
        values = summary[col].values
        norm = values / values.max()
        ax.bar(x + i * width - width, norm, width,
               label=label, color=bar_colors[i], edgecolor="white", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(segments, fontsize=10, rotation=10, ha="right")
    ax.set_ylabel("Normalised value (0 = min, 1 = max)", fontsize=11)
    ax.set_title("Median RFM Profiles by Segment", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = REPORT_DIR / "rfm_profiles.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def chart_recency_vs_spend(rfm: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    for segment, group in rfm.groupby("segment"):
        ax.scatter(
            group["recency"], group["monetary"],
            label=segment, alpha=0.45, s=18,
            color=PALETTE[segment], edgecolors="none",
        )

    ax.set_xlabel("Recency (days since last purchase)", fontsize=11)
    ax.set_ylabel("Total spend (GBP)", fontsize=11)
    ax.set_title("Recency vs. Spend by Segment", fontsize=13,
                 fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{int(x):,}")
    )
    ax.legend(fontsize=9, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.35)

    plt.tight_layout()
    path = REPORT_DIR / "recency_vs_spend.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rfm = pd.read_csv(RFM_FILE)

    print("[1/4] Assigning segment labels...")
    rfm = assign_labels(rfm)

    print("[2/4] Building segment summary...")
    summary = build_summary(rfm)

    # Persist labelled customer table and summary
    rfm.to_csv(DATA_DIR / "customer_rfm.csv", index=False)
    summary.to_csv(DATA_DIR / "segment_summary.csv", index=False)

    print("\n--- UPDATED SEGMENT SUMMARY ---")
    pd.set_option("display.float_format", "{:,.2f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 130)
    print(summary.to_string(index=False))

    print("\n[3/4] Generating charts...")
    chart_segment_sizes(summary)
    chart_rfm_profiles(summary)
    chart_recency_vs_spend(rfm)

    print("\n[4/4] Done.")
    print(f"  customer_rfm.csv   : {len(rfm):,} rows with segment column")
    print(f"  segment_summary.csv: {len(summary)} segments")


if __name__ == "__main__":
    main()
