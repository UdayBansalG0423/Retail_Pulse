"""
prepare_sales.py
----------------
Aggregates raw transaction-level data into daily product-level sales.

Input  : data/raw/online_retail.csv
Output : data/processed/daily_product_sales.csv

Each row in the output represents the total units sold for a single
product on a single calendar day.  Returns, cancellations, and rows
with missing dates or descriptions are excluded so that only genuine
positive demand is counted.

This is a demand proxy, not a perfect measurement of true customer
demand: stockouts, missing invoices, and post-sale returns can affect
the observations.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE   = PROJECT_ROOT / "data" / "raw" / "online_retail.csv"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE  = OUTPUT_DIR / "daily_product_sales.csv"


def main() -> None:
    print(f"Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # ── 1. Parse dates ────────────────────────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # ── 2. Drop rows that cannot contribute to positive demand ────────────
    df = df.dropna(subset=["InvoiceDate", "Description"])
    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ].copy()

    # ── 3. Standardise product labels ─────────────────────────────────────
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()

    # ── 4. Aggregate to daily product demand ──────────────────────────────
    daily_sales = (
        df.groupby(
            [pd.Grouper(key="InvoiceDate", freq="D"), "Description"],
            as_index=False,
        )["Quantity"]
        .sum()
        .rename(columns={
            "InvoiceDate": "date",
            "Description": "product",
            "Quantity":    "units_sold",
        })
    )

    # ── 5. Save ───────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_sales.to_csv(OUTPUT_FILE, index=False)

    # ── 6. Summary ────────────────────────────────────────────────────────
    print(f"\nSaved  : {OUTPUT_FILE}")
    print(f"Rows   : {len(daily_sales):,}")
    print(f"Products: {daily_sales['product'].nunique():,}")
    print(
        f"Date range: {daily_sales['date'].min().date()} "
        f"to {daily_sales['date'].max().date()}"
    )
    print("\nTop 10 rows:")
    print(daily_sales.head(10).to_string(index=False))

    # ── 7. Extra validation: flag products with very sparse history ────────
    obs_per_product = daily_sales.groupby("product")["units_sold"].count()
    sparse = (obs_per_product < 30).sum()
    print(
        f"\nNote: {sparse:,} products have fewer than 30 daily observations "
        "(may be unsuitable for time-series forecasting)."
    )
    print(
        f"      {(obs_per_product >= 30).sum():,} products have 30+ observations."
    )


if __name__ == "__main__":
    main()
