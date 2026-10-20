"""
clean_dataset.py
----------------
Reads data/raw/online_retail.csv (6-column UCI download) and writes two
processed files to data/processed/:

  transactions_clean.csv  – deduplicated, positive-quantity sales rows with
                            a Revenue column. Safe for revenue analysis,
                            forecasting, and customer-level RFM.

  transactions_all.csv    – deduplicated full dataset including returns and
                            zero-price rows, with an is_return flag.
                            Preserved for auditing and returns analysis.

NOTE: The ucimlrepo download omits InvoiceNo and StockCode.  We therefore:
  - Use negative Quantity as the return/cancellation indicator.
  - Cannot group by invoice number; use (CustomerID, InvoiceDate, Description)
    for deduplication instead.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "online_retail.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Column names produced by the ucimlrepo download ──────────────────────────
REQUIRED_COLUMNS = {
    "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country",
}


def load_raw(path: Path) -> pd.DataFrame:
    """Load CSV and validate the expected schema."""
    df = pd.read_csv(path)

    # Strip accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Unexpected schema. Missing columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps in a documented, reproducible order."""

    rows_start = len(df)
    print(f"  Starting rows : {rows_start:,}")

    # ── Step 1: Drop exact duplicate rows ────────────────────────────────────
    df = df.drop_duplicates()
    print(f"  After drop_duplicates     : {len(df):,}  "
          f"(removed {rows_start - len(df):,})")

    # ── Step 2: Parse InvoiceDate to datetime ────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # ── Step 3: Flag returns / cancellations ─────────────────────────────────
    # Without InvoiceNo we rely on Quantity < 0 as the return indicator.
    # This is a practical heuristic; true returns analysis needs InvoiceNo.
    df["is_return"] = df["Quantity"] < 0

    # ── Step 4: Strip leading/trailing whitespace from text columns ──────────
    df["Description"] = df["Description"].str.strip()
    df["Country"] = df["Country"].str.strip()

    # ── Step 5: Drop rows where Description is missing ───────────────────────
    before = len(df)
    df = df.dropna(subset=["Description"])
    print(f"  After drop missing Desc   : {len(df):,}  "
          f"(removed {before - len(df):,})")

    # ── Step 6: CustomerID – keep as nullable Int64 ──────────────────────────
    # Missing CustomerID rows are preserved in both output files.
    # They are excluded from customer-level models (RFM, churn) but valid for
    # product-level and time-series analysis.
    df["CustomerID"] = df["CustomerID"].astype("Int64")

    # ── Step 7: Compute Revenue on the full (all-transactions) dataset ────────
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    return df


def make_clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subset of df for positive-sales analysis.

    Keeps rows where:
      - Quantity > 0  (no returns or corrections)
      - UnitPrice > 0 (no free samples or data errors)
    """
    mask = (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    sales = df[mask].copy()
    # Revenue is already correct; recalculate to be safe after filtering.
    sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]
    return sales


def main():
    print("\n[1/4] Loading raw data...")
    df = load_raw(RAW_FILE)

    print(f"\n[2/4] Cleaning {len(df):,} rows...")
    df_all = clean(df)

    print(f"\n[3/4] Subsetting to positive sales...")
    df_clean = make_clean_sales(df_all)
    print(f"  Positive-sales rows       : {len(df_clean):,}")
    print(f"  Rows with CustomerID      : {df_clean['CustomerID'].notna().sum():,}")
    print(f"  Rows without CustomerID   : {df_clean['CustomerID'].isna().sum():,}")

    print(f"\n[4/4] Saving processed files...")
    all_path = PROCESSED_DIR / "transactions_all.csv"
    clean_path = PROCESSED_DIR / "transactions_clean.csv"

    df_all.to_csv(all_path, index=False)
    df_clean.to_csv(clean_path, index=False)

    print(f"  transactions_all.csv   -> {len(df_all):,} rows  ->  {all_path}")
    print(f"  transactions_clean.csv -> {len(df_clean):,} rows  ->  {clean_path}")
    print("\nCleaning complete.")


if __name__ == "__main__":
    main()
