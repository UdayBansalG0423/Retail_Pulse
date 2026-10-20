"""
validate_dataset.py
-------------------
Runs reproducible assertion checks on transactions_clean.csv.

Checks test the assumptions that downstream models (RFM, forecasting, churn)
will rely on. Passing all checks does not guarantee a perfect dataset, but it
establishes a documented, reproducible baseline.

NOTE: StockCode is not present in the ucimlrepo 6-column download, so product
uniqueness is counted on Description instead.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE = PROJECT_ROOT / "data" / "processed" / "transactions_clean.csv"


def main():
    df = pd.read_csv(FILE, parse_dates=["InvoiceDate"])

    checks = {
        "Dataset is not empty":
            len(df) > 0,

        "No missing invoice dates":
            df["InvoiceDate"].notna().all(),

        "All quantities are positive":
            df["Quantity"].gt(0).all(),

        "All unit prices are positive":
            df["UnitPrice"].gt(0).all(),

        "No is_return rows in clean file":
            (~df["is_return"]).all(),

        "Revenue matches quantity × price":
            (
                (df["Revenue"] - df["Quantity"] * df["UnitPrice"])
                .abs()
                .lt(0.000001)
                .all()
            ),

        "InvoiceDate within expected range (2010–2012)":
            df["InvoiceDate"].between("2010-01-01", "2012-12-31").all(),

        "Country column has no blanks":
            df["Country"].notna().all(),
    }

    print("\n--- VALIDATION RESULTS ---")
    all_passed = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if not all_passed:
        raise ValueError("\nValidation failed. Inspect the data before proceeding.")

    print("\nAll validation checks passed.")
    print(f"  Rows              : {len(df):,}")
    print(f"  Date range        : {df['InvoiceDate'].min().date()} -> "
          f"{df['InvoiceDate'].max().date()}")
    print(f"  Unique customers  : {df['CustomerID'].nunique():,}  "
          f"(excludes guest rows)")
    print(f"  Unique products   : {df['Description'].nunique():,}")
    print(f"  Countries         : {df['Country'].nunique()}")
    print(f"  Total revenue     : GBP {df['Revenue'].sum():,.2f}")


if __name__ == "__main__":
    main()
