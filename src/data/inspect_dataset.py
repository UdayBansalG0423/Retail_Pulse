from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "online_retail.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    print("\n--- DATASET SHAPE ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- COLUMN TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    missing = df.isna().sum()
    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": (missing / len(df) * 100).round(2),
    })
    print(missing_report.sort_values("missing_percent", ascending=False))

    print("\n--- DUPLICATE ROWS ---")
    print(f"Exact duplicates: {df.duplicated().sum():,}")

    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe(include="all").T.to_string())

    print("\n--- SAMPLE RECORDS ---")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
