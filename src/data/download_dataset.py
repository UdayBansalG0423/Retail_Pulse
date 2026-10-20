from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Find the RetailPulse project root from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "online_retail.csv"


def main():
    print("Downloading UCI Online Retail dataset...")
    # Dataset ID 352 identifies the official UCI Online Retail dataset.
    retail = fetch_ucirepo(id=352)

    # Combine feature and target tables if both contain columns.
    features = retail.data.features
    targets = retail.data.targets

    if targets is not None and not targets.empty:
        df = pd.concat([features, targets], axis=1)
    else:
        df = features.copy()

    # Keep the original column names for now.
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDownload complete.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
