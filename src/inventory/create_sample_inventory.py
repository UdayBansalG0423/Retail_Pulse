"""
create_sample_inventory.py
--------------------------
Generates explicitly labelled simulated inventory records for demonstration
purposes in the RetailPulse inventory optimization engine.

NOTE: The UCI Online Retail dataset contains historical sales transactions,
not warehouse stock levels, supplier lead times, or backorders. The values
in this file are SIMULATED DEMO DATA for testing inventory algorithms.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "sample_inventory.csv"

# DEMO DATA ONLY — explicitly marked simulated records
SAMPLE_ROWS = [
    {
        "product": "WHITE HANGING HEART T-LIGHT HOLDER",
        "current_stock": 250,
        "on_order": 0,
        "backorders": 0,
        "lead_time_days": 7,
        "safety_stock": 100,
        "review_period_days": 7,
        "data_type": "SIMULATED",
    },
    {
        "product": "REGENCY CAKESTAND 3 TIER",
        "current_stock": 450,
        "on_order": 50,
        "backorders": 0,
        "lead_time_days": 14,
        "safety_stock": 40,
        "review_period_days": 7,
        "data_type": "SIMULATED",
    },
    {
        "product": "JUMBO BAG RED RETROSPOT",
        "current_stock": 500,
        "on_order": 0,
        "backorders": 30,
        "lead_time_days": 10,
        "safety_stock": 150,
        "review_period_days": 7,
        "data_type": "SIMULATED",
    },
    {
        "product": "ASSORTED COLOUR BIRD ORNAMENT",
        "current_stock": 2500,
        "on_order": 0,
        "backorders": 0,
        "lead_time_days": 7,
        "safety_stock": 100,
        "review_period_days": 7,
        "data_type": "SIMULATED",
    },
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(SAMPLE_ROWS)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Sample inventory file created at: {OUTPUT_FILE}")
    print(f"Products count: {len(df)}")
    print("\nSimulated Inventory Records:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
