"""
setup_pipeline.py
-----------------
Convenience script: runs the entire RetailPulse data and model pipeline
in the correct dependency order so that a fresh clone can be made demo-ready
with a single command:

    python setup_pipeline.py

Each stage prints its status. Stages that depend on optional packages
(prophet, xgboost, shap) print a clear skip message if those packages
are missing rather than raising a hard error.

Usage
-----
    # Full pipeline
    python setup_pipeline.py

    # Skip individual slow stages
    python setup_pipeline.py --skip-forecast --skip-churn
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str, label: str) -> bool:
    """Run a pipeline script and return True on success."""
    path = ROOT / script
    if not path.exists():
        print(f"  ⚠️  {label}: script not found at {script} — skipping")
        return False
    print(f"  ▶  {label} ...", flush=True)
    result = subprocess.run([sys.executable, str(path)], capture_output=False)
    if result.returncode != 0:
        print(f"  ❌ {label} FAILED (exit {result.returncode})")
        return False
    print(f"  ✅ {label} done")
    return True


def check_import(package: str) -> bool:
    """Return True if `package` can be imported."""
    import importlib.util
    return importlib.util.find_spec(package) is not None


def main():
    parser = argparse.ArgumentParser(
        description="RetailPulse — full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Stages (in order):
              1. Data ingestion & cleaning
              2. Customer segmentation
              3. Demand forecasting      (requires prophet)
              4. Inventory optimisation
              5. Churn prediction        (requires xgboost, shap)
        """),
    )
    parser.add_argument("--skip-forecast", action="store_true", help="Skip demand forecasting stage")
    parser.add_argument("--skip-churn", action="store_true", help="Skip churn prediction stage")
    args = parser.parse_args()

    print("\n═══════════════════════════════════════════")
    print("  RetailPulse — Pipeline Setup")
    print("═══════════════════════════════════════════\n")

    # ── Stage 1: Data ingestion ───────────────────────────────────────────────
    print("Stage 1 · Data ingestion & cleaning")
    run("src/data/ingest_data.py", "Ingest & validate UCI Online Retail dataset")

    # ── Stage 2: Segmentation ─────────────────────────────────────────────────
    print("\nStage 2 · Customer segmentation")
    run("src/segmentation/rfm_segmentation.py", "RFM feature engineering + K-Means")
    run("src/segmentation/label_and_visualize.py", "Segment labelling & visualisations")

    # ── Stage 3: Forecasting ──────────────────────────────────────────────────
    print("\nStage 3 · Demand forecasting")
    if args.skip_forecast:
        print("  ⏭  Skipped (--skip-forecast)")
    elif not check_import("prophet"):
        print("  ⚠️  prophet not installed — skipping (pip install prophet)")
    else:
        run("src/forecasting/prepare_sales.py", "Prepare daily sales time series")
        run("src/forecasting/forecast_baseline.py", "Seasonal-Naïve baseline")
        run("src/forecasting/prophet_forecast.py", "Meta Prophet training & holdout eval")
        run("src/forecasting/generate_future_forecast.py", "30-day forward projections")

    # ── Stage 4: Inventory ────────────────────────────────────────────────────
    print("\nStage 4 · Inventory optimisation")
    run("src/inventory/create_sample_inventory.py", "Generate simulated inventory inputs")
    run("src/inventory/inventory_optimizer.py", "Compute ROP, Target Stock & orders")

    # ── Stage 5: Churn prediction ─────────────────────────────────────────────
    print("\nStage 5 · Churn prediction")
    if args.skip_churn:
        print("  ⏭  Skipped (--skip-churn)")
    elif not (check_import("xgboost") and check_import("shap")):
        print("  ⚠️  xgboost / shap not installed — skipping (pip install xgboost shap)")
    else:
        run("src/churn/churn_prediction.py", "XGBoost training + SHAP interpretability")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n═══════════════════════════════════════════")
    print("  Pipeline complete — launch dashboard with:")
    print("    streamlit run dashboard/app.py")
    print("═══════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
