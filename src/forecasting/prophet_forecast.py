"""
prophet_forecast.py
-------------------
Trains and evaluates a Meta Prophet demand forecasting model on the pilot
product (WHITE HANGING HEART T-LIGHT HOLDER) using historical daily sales.

Methodology
-----------
* Input: data/processed/daily_product_sales.csv
* Continuous daily reindexing: dates with no sales are zero-filled.
* Holdout test window: final 42 days (2011-10-28 to 2011-12-08).
* Settings: weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False.
* Non-negativity: predictions are clipped at 0.
* Outputs:
    - data/processed/prophet_test_predictions.csv
    - reports/forecasting/prophet_forecast.png
"""

from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

import logging
import sys
import shutil

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure Windows finds tbb.dll for cmdstanpy/Prophet if needed
if sys.platform == "win32":
    import prophet
    prophet_stan_dir = Path(prophet.__file__).resolve().parent / "stan_model"
    target_tbb = prophet_stan_dir / "tbb.dll"
    if not target_tbb.exists():
        src_tbb = next(prophet_stan_dir.glob("**/tbb/tbb.dll"), None)
        if src_tbb and src_tbb.exists():
            shutil.copyfile(src_tbb, target_tbb)

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "daily_product_sales.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "prophet_test_predictions.csv"
BASELINE_FILE = ROOT / "data" / "processed" / "baseline_test_predictions.csv"
REPORT_IMG = ROOT / "reports" / "forecasting" / "prophet_forecast.png"

PRODUCT = "WHITE HANGING HEART T-LIGHT HOLDER"
TEST_DAYS = 42


def load_continuous_series(path: Path, product: str) -> pd.DataFrame:
    """Load daily sales and reindex across full calendar span with fill_value=0."""
    df = pd.read_csv(path, parse_dates=["date"])
    product_df = df.loc[df["product"] == product, ["date", "units_sold"]].copy()

    if product_df.empty:
        raise ValueError(f"Product '{product}' not found in {path}")

    # Aggregate in case duplicate entries exist for a single day
    product_df = product_df.groupby("date", as_index=False)["units_sold"].sum().sort_values("date")

    # Reindex to full contiguous daily calendar
    full_index = pd.date_range(product_df["date"].min(), product_df["date"].max(), freq="D")
    full_df = (
        product_df.set_index("date")
        .reindex(full_index, fill_value=0.0)
        .reset_index()
        .rename(columns={"index": "ds", "units_sold": "y"})
    )
    full_df["y"] = full_df["y"].astype(float)
    return full_df


def main():
    print("=" * 60)
    print(" RetailPulse - Prophet Demand Forecasting Model")
    print("=" * 60)
    print(f"Product    : {PRODUCT}")
    print(f"Input file : {INPUT_FILE}")

    # 1. Load and align continuous daily calendar
    df = load_continuous_series(INPUT_FILE, PRODUCT)
    total_days = len(df)
    print(f"Calendar span   : {df['ds'].min().date()} to {df['ds'].max().date()} ({total_days} days)")
    print(f"Non-zero days   : {(df['y'] > 0).sum()} | Zero-sales days: {(df['y'] == 0).sum()}")

    if total_days <= TEST_DAYS:
        raise ValueError(f"Not enough observations ({total_days}) for {TEST_DAYS}-day test window.")

    # 2. Strict train / test split
    train = df.iloc[:-TEST_DAYS].copy()
    test = df.iloc[-TEST_DAYS:].copy()

    print(f"\nTrain window    : {train['ds'].min().date()} to {train['ds'].max().date()} ({len(train)} days)")
    print(f"Test window     : {test['ds'].min().date()} to {test['ds'].max().date()} ({len(test)} days)")

    # 3. Initialize & fit Prophet model
    print("\nFitting Prophet model (weekly seasonality=True, yearly=False)...")
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(train)

    # 4. Predict on holdout test window
    future = test[["ds"]].copy()
    forecast = model.predict(future)

    # Merge predictions with actuals
    results = test.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left")
    results["yhat"] = results["yhat"].clip(lower=0.0)  # Retail demand cannot be negative
    results["absolute_error"] = (results["y"] - results["yhat"]).abs()

    # 5. Calculate Metrics
    mae = mean_absolute_error(results["y"], results["yhat"])
    total_actual = results["y"].sum()
    wape = (results["absolute_error"].sum() / total_actual * 100) if total_actual > 0 else float("nan")

    print("\n" + "-" * 60)
    print(" Evaluation Results: Prophet")
    print("-" * 60)
    print(f"  Prophet MAE   : {mae:.2f} units/day")
    print(f"  Prophet WAPE  : {wape:.2f}%")

    # 6. Compare with Baseline
    baseline_mae, baseline_wape = None, None
    baseline_df = None
    if BASELINE_FILE.exists():
        baseline_df = pd.read_csv(BASELINE_FILE, parse_dates=["date"])
        b_actual = baseline_df["actual_units"]
        b_pred = baseline_df["predicted_units"]
        baseline_mae = (b_actual - b_pred).abs().mean()
        baseline_wape = (b_actual - b_pred).abs().sum() / b_actual.sum() * 100

        print("\n" + "=" * 60)
        print(f" {'Model Comparison':<28} | {'MAE (units/day)':<15} | {'WAPE (%)':<10}")
        print("-" * 60)
        print(f" {'Seasonal-naive baseline':<28} | {baseline_mae:<15.2f} | {baseline_wape:<10.2f}%")
        print(f" {'Prophet':<28} | {mae:<15.2f} | {wape:<10.2f}%")
        mae_diff = baseline_mae - mae
        wape_diff = baseline_wape - wape
        print("-" * 60)
        if mae_diff > 0:
            print(f" Improvement: Prophet reduces MAE by {mae_diff:.2f} units/day ({mae_diff/baseline_mae*100:.1f}%)")
            print(f" Improvement: Prophet reduces WAPE by {wape_diff:.2f} percentage points")
        else:
            print(f" Notice: Prophet MAE is {-mae_diff:.2f} higher than baseline.")
        print("=" * 60)

    # 7. Save test predictions CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    export_df = pd.DataFrame({
        "date": results["ds"].dt.strftime("%Y-%m-%d"),
        "actual_units": results["y"],
        "predicted_units": results["yhat"].round(2),
        "predicted_lower": results["yhat_lower"].clip(lower=0.0).round(2),
        "predicted_upper": results["yhat_upper"].clip(lower=0.0).round(2),
    })
    export_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved test predictions to: {OUTPUT_FILE}")

    # 8. Generate & save visual plot
    REPORT_IMG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot last 60 days of train for context
    train_slice = train.iloc[-60:]
    ax.plot(train_slice["ds"], train_slice["y"], color="#7F8C8D", lw=1.2, label="Train history (last 60d)")

    # Plot test actuals
    ax.plot(test["ds"], test["y"], color="#27AE60", lw=2.0, marker="o", markersize=4, label="Actual demand (Test)")

    # Plot baseline predictions if available
    if baseline_df is not None:
        ax.plot(baseline_df["date"], baseline_df["predicted_units"], color="#E67E22",
                linestyle=":", lw=1.8, label=f"Baseline Naive (MAE={baseline_mae:.1f}, WAPE={baseline_wape:.1f}%)")

    # Plot Prophet predictions and uncertainty band
    ax.plot(results["ds"], results["yhat"], color="#2980B9", linestyle="--", lw=2.0,
            label=f"Prophet Forecast (MAE={mae:.1f}, WAPE={wape:.1f}%)")
    ax.fill_between(results["ds"], results["yhat_lower"].clip(lower=0.0), results["yhat_upper"],
                    color="#2980B9", alpha=0.15, label="Prophet 80% Uncertainty Band")

    ax.set_title(
        f"Demand Forecast Comparison - {PRODUCT}\n"
        f"Baseline WAPE: {baseline_wape:.1f}% vs Prophet WAPE: {wape:.1f}%",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Units Sold", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(REPORT_IMG, dpi=150)
    plt.close(fig)
    print(f"Saved evaluation plot to : {REPORT_IMG}")


if __name__ == "__main__":
    main()
