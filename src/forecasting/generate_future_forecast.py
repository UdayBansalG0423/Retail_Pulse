"""
generate_future_forecast.py
---------------------------
Generates a 30-day future demand forecast for each product specified in the
inventory system (sample_inventory.csv).

Refits the selected Meta Prophet model on ALL available historical data for
each product and outputs future demand predictions.

Output: data/processed/forecast_30_day.csv
Columns: product, date, yhat
"""

from pathlib import Path
import sys
import shutil
import warnings
import logging
import pandas as pd
from prophet import Prophet

# Suppress Prophet / cmdstanpy logs
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
INPUT_SALES = ROOT / "data" / "processed" / "daily_product_sales.csv"
INPUT_INVENTORY = ROOT / "data" / "processed" / "sample_inventory.csv"
OUTPUT_FORECAST = ROOT / "data" / "processed" / "forecast_30_day.csv"

FUTURE_PERIODS = 30


def get_product_series(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Extract continuous zero-filled daily series for a product."""
    pdf = df[df["product"] == product].copy()
    if pdf.empty:
        raise ValueError(f"Product '{product}' not found in sales dataset.")

    pdf = pdf.groupby("date", as_index=False)["units_sold"].sum().sort_values("date")
    full_idx = pd.date_range(pdf["date"].min(), pdf["date"].max(), freq="D")
    cont_df = (
        pdf.set_index("date")
        .reindex(full_idx, fill_value=0.0)
        .reset_index()
        .rename(columns={"index": "ds", "units_sold": "y"})
    )
    cont_df["y"] = cont_df["y"].astype(float)
    return cont_df


def forecast_product(cont_df: pd.DataFrame, product: str, periods: int = 30) -> pd.DataFrame:
    """Fit Prophet on full history and produce future 30-day forecast."""
    last_date = cont_df["ds"].max()

    # Train model on all historical observations
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(cont_df)

    # Generate future 30 dates starting the day after the last historical date
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})

    preds = model.predict(future_df)
    preds["yhat"] = preds["yhat"].clip(lower=0.0)

    result = pd.DataFrame({
        "product": product,
        "date": preds["ds"].dt.strftime("%Y-%m-%d"),
        "yhat": preds["yhat"].round(2),
    })
    return result


def main():
    print("=" * 65)
    print(" RetailPulse — 30-Day Future Demand Forecast Generator")
    print("=" * 65)

    if not INPUT_SALES.exists():
        raise FileNotFoundError(f"Sales dataset not found at: {INPUT_SALES}")
    if not INPUT_INVENTORY.exists():
        raise FileNotFoundError(f"Inventory dataset not found at: {INPUT_INVENTORY}")

    sales_df = pd.read_csv(INPUT_SALES, parse_dates=["date"])
    inv_df = pd.read_csv(INPUT_INVENTORY)

    products = inv_df["product"].unique().tolist()
    print(f"Target products for future forecasting ({len(products)}):")
    for p in products:
        print(f" - {p}")

    all_forecasts = []
    for prod in products:
        print(f"\nProcessing '{prod}'...")
        cont_df = get_product_series(sales_df, prod)
        print(f"  Historical range: {cont_df['ds'].min().date()} to {cont_df['ds'].max().date()} ({len(cont_df)} days)")
        
        prod_forecast = forecast_product(cont_df, prod, periods=FUTURE_PERIODS)
        print(f"  Future forecast : {prod_forecast['date'].min()} to {prod_forecast['date'].max()} (30 days)")
        print(f"  Avg daily demand: {prod_forecast['yhat'].mean():.2f} units/day")
        all_forecasts.append(prod_forecast)

    final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
    OUTPUT_FORECAST.parent.mkdir(parents=True, exist_ok=True)
    final_forecast_df.to_csv(OUTPUT_FORECAST, index=False)

    print("\n" + "=" * 65)
    print(f"Successfully generated 30-day forecast at: {OUTPUT_FORECAST}")
    print(f"Total rows: {len(final_forecast_df)} ({len(products)} products x {FUTURE_PERIODS} days)")
    print("Sample rows:")
    print(final_forecast_df.head(6).to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
