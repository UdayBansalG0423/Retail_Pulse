"""
forecast_baseline.py
--------------------
Builds and evaluates a seasonal-naïve demand forecast for a single pilot
product using daily historical sales from the Online Retail II dataset.

Methodology
-----------
* Seasonal-naïve: each day is predicted using the actual sales value from
  the same weekday 7 days earlier (lag-7 / weekly seasonality).
* Evaluation window: the last TEST_DAYS days of the observed series are
  held out chronologically; the model is never exposed to future values.
* Zero-fill: dates within the product's first-to-last recorded date range
  that have no transaction rows are treated as zero recorded sales. See the
  "Data caveat" note in the summary printed at runtime.

Outputs
-------
  data/processed/baseline_test_predictions.csv  – actual vs predicted
  reports/forecasting/baseline_forecast.png      – train/test/forecast chart
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display server required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE   = PROJECT_ROOT / "data" / "processed" / "daily_product_sales.csv"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed"
REPORT_DIR   = PROJECT_ROOT / "reports" / "forecasting"

# ── Config ─────────────────────────────────────────────────────────────────
PRODUCT         = "WHITE HANGING HEART T-LIGHT HOLDER"
TEST_DAYS       = 42      # hold-out window: 6 weeks
SEASONAL_PERIOD = 7       # weekly seasonality lag


# ── Helpers ────────────────────────────────────────────────────────────────

def load_product_series(path: Path, product: str) -> pd.Series:
    """Return a zero-filled daily series for *product* within its date window."""
    df = pd.read_csv(path, parse_dates=["date"])
    product_df = df.loc[
        df["product"].eq(product), ["date", "units_sold"]
    ].copy()

    if product_df.empty:
        raise ValueError(f"Product not found: {product!r}")

    # Collapse any duplicate date rows (shouldn't happen after prepare_sales,
    # but guard against it anyway).
    product_df = (
        product_df.groupby("date", as_index=False)["units_sold"]
        .sum()
        .sort_values("date")
    )

    # Build a contiguous daily index spanning the whole observation window.
    full_index = pd.date_range(
        product_df["date"].min(),
        product_df["date"].max(),
        freq="D",
    )
    series = (
        product_df.set_index("date")["units_sold"]
        .reindex(full_index, fill_value=0)
        .astype(float)
        .rename("units_sold")
    )
    series.index.name = "date"
    return series


def seasonal_naive_forecast(train: pd.Series, test_index: pd.DatetimeIndex,
                            lag: int) -> pd.Series:
    """
    Produce a one-step-ahead seasonal-naïve forecast.

    For each date in *test_index*, the prediction equals the observed value
    *lag* days earlier, appending actuals to the history as we go so that
    the forecast for day t+2 can use the actual of day t+1.
    """
    history = train.tolist()
    preds = []
    full_series = train  # we extend this inside the loop

    for i, date in enumerate(test_index):
        preds.append(history[-lag])
        # Reveal the actual so subsequent predictions can use it.
        actual = full_series.get(date, history[-lag])
        history.append(float(actual))

    return pd.Series(preds, index=test_index, name="predicted_units")


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    """Return MAE and WAPE (weighted absolute percentage error)."""
    errors = actual - predicted
    mae = errors.abs().mean()
    total_actual = actual.sum()
    wape = errors.abs().sum() / total_actual * 100 if total_actual > 0 else np.nan
    return {"MAE": mae, "WAPE_pct": wape}


def plot_forecast(train: pd.Series, test: pd.Series,
                  forecast: pd.Series, metrics: dict,
                  product: str, out_path: Path) -> None:
    """Save a train / test / forecast chart to *out_path*."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Show only the last 90 days of training data for readability.
    ax.plot(train.iloc[-90:], color="#4A90D9", lw=1.2, label="Train (last 90 days)")
    ax.plot(test,             color="#2ECC71", lw=1.5, label="Actual (test)")
    ax.plot(forecast,         color="#E74C3C", lw=1.5, linestyle="--",
            label="Seasonal-naïve forecast")

    ax.set_title(
        f"Seasonal-Naïve Baseline — {product}\n"
        f"MAE = {metrics['MAE']:.1f} units/day  |  WAPE = {metrics['WAPE_pct']:.1f}%",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Product : {PRODUCT}")
    print(f"Input   : {INPUT_FILE}\n")

    series = load_product_series(INPUT_FILE, PRODUCT)

    print(f"Observation window : {series.index[0].date()} to {series.index[-1].date()}")
    print(f"Total days in window : {len(series)}")
    print(f"Non-zero days        : {(series > 0).sum()}")
    print(f"Zero-sales days      : {(series == 0).sum()}  (treated as zero demand)")
    print(f"Total units in window: {series.sum():,.0f}\n")

    if len(series) <= TEST_DAYS + SEASONAL_PERIOD:
        raise ValueError(
            f"Insufficient history: need > {TEST_DAYS + SEASONAL_PERIOD} days, "
            f"got {len(series)}."
        )

    # Chronological split — never shuffle time-series data.
    train = series.iloc[:-TEST_DAYS]
    test  = series.iloc[-TEST_DAYS:]

    print(f"Train period : {train.index[0].date()} to {train.index[-1].date()}  "
          f"({len(train)} days)")
    print(f"Test period  : {test.index[0].date()} to {test.index[-1].date()}  "
          f"({len(test)} days)\n")

    # Build forecast using only the training series; actuals are revealed
    # one step at a time inside the function.
    forecast = seasonal_naive_forecast(train, test.index, SEASONAL_PERIOD)

    metrics = evaluate(test, forecast)

    print("-- Evaluation --------------------------------------------------")
    print(f"  MAE  = {metrics['MAE']:.2f} units/day")
    print(f"  WAPE = {metrics['WAPE_pct']:.2f}%")
    print("----------------------------------------------------------------\n")

    # Save predictions CSV.
    results = pd.DataFrame({
        "date":            test.index,
        "actual_units":    test.values,
        "predicted_units": forecast.values,
    })
    out_csv = OUTPUT_DIR / "baseline_test_predictions.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    print(f"Predictions saved : {out_csv}")

    # Save chart.
    out_png = REPORT_DIR / "baseline_forecast.png"
    plot_forecast(train, test, forecast, metrics, PRODUCT, out_png)

    # ── Data caveat ────────────────────────────────────────────────────────
    zero_pct = (series == 0).sum() / len(series) * 100
    print(
        f"\nData caveat: {zero_pct:.1f}% of days within the observation window "
        "have zero recorded sales. These are treated as genuine zero-demand days. "
        "If some represent missing or unrecorded data, MAE and WAPE may be "
        "underestimated."
    )


if __name__ == "__main__":
    main()
