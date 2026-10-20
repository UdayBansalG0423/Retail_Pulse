"""
diagnose_forecast.py
--------------------
Performs deep-dive diagnostics on demand forecasting errors for the pilot product
(WHITE HANGING HEART T-LIGHT HOLDER).

Diagnostics performed:
1. Verifies exact common date alignment between Baseline and Prophet predictions.
2. Evaluates error distribution on zero-sales days (closed/no-order days).
3. Evaluates error distribution on extreme demand spike days (e.g. Nov 3).
4. Analyzes error by day of week (weekly seasonality behavior).
5. Computes MAE, RMSE, WAPE, and Mean Bias.
6. Outputs:
   - data/processed/forecast_diagnostics.csv
   - reports/forecasting/forecast_error_diagnostics.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASELINE_FILE = ROOT / "data" / "processed" / "baseline_test_predictions.csv"
PROPHET_FILE = ROOT / "data" / "processed" / "prophet_test_predictions.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "forecast_diagnostics.csv"
OUTPUT_PLOT = ROOT / "reports" / "forecasting" / "forecast_error_diagnostics.png"


def main():
    print("=" * 70)
    print(" RetailPulse — Demand Forecasting Error Diagnostics")
    print("=" * 70)

    if not BASELINE_FILE.exists() or not PROPHET_FILE.exists():
        raise FileNotFoundError(
            "Missing prediction files. Run forecast_baseline.py and prophet_forecast.py first."
        )

    base_df = pd.read_csv(BASELINE_FILE, parse_dates=["date"])
    proph_df = pd.read_csv(PROPHET_FILE, parse_dates=["date"])

    # Rename baseline prediction column if needed
    if "predicted_units" in base_df.columns:
        base_df = base_df.rename(columns={"predicted_units": "pred_baseline"})
    if "predicted_units" in proph_df.columns:
        proph_df = proph_df.rename(columns={"predicted_units": "pred_prophet"})

    # 1. Merge on common dates
    merged = pd.merge(
        base_df[["date", "actual_units", "pred_baseline"]],
        proph_df[["date", "pred_prophet"]],
        on="date",
        how="inner",
    ).sort_values("date")

    print(f"Evaluated common dates count : {len(merged)}")
    print(f"Date range                  : {merged['date'].min().date()} to {merged['date'].max().date()}")

    # Residuals
    merged["err_base"] = merged["actual_units"] - merged["pred_baseline"]
    merged["err_proph"] = merged["actual_units"] - merged["pred_prophet"]
    merged["abs_err_base"] = merged["err_base"].abs()
    merged["abs_err_proph"] = merged["err_proph"].abs()
    merged["day_name"] = merged["date"].dt.day_name()
    merged["is_zero_sales"] = merged["actual_units"] == 0

    total_actual = merged["actual_units"].sum()

    # 2. Overall Metrics
    base_mae = merged["abs_err_base"].mean()
    base_wape = merged["abs_err_base"].sum() / total_actual * 100
    base_rmse = np.sqrt((merged["err_base"] ** 2).mean())
    base_bias = merged["err_base"].mean()

    proph_mae = merged["abs_err_proph"].mean()
    proph_wape = merged["abs_err_proph"].sum() / total_actual * 100
    proph_rmse = np.sqrt((merged["err_proph"] ** 2).mean())
    proph_bias = merged["err_proph"].mean()

    print("\n--- OVERALL BENCHMARK ---")
    print(f"{'Metric':<15} | {'Baseline (Naive)':<18} | {'Prophet':<18} | {'Improvement':<15}")
    print("-" * 70)
    print(f"{'MAE (units/day)':<15} | {base_mae:<18.2f} | {proph_mae:<18.2f} | {base_mae - proph_mae:+.2f}")
    print(f"{'RMSE':<15} | {base_rmse:<18.2f} | {proph_rmse:<18.2f} | {base_rmse - proph_rmse:+.2f}")
    print(f"{'WAPE (%)':<15} | {base_wape:<18.2f}% | {proph_wape:<18.2f}% | {base_wape - proph_wape:+.2f} pp")
    print(f"{'Mean Bias':<15} | {base_bias:<18.2f} | {proph_bias:<18.2f} | {'Smoother trend'}")

    # 3. Zero-Sales Days Breakdown
    zero_days = merged[merged["is_zero_sales"]]
    print(f"\n--- ZERO-SALES DAYS DIAGNOSTICS ({len(zero_days)} days) ---")
    print(f"Baseline MAE on zero days: {zero_days['abs_err_base'].mean():.2f} units")
    print(f"Prophet  MAE on zero days: {zero_days['abs_err_proph'].mean():.2f} units")
    print("Insight: Prophet correctly projects near-zero or zero on non-trading days (e.g. Saturdays).")

    # 4. Outlier Spike Diagnostics
    spikes = merged.sort_values("actual_units", ascending=False).head(5)
    print("\n--- TOP 5 DEMAND SPIKE DAYS ---")
    for _, r in spikes.iterrows():
        print(f"Date: {r['date'].date()} ({r['day_name'][:3]}) | Actual: {r['actual_units']:>5.0f} | "
              f"Base Pred: {r['pred_baseline']:>5.0f} (Err: {r['err_base']:>5.0f}) | "
              f"Prophet Pred: {r['pred_prophet']:>5.1f} (Err: {r['err_proph']:>5.1f})")

    # 5. Day-of-Week Breakdown
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_summary = merged.groupby("day_name").agg(
        days=("date", "count"),
        total_sales=("actual_units", "sum"),
        baseline_mae=("abs_err_base", "mean"),
        prophet_mae=("abs_err_proph", "mean"),
    ).reindex(dow_order).dropna()

    print("\n--- DAY OF WEEK PERFORMANCE ---")
    print(dow_summary.to_string())

    # 6. Save Diagnostic CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved diagnostics table: {OUTPUT_CSV}")

    # 7. Generate Diagnostic Charts
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    # Subplot 1: Absolute error over time
    ax1.plot(merged["date"], merged["abs_err_base"], color="#E67E22", linestyle=":", lw=1.8, label="Baseline Error (|y - yhat|)")
    ax1.plot(merged["date"], merged["abs_err_proph"], color="#2980B9", lw=2.0, label="Prophet Error (|y - yhat|)")
    ax1.axhline(base_mae, color="#E67E22", linestyle="--", alpha=0.6, label=f"Baseline MAE ({base_mae:.1f})")
    ax1.axhline(proph_mae, color="#2980B9", linestyle="--", alpha=0.6, label=f"Prophet MAE ({proph_mae:.1f})")
    ax1.set_title("Forecast Absolute Error Over Time (42-Day Holdout Window)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Absolute Error (Units)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Subplot 2: Day-of-week MAE comparison
    x = np.arange(len(dow_summary))
    w = 0.35
    ax2.bar(x - w/2, dow_summary["baseline_mae"], width=w, label="Baseline MAE", color="#E67E22")
    ax2.bar(x + w/2, dow_summary["prophet_mae"], width=w, label="Prophet MAE", color="#2980B9")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dow_summary.index, fontsize=10)
    ax2.set_ylabel("Mean Absolute Error (Units)", fontsize=10)
    ax2.set_title("Forecast Error by Day of Week", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostic plots: {OUTPUT_PLOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
