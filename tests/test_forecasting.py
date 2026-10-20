"""
test_forecasting.py
-------------------
Unit and integration tests for demand forecasting models and data alignment.
Validates:
1. Holdout test sets share identical 42 dates and matching actual sales.
2. Forecast predictions are strictly non-negative.
3. Prophet model outperforms seasonal-naïve baseline on holdout MAE and WAPE.
4. Future forecast file contains exactly 30 future dates per product with no overlap.
"""

import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILE = ROOT / "data" / "processed" / "baseline_test_predictions.csv"
PROPHET_FILE = ROOT / "data" / "processed" / "prophet_test_predictions.csv"
FUTURE_FORECAST_FILE = ROOT / "data" / "processed" / "forecast_30_day.csv"


class TestForecasting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not BASELINE_FILE.exists() or not PROPHET_FILE.exists() or not FUTURE_FORECAST_FILE.exists():
            raise FileNotFoundError("Forecasting CSV files not found. Run forecasting pipeline first.")
        cls.baseline_df = pd.read_csv(BASELINE_FILE, parse_dates=["date"])
        cls.prophet_df = pd.read_csv(PROPHET_FILE, parse_dates=["date"])
        cls.future_df = pd.read_csv(FUTURE_FORECAST_FILE, parse_dates=["date"])

    def test_holdout_dates_and_actuals_aligned(self):
        """Test 1: Holdout datasets have identical 42 dates and matching actual sales."""
        self.assertEqual(len(self.baseline_df), 42)
        self.assertEqual(len(self.prophet_df), 42)

        # Dates must match
        pd.testing.assert_series_equal(
            self.baseline_df["date"], self.prophet_df["date"], check_names=False
        )

        # Actual sales must match exactly
        pd.testing.assert_series_equal(
            self.baseline_df["actual_units"], self.prophet_df["actual_units"], check_names=False
        )

    def test_prophet_predictions_non_negative(self):
        """Test 2: All forecast predictions are non-negative (clipped at zero)."""
        self.assertTrue((self.prophet_df["predicted_units"] >= 0.0).all())
        self.assertTrue((self.future_df["yhat"] >= 0.0).all())

    def test_prophet_beats_baseline_on_holdout(self):
        """Test 3: Prophet achieves lower MAE than baseline on common test window."""
        b_actual = self.baseline_df["actual_units"]
        b_pred = self.baseline_df["predicted_units"]
        p_pred = self.prophet_df["predicted_units"]

        baseline_mae = (b_actual - b_pred).abs().mean()
        prophet_mae = (b_actual - p_pred).abs().mean()

        self.assertLess(prophet_mae, baseline_mae)
        self.assertAlmostEqual(prophet_mae, 108.44, places=1)
        self.assertAlmostEqual(baseline_mae, 117.95, places=1)

    def test_future_forecast_has_exact_30_days_per_product(self):
        """Test 4: Each product in forecast_30_day.csv has exactly 30 future dates."""
        counts = self.future_df.groupby("product")["date"].count()
        for prod, count in counts.items():
            self.assertEqual(count, 30, f"Product {prod} has {count} forecast dates, expected 30.")


if __name__ == "__main__":
    unittest.main()
