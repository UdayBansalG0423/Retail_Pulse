"""
test_inventory_optimizer.py
---------------------------
Unit tests for RetailPulse inventory optimization engine.
Validates replenishment calculations, edge cases, input validation, and data labels.

Test Coverage:
1. Stock exceeds target -> Suggested order is zero (Q = 0)
2. Stock position below target -> Suggested order fills gap (Q = ceil(T - IP))
3. On-order stock increases -> Reorder quantity decreases
4. Backorders increase -> Inventory position decreases (Q increases)
5. Demand or lead time invalid -> Raises ValueError
6. Sample inventory used -> Retains SIMULATED_DEMO_DATA label
"""

import unittest
from pathlib import Path
import pandas as pd
import sys

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inventory.inventory_optimizer import (
    calculate_inventory_metrics,
    compute_recommendations,
)


class TestInventoryOptimizer(unittest.TestCase):

    def test_stock_exceeds_target_yields_zero_order(self):
        """Rule 1: If inventory position exceeds target stock, suggested order is 0 (NO_REORDER)."""
        # d = 10, L = 7, R = 7, SS = 20 -> T = 10*(14) + 20 = 160
        # Current stock = 200 -> IP = 200 > 160
        res = calculate_inventory_metrics(
            current_stock=200,
            on_order=0,
            backorders=0,
            lead_time_days=7,
            safety_stock=20,
            review_period_days=7,
            avg_daily_demand=10.0,
        )
        self.assertEqual(res["suggested_order_qty"], 0)
        self.assertEqual(res["action"], "NO_REORDER")
        self.assertEqual(res["urgency"], "LOW")

    def test_stock_below_target_fills_gap(self):
        """Rule 2: If inventory position is below target, order fills the gap to target."""
        # d = 20, L = 7, R = 7, SS = 30 -> T = 20*14 + 30 = 310
        # IP = 100 -> Gap = 310 - 100 = 210
        res = calculate_inventory_metrics(
            current_stock=100,
            on_order=0,
            backorders=0,
            lead_time_days=7,
            safety_stock=30,
            review_period_days=7,
            avg_daily_demand=20.0,
        )
        self.assertEqual(res["target_stock"], 310)
        self.assertEqual(res["inventory_position"], 100)
        self.assertEqual(res["suggested_order_qty"], 210)
        self.assertEqual(res["action"], "REORDER_NOW")
        self.assertEqual(res["urgency"], "HIGH")

    def test_on_order_stock_increases_reduces_reorder_qty(self):
        """Rule 3: Increasing on-order inventory reduces suggested order quantity by exact amount."""
        res_no_order = calculate_inventory_metrics(
            current_stock=100,
            on_order=0,
            backorders=0,
            lead_time_days=7,
            safety_stock=30,
            review_period_days=7,
            avg_daily_demand=20.0,
        )
        res_with_order = calculate_inventory_metrics(
            current_stock=100,
            on_order=50,
            backorders=0,
            lead_time_days=7,
            safety_stock=30,
            review_period_days=7,
            avg_daily_demand=20.0,
        )
        # Order should decrease by exactly 50
        self.assertEqual(
            res_no_order["suggested_order_qty"] - res_with_order["suggested_order_qty"],
            50,
        )
        self.assertEqual(res_with_order["inventory_position"], 150)

    def test_backorders_increase_decreases_inventory_position(self):
        """Rule 4: Increasing backorders decreases IP and increases suggested order quantity."""
        res_normal = calculate_inventory_metrics(
            current_stock=100,
            on_order=0,
            backorders=0,
            lead_time_days=7,
            safety_stock=30,
            review_period_days=7,
            avg_daily_demand=20.0,
        )
        res_backlog = calculate_inventory_metrics(
            current_stock=100,
            on_order=0,
            backorders=40,
            lead_time_days=7,
            safety_stock=30,
            review_period_days=7,
            avg_daily_demand=20.0,
        )
        # IP drops from 100 to 60
        self.assertEqual(res_backlog["inventory_position"], 60)
        self.assertEqual(
            res_backlog["suggested_order_qty"] - res_normal["suggested_order_qty"],
            40,
        )

    def test_invalid_parameters_raise_value_error(self):
        """Rule 5: Negative lead times, negative safety stock, or negative demand raise ValueError."""
        # Negative lead time
        with self.assertRaises(ValueError):
            calculate_inventory_metrics(100, 0, 0, -5, 10, 7, 20.0)

        # Negative safety stock
        with self.assertRaises(ValueError):
            calculate_inventory_metrics(100, 0, 0, 7, -10, 7, 20.0)

        # Zero or negative review period
        with self.assertRaises(ValueError):
            calculate_inventory_metrics(100, 0, 0, 7, 10, 0, 20.0)

        # Negative demand
        with self.assertRaises(ValueError):
            calculate_inventory_metrics(100, 0, 0, 7, 10, 7, -5.0)

    def test_sample_inventory_preserves_simulated_data_label(self):
        """Rule 6: compute_recommendations output retains data_source='SIMULATED_DEMO_DATA'."""
        forecast_df = pd.DataFrame([
            {"product": "TEST_SKU", "date": f"2011-12-{d:02d}", "yhat": 15.0}
            for d in range(1, 31)
        ])
        inventory_df = pd.DataFrame([{
            "product": "TEST_SKU",
            "current_stock": 50,
            "on_order": 0,
            "backorders": 0,
            "lead_time_days": 7,
            "safety_stock": 25,
            "review_period_days": 7,
            "data_type": "SIMULATED_DEMO_DATA",
        }])

        rec_df = compute_recommendations(forecast_df, inventory_df)
        self.assertEqual(len(rec_df), 1)
        self.assertEqual(rec_df.iloc[0]["data_source"], "SIMULATED_DEMO_DATA")
        self.assertEqual(rec_df.iloc[0]["product"], "TEST_SKU")
        self.assertIn("suggested_order_qty", rec_df.columns)
        self.assertGreater(rec_df.iloc[0]["suggested_order_qty"], 0)


if __name__ == "__main__":
    unittest.main()
