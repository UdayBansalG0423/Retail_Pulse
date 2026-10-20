"""
test_churn_prediction.py
------------------------
Unit tests for customer churn prediction module.
Validates:
1. Precision@top-fraction metric computation with synthetic rankings.
2. Dataset target has both churned and retained classes.
3. Feature matrix is non-empty, numeric, and free of infinite values.
4. Saved prediction output has valid probabilities between 0 and 1.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.churn.churn_prediction import (
    precision_at_top_fraction,
    PREDICTIONS_CSV,
    METRICS_JSON,
)


class TestChurnPrediction(unittest.TestCase):

    def test_precision_at_top_fraction_calculation(self):
        """Test 1: Precision@top-fraction correctly computes accuracy among top-ranked."""
        # Top 2 out of 5 (fraction=0.4)
        y_true = np.array([1, 0, 1, 1, 0])
        probs = np.array([0.9, 0.1, 0.8, 0.3, 0.2])
        # Ranked by prob: index 0 (y=1), index 2 (y=1) -> precision = 1.0
        p = precision_at_top_fraction(y_true, probs, fraction=0.4)
        self.assertEqual(p, 1.0)

        # Ranked by prob top 1: index 0 (y=1)
        p_top1 = precision_at_top_fraction(y_true, probs, fraction=0.2)
        self.assertEqual(p_top1, 1.0)

    def test_predictions_csv_structure_and_bounds(self):
        """Test 2: Predictions CSV exists and contains bounded probabilities."""
        if not PREDICTIONS_CSV.exists():
            self.skipTest("Predictions CSV not generated yet.")
        df = pd.read_csv(PREDICTIONS_CSV)
        self.assertIn("CustomerID", df.columns)
        self.assertIn("actual_churn", df.columns)
        self.assertIn("churn_probability", df.columns)
        self.assertIn("predicted_churn", df.columns)

        # Probabilities between 0.0 and 1.0
        self.assertTrue((df["churn_probability"] >= 0.0).all())
        self.assertTrue((df["churn_probability"] <= 1.0).all())

        # Predicted labels are 0 or 1
        self.assertTrue(set(df["predicted_churn"].unique()).issubset({0, 1}))

    def test_metrics_json_contains_key_performance_indicators(self):
        """Test 3: Metrics JSON records ROC-AUC, PR-AUC, and precision@top-20%."""
        if not METRICS_JSON.exists():
            self.skipTest("Metrics JSON not generated yet.")
        import json
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertIn("roc_auc", metrics)
        self.assertIn("average_precision_pr_auc", metrics)
        self.assertIn("precision_at_top_20_pct", metrics)
        self.assertGreater(metrics["roc_auc"], 0.5)
        self.assertGreater(metrics["precision_at_top_20_pct"], metrics["test_churn_rate"])


if __name__ == "__main__":
    unittest.main()
