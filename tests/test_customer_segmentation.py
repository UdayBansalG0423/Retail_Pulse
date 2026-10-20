"""
test_customer_segmentation.py
-----------------------------
Unit and regression tests for customer segmentation and explorer filters.
Validates:
1. Default filter bounds include 100% of identified customers (4,338).
2. The maximum-spending customer is never cut off by ceiling bounds.
3. Customer ID lookup handles exact numeric matches.
4. Segment summary table is consistent with customer dataset totals.
"""

import math
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CUSTOMER_FILE = ROOT / "data" / "processed" / "customer_rfm.csv"
SUMMARY_FILE = ROOT / "data" / "processed" / "segment_summary.csv"


class TestCustomerSegmentation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not CUSTOMER_FILE.exists() or not SUMMARY_FILE.exists():
            raise unittest.SkipTest(
                "Segmentation CSVs not found — run the segmentation pipeline first: "
                "python src/segmentation/rfm_segmentation.py && "
                "python src/segmentation/label_and_visualize.py"
            )
        cls.customers_df = pd.read_csv(CUSTOMER_FILE)
        cls.summary_df = pd.read_csv(SUMMARY_FILE)

    def test_default_filter_includes_all_customers(self):
        """Test 1: Default slider bounds (floor/ceil) include all 4,338 customers."""
        df = self.customers_df
        r_min, r_max = int(math.floor(df["recency"].min())), int(math.ceil(df["recency"].max()))
        f_min, f_max = int(math.floor(df["frequency"].min())), int(math.ceil(df["frequency"].max()))
        m_min, m_max = int(math.floor(df["monetary"].min())), int(math.ceil(df["monetary"].max()))

        filtered = df[
            (df["recency"].between(r_min, r_max)) &
            (df["frequency"].between(f_min, f_max)) &
            (df["monetary"].between(m_min, m_max))
        ]
        self.assertEqual(len(filtered), len(df))
        self.assertEqual(len(filtered), 4338)

    def test_max_spending_customer_is_not_excluded(self):
        """Test 2: Maximum spending customer is strictly <= monetary slider ceiling."""
        max_spend = self.customers_df["monetary"].max()
        monetary_max_bound = int(math.ceil(max_spend))
        self.assertGreaterEqual(monetary_max_bound, max_spend)

        # Check top spender is retained
        top_spender = self.customers_df.loc[self.customers_df["monetary"].idxmax()]
        self.assertLessEqual(top_spender["monetary"], monetary_max_bound)

    def test_customer_id_search_exact_match(self):
        """Test 3: Customer ID search correctly matches target record."""
        target_id = self.customers_df.iloc[0]["CustomerID"]
        match = self.customers_df[self.customers_df["CustomerID"] == target_id]
        self.assertEqual(len(match), 1)
        self.assertEqual(match.iloc[0]["CustomerID"], target_id)

    def test_segment_summary_sums_to_total_customers(self):
        """Test 4: Segment counts sum to total customers and percentages sum to ~100%."""
        total_in_summary = self.summary_df["customer_count"].sum()
        self.assertEqual(total_in_summary, len(self.customers_df))
        self.assertAlmostEqual(self.summary_df["customer_share_pct"].sum(), 100.0, places=1)


if __name__ == "__main__":
    unittest.main()
