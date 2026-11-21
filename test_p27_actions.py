# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from p27_actions import action_frame


class P27ActionReconciliationTest(unittest.TestCase):
    def test_exchange_reference_exposes_and_repairs_missing_cash(self):
        from p27_actions import reconcile_with_raw
        actions = pd.DataFrame({
            "instrument": ["SH600000"], "ex_date": [pd.Timestamp("2024-07-18")],
            "pay_date": [pd.Timestamp("2024-07-18")], "cash_div": [0.0],
            "cash_div_tax": [0.0], "stk_div": [0.0], "raw_cash_sum": [0.0],
            "cash_source": ["baostock_implementation"],
        })
        raw = pd.DataFrame({
            "date": pd.to_datetime(["2024-07-17", "2024-07-18"]),
            "close": [10.0, 9.68], "preclose": [10.0, 9.68],
        })
        result = reconcile_with_raw(actions, raw)
        self.assertAlmostEqual(result["reference_implied_cash"].iloc[0], 0.32)
        self.assertAlmostEqual(result["cash_div"].iloc[0], 0.32)
        self.assertEqual(result["cash_source"].iloc[0], "exchange_reference_imputed")
        self.assertAlmostEqual(result["cash_reconciliation_gap"].iloc[0], 0.0)

    def test_stock_split_uses_total_multiplier_in_reference(self):
        from p27_actions import reconcile_with_raw
        actions = pd.DataFrame({
            "instrument": ["SZ000001"], "ex_date": [pd.Timestamp("2024-06-14")],
            "pay_date": [pd.Timestamp("2024-06-14")], "cash_div": [0.2],
            "cash_div_tax": [0.2], "stk_div": [0.3], "raw_cash_sum": [0.2],
            "cash_source": ["baostock_implementation"],
        })
        raw = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-13", "2024-06-14"]),
            "close": [13.2, 10.0], "preclose": [13.2, 10.0],
        })
        result = reconcile_with_raw(actions, raw)
        # 13.2 - 10.0 * (1 + 0.3) = 0.2
        self.assertAlmostEqual(result["reference_implied_cash"].iloc[0], 0.2)
        self.assertEqual(result["cash_source"].iloc[0], "baostock_implementation_stock_plan")
        self.assertAlmostEqual(result["cash_reconciliation_gap"].iloc[0], 0.0)

    def test_pure_stock_plan_keeps_implementation_despite_negative_implied_cash(self):
        from p27_actions import reconcile_with_raw
        actions = pd.DataFrame({
            "instrument": ["SH688005"], "ex_date": [pd.Timestamp("2025-03-06")],
            "pay_date": [pd.Timestamp("2025-03-06")], "cash_div": [0.0],
            "cash_div_tax": [0.0], "stk_div": [0.49], "cash_source": ["baostock_implementation"],
        })
        raw = pd.DataFrame({
            "date": pd.to_datetime(["2025-03-05", "2025-03-06"]),
            "close": [10.0, 6.70], "preclose": [10.0, 6.85],
        })
        result = reconcile_with_raw(actions, raw)
        self.assertLess(result["reference_implied_cash"].iloc[0], 0)
        self.assertEqual(result["cash_div"].iloc[0], 0.0)
        self.assertEqual(result["cash_source"].iloc[0], "baostock_implementation_stock_plan")
        self.assertTrue(pd.isna(result["pay_date"].iloc[0]))

    def test_pure_cash_negative_reference_remains_a_failure(self):
        from p27_actions import reconcile_with_raw
        actions = pd.DataFrame({
            "instrument": ["SH600000"], "ex_date": [pd.Timestamp("2024-07-18")],
            "pay_date": [pd.Timestamp("2024-07-18")], "cash_div": [0.2],
            "cash_div_tax": [0.2], "stk_div": [0.0], "cash_source": ["baostock_implementation"],
        })
        raw = pd.DataFrame({
            "date": pd.to_datetime(["2024-07-17", "2024-07-18"]),
            "close": [10.0, 10.0], "preclose": [10.0, 10.1],
        })
        result = reconcile_with_raw(actions, raw)
        self.assertEqual(result["cash_source"].iloc[0], "baostock_implementation_negative_reference")
        self.assertEqual(result["cash_div"].iloc[0], 0.2)


if __name__ == "__main__":
    unittest.main()
