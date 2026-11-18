# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from execution_replay import ReplayAccount, _metric_row, topk_plan


class TopkPlanTest(unittest.TestCase):
    def test_underfilled_position_buys_exact_gap(self):
        scores = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
        sell, buy = topk_plan(scores, {"A": {}}, topk=3, n_drop=1)
        self.assertEqual(sell, [])
        self.assertEqual(buy, ["B", "C"])

    def test_merged_ranking_prevents_sell_high_buy_low(self):
        scores = pd.Series({"A": 100.0, "B": 2.0, "C": 1.0, "D": 50.0})
        sell, buy = topk_plan(scores, {"A": {}, "B": {}, "C": {}}, topk=3, n_drop=1)
        self.assertEqual(sell, ["C"])
        self.assertEqual(buy, ["D"])

    def test_missing_held_score_falls_to_bottom(self):
        scores = pd.Series({"A": 3.0, "B": 2.0, "D": 4.0})
        sell, buy = topk_plan(scores, {"A": {}, "B": {}, "C": {}}, topk=3, n_drop=1)
        self.assertEqual(sell, ["C"])
        self.assertEqual(buy, ["D"])


class AccountTest(unittest.TestCase):
    def test_lot_fee_and_cash_constraint(self):
        a = ReplayAccount(capital=1006.0)
        bar = {"open": 10.0, "pre_close": 9.9, "up_limit": 11.0, "down_limit": 9.0}
        ok = a.buy(pd.Timestamp("2024-01-02"), pd.Timestamp("2023-12-29"), "SZ000001", 2000.0, bar)
        self.assertTrue(ok)
        self.assertEqual(a.pos["SZ000001"]["qty"], 100)
        self.assertAlmostEqual(a.cash, 0.0, places=6)
        self.assertEqual(a.orders[-1]["status"], "FILLED")

    def test_gap_abandon_does_not_spend_cash(self):
        a = ReplayAccount(capital=10000.0)
        bar = {"open": 10.2, "pre_close": 10.0, "up_limit": 11.0, "down_limit": 9.0}
        ok = a.buy(pd.Timestamp("2024-01-02"), pd.Timestamp("2023-12-29"), "SZ000001", 2000.0, bar)
        self.assertFalse(ok)
        self.assertEqual(a.cash, 10000.0)
        self.assertEqual(a.orders[-1]["status"], "BLOCKED")

    def test_dividend_and_stock_split_preserve_nav(self):
        a = ReplayAccount(capital=0.0)
        a.pos["SH600519"] = {"qty": 100.0, "last_px": 20.0, "entry": "2024-01-01"}
        action = pd.DataFrame({
            "instrument": ["SH600519"], "ex_date": [pd.Timestamp("2024-06-03")],
            "pay_date": [pd.Timestamp("2024-06-04")], "cash_div": [1.0],
            "stk_div": [0.5], "stk_bo_rate": [0.0], "stk_co_rate": [0.0],
        })
        a.apply_actions(pd.Timestamp("2024-06-03"), {"SH600519": action})
        self.assertEqual(a.pos["SH600519"]["qty"], 150.0)
        a.pos["SH600519"]["last_px"] = (20.0 - 1.0) / 1.5
        self.assertAlmostEqual(a.equity(), 2000.0, places=6)
        self.assertAlmostEqual(a.receivable_total(), 100.0, places=6)
        a.process_payables(pd.Timestamp("2024-06-04"))
        self.assertAlmostEqual(a.cash, 100.0, places=6)
        self.assertAlmostEqual(a.receivable_total(), 0.0, places=6)
        self.assertAlmostEqual(a.equity(), 2000.0, places=6)

    def test_blocked_sell_keeps_position(self):
        a = ReplayAccount(capital=0.0)
        a.pos["SZ000001"] = {"qty": 100.0, "last_px": 10.0, "entry": "2024-01-01"}
        bar = {"open": 9.0, "pre_close": 10.0, "up_limit": 11.0, "down_limit": 9.0}
        ok = a.sell(pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02"), "SZ000001", bar)
        self.assertFalse(ok)
        self.assertIn("SZ000001", a.pos)
        self.assertEqual(a.orders[-1]["status"], "BLOCKED")


class MetricTest(unittest.TestCase):
    def test_linked_nav_does_not_distort_independent_account_cash_weight(self):
        daily = pd.DataFrame({
            "strategy_return": [0.0, 0.0], "excess": [0.0, 0.0],
            "nav": [2_000_000.0, 2_000_000.0],
            "account_nav": [1_000_000.0, 500_000.0],
            "cash": [100_000.0, 50_000.0], "bench": [0.0, 0.0],
            "turnover": [0.0, 0.0], "fee": [0.0, 0.0], "slippage": [0.0, 0.0],
            "n_pos": [1, 1],
        })
        out = _metric_row(daily)
        self.assertAlmostEqual(out["mean_cash_weight"], 0.10, places=12)


if __name__ == "__main__":
    unittest.main()
