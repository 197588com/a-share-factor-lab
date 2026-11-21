# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import execution_replay as er
import shadow_execution as se


class ShadowExecutionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.raw = root / "data" / "p27_universe_expansion" / "raw_daily"
        self.actions = root / "data" / "p27_universe_expansion" / "actions"
        self.raw.mkdir(parents=True)
        self.actions.mkdir(parents=True)
        self.root = root
        self.code = "SH600000"
        self.frame = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-03")], "open": [10.0], "close": [10.1],
            "preclose": [9.9], "up_limit": [10.89], "down_limit": [8.91],
            "tradestatus": [1], "instrument": [self.code],
        })

    def tearDown(self):
        self.tmp.cleanup()

    def write_price(self, frame=None):
        (self.frame if frame is None else frame).to_parquet(self.raw / f"{self.code.lower()}.parquet", index=False)

    def write_action(self):
        pd.DataFrame(columns=["instrument", "ex_date", "pay_date", "cash_div", "stk_div"]).to_parquet(
            self.actions / f"{self.code.lower()}.parquet", index=False)

    def test_preclose_maps_to_account_interface(self):
        self.write_price()
        self.write_action()
        with patch.object(se, "WS", self.root):
            store = se.P27RawStore([self.code])
        bar = store.bar(self.code, "2024-01-03")
        self.assertIsNotNone(bar)
        self.assertEqual(bar["pre_close"], 9.9)

    def test_suspended_or_nonpositive_open_cannot_trade(self):
        frame = self.frame.copy()
        frame["tradestatus"] = 0
        self.write_price(frame)
        self.write_action()
        with patch.object(se, "WS", self.root):
            store = se.P27RawStore([self.code])
        self.assertIsNone(store.bar(self.code, "2024-01-03"))

    def test_missing_corporate_action_cache_fails_closed(self):
        self.write_price()
        with patch.object(se, "WS", self.root):
            with self.assertRaises(FileNotFoundError):
                se.P27RawStore([self.code])

    def test_gap_and_limit_orders_are_blocked(self):
        acct = er.ReplayAccount(capital=100_000)
        gap = {"open": 10.2, "pre_close": 10.0, "up_limit": 11.0, "down_limit": 9.0}
        self.assertFalse(acct.buy("2024-01-03", "2024-01-02", self.code, 10_000, gap))
        self.assertEqual(acct.orders[-1]["status"], "BLOCKED")
        limit = {"open": 11.0, "pre_close": 10.0, "up_limit": 11.0, "down_limit": 9.0}
        self.assertFalse(acct.buy("2024-01-03", "2024-01-02", self.code, 10_000, limit))
        self.assertEqual(acct.orders[-1]["status"], "BLOCKED")

    def test_minimum_commission_round_lot_and_cash_constraint(self):
        self.assertEqual(er.fee_of("BUY", self.code, 1000.0), 5.01)
        bar = {"open": 1.0, "pre_close": 1.0, "up_limit": 1.1, "down_limit": 0.9}
        acct = er.ReplayAccount(capital=105.0)
        self.assertFalse(acct.buy("2024-01-03", "2024-01-02", self.code, 1000.0, bar))
        self.assertEqual(acct.orders[-1]["status"], "SKIPPED")
        self.assertGreaterEqual(acct.cash, 0)
        acct = er.ReplayAccount(capital=106.0)
        self.assertTrue(acct.buy("2024-01-03", "2024-01-02", self.code, 1000.0, bar))
        self.assertEqual(acct.pos[self.code]["qty"] % 100, 0)
        self.assertGreaterEqual(acct.cash, 0)


class ShadowScoreValidationTest(unittest.TestCase):
    def test_duplicate_or_nonfinite_scores_fail_before_qlib(self):
        valid = pd.DataFrame({"date": ["2024-01-02", "2024-01-02"],
                              "instrument": ["SH600000", "SZ000001"], "score": [0.1, 0.2]})
        se.validate_score_frame(valid)
        with self.assertRaisesRegex(AssertionError, "重复"):
            se.validate_score_frame(pd.concat([valid, valid.iloc[[0]]], ignore_index=True))
        invalid = valid.copy()
        invalid.loc[0, "score"] = float("inf")
        with self.assertRaisesRegex(AssertionError, "无穷"):
            se.validate_score_frame(invalid)


if __name__ == "__main__":
    unittest.main()


class ShadowMetricTest(unittest.TestCase):
    def test_total_return_is_independent_of_capital_scale(self):
        daily = pd.DataFrame({
            "strategy_return": [0.0, 0.10], "excess": [0.0, 0.08], "bench": [0.0, 0.02],
            "turnover": [0.0, 0.1], "fee": [0.0, 10.0], "slippage": [0.0, 5.0],
            "n_pos": [0, 1], "cash": [1_000_000.0, 900_000.0],
            "account_nav": [1_000_000.0, 1_100_000.0],
        })
        result = se.metric(daily, capital=10_000_000.0)
        self.assertAlmostEqual(result["total_net_return"], 0.10, places=12)


class ShadowOrderStatsTest(unittest.TestCase):
    def test_one_lot_skip_rate_is_explicit(self):
        orders = pd.DataFrame({
            "side": ["BUY", "BUY", "BUY", "SELL"],
            "status": ["FILLED", "SKIPPED", "BLOCKED", "FILLED"],
        })
        result = se.order_stats(orders)
        self.assertEqual(result["planned_buys"], 3)
        self.assertEqual(result["one_lot_skips"], 1)
        self.assertAlmostEqual(result["one_lot_skip_rate"], 1 / 3)
