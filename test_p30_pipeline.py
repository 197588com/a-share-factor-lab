# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

from p30_pipeline import FORMAL_END, REFERENCE_ORDER_VALUE, health, make_labels


class P30PipelineTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(["2026-06-24", "2026-06-25", "2026-06-26",
                                     "2026-06-29", "2026-06-30", "2026-07-01"])
        self.old = np.ones((6, 1), dtype=np.float32)
        self.open = np.array([[10.0], [10.0], [11.0], [12.1], [12.0], [20.0]])
        self.close = self.open.copy()

    def test_open_gross_formula_and_formal_end(self):
        label = make_labels(self.old, self.open, self.close, self.dates, ["SH600000"], "p30_open_gross")
        np.testing.assert_allclose(label[:3, 0], [0.10, 0.10, 12.0 / 12.1 - 1], rtol=1e-6)
        self.assertTrue(np.isnan(label[3:]).all())

    def test_segment_end_does_not_borrow_next_segment_returns(self):
        label = make_labels(self.old, self.open, self.close, self.dates, ["SH600000"],
                            "p30_open_gross", allowed_end="2026-06-26")
        self.assertTrue(np.isfinite(label[0, 0]))
        self.assertTrue(np.isnan(label[1:]).all())

    def test_open_net_uses_minimum_commission(self):
        label = make_labels(self.old, self.open, self.close, self.dates, ["SH600000"],
                            "p30_open_net_masked")
        commission = max(5.0, REFERENCE_ORDER_VALUE * 0.00025) / REFERENCE_ORDER_VALUE
        expected = ((1 - (0.001 + commission + 0.0005 + 0.00001)) /
                    (1 + (0.001 + commission + 0.00001))) * 1.10 - 1
        self.assertAlmostEqual(float(label[0, 0]), expected, places=7)

    def test_net_mask_rejects_gap_limit_and_invalid_open(self):
        old = np.ones((6, 4), dtype=np.float32)
        opened = np.repeat(self.open, 4, axis=1)
        closed = opened.copy()
        insts = ["SH600000", "SZ000001", "SZ000002", "SH600003"]
        opened[1, 1] = 10.16  # 跳空 >1.5%
        opened[1, 2] = 10.99  # 达到近似开盘涨停阈值
        opened[2, 3] = np.nan  # T+2 无效开盘
        label = make_labels(old, opened, closed, self.dates, insts, "p30_open_net_masked")
        self.assertTrue(np.isfinite(label[0, 0]))
        self.assertTrue(np.isnan(label[0, 1:]).all())

    def test_net_mask_rejects_missing_signal_close(self):
        closed = self.close.copy()
        closed[0, 0] = np.nan
        label = make_labels(self.old, self.open, closed, self.dates, ["SH600000"],
                            "p30_open_net_masked")
        self.assertTrue(np.isnan(label[0, 0]))

    def test_nonpositive_entry_or_exit_open_is_not_a_label(self):
        for kind in ("p30_open_gross", "p30_open_net_masked"):
            for index in (1, 2):
                opened = self.open.copy()
                opened[index, 0] = 0.0
                label = make_labels(self.old, opened, self.close, self.dates,
                                    ["SH600000"], kind)
                self.assertTrue(np.isnan(label[0, 0]), (kind, index))

    def test_health_handles_last_rows_and_sealed_dates(self):
        label = make_labels(self.old, self.open, self.close, self.dates, ["SH600000"], "p30_open_gross")
        blob = {"panels": {s: (np.zeros((6, 1, 1)), np.ones((6, 1))) for s in ("train", "valid", "test")},
                "meta": {"dates": self.dates}}
        checks = health(blob, {s: label.copy() for s in ("train", "valid", "test")})
        self.assertTrue(all(checks.values()))
        bad = label.copy()
        bad[-1, 0] = 0.01
        with self.assertRaises(AssertionError):
            health(blob, {s: bad.copy() for s in ("train", "valid", "test")})


if __name__ == "__main__":
    unittest.main()


class P30ScoreBoundaryTest(unittest.TestCase):
    def test_score_indices_exclude_incomplete_and_sealed_dates(self):
        from p30_pipeline import score_indices
        dates = pd.to_datetime(["2026-06-24", "2026-06-25", "2026-06-26",
                                "2026-06-29", "2026-06-30", "2026-07-01"])
        old = np.ones((6, 1), dtype=np.float32)
        opened = np.array([[10.0], [10.0], [11.0], [12.1], [12.0], [20.0]])
        labels = make_labels(old, opened, opened, dates, ["SH600000"], "p30_open_gross")
        blob = {"meta": {"dates": dates}}
        np.testing.assert_array_equal(score_indices(blob, labels), np.array([0, 1, 2]))
