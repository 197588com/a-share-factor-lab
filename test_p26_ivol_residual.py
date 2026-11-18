# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

from p26_ivol_residual import (annual_cluster_if_coverage_valid, annual_cluster_negative,
                               cross_section_residual, last_allowed_signal)


class P26Test(unittest.TestCase):
    def test_residual_is_cross_sectionally_orthogonal(self):
        n = 240
        x1 = np.linspace(-1, 1, n)
        x2 = np.sin(np.arange(n) / 7)
        independent = np.cos(np.arange(n) / 11)
        df = pd.DataFrame({"X1": x1, "X2": x2,
                           "IVOL": 0.3 * pd.Series(x1).rank(pct=True) - 0.2 * pd.Series(x2).rank(pct=True) + independent,
                           "y": independent + np.linspace(0, 0.01, n)})
        out = cross_section_residual(df, ["X1", "X2"])
        self.assertEqual(out["n"], n)
        self.assertLess(out["orthogonality_mean_abs"], 1e-10)
        self.assertLess(out["orthogonality_cov_max_abs"], 1e-8)
        self.assertTrue(np.isfinite(out["residual_ic"]))
        self.assertTrue(0 <= out["projection_r2"] <= 1)

    def test_label_end_not_signal_date_controls_holdout(self):
        dates = pd.date_range("2026-06-20", "2026-07-02", freq="B")
        idx = pd.MultiIndex.from_product([dates, ["sh600000"]], names=["datetime", "instrument"])
        labels = pd.DataFrame({"y": np.zeros(len(idx))}, index=idx)
        signal_end, label_end = last_allowed_signal(labels)
        self.assertEqual(signal_end, pd.Timestamp("2026-06-26"))
        self.assertEqual(label_end, pd.Timestamp("2026-06-30"))

    def test_year_cluster_direction_is_peregistered_negative(self):
        yearly = pd.DataFrame({"year": list(range(2015, 2027)),
                               "residual_ic": -0.02 - np.arange(12) * 0.0001})
        out = annual_cluster_negative(yearly)
        self.assertLess(out["t"], 0)
        self.assertLess(out["p_one_sided_negative"], 0.0125)
        self.assertEqual(out["negative_years"], 12)

    def test_incomplete_coverage_rejects_without_fabricating_p_value(self):
        yearly = pd.DataFrame({"year": list(range(2019, 2027)),
                               "days": [241, 243, 243, 242, 242, 242, 243, 114],
                               "residual_ic": [-0.03] * 8})
        out = annual_cluster_if_coverage_valid(yearly)
        self.assertFalse(out["evaluable"])
        self.assertEqual(out["missing_years"], [2015, 2016, 2017, 2018])
        self.assertIsNone(out["mean"])
        self.assertIsNone(out["t"])
        self.assertIsNone(out["p_one_sided_negative"])


if __name__ == "__main__":
    unittest.main()
