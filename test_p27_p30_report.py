# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

import p27_p30_report as report


class LabelDiagnosticsTest(unittest.TestCase):
    def test_ic_and_quintiles_are_label_only(self):
        scores = pd.concat([
            pd.DataFrame({
                "date": [pd.Timestamp(day)] * 5,
                "instrument": [f"S{i}" for i in range(5)],
                "score": np.arange(5, dtype=float),
                "label": np.arange(-2, 3, dtype=float) / 100,
            }) for day in ("2024-01-02", "2024-01-03")
        ], ignore_index=True)
        ic, quintiles = report.label_diagnostics(scores, min_names=1)
        self.assertEqual(len(ic), 2)
        np.testing.assert_allclose(ic.to_numpy(), [1.0, 1.0])
        np.testing.assert_allclose(quintiles.to_numpy(), [-0.02, -0.01, 0.0, 0.01, 0.02])

    def test_invalid_scores_fail_closed(self):
        bad = pd.DataFrame({"date": ["2024-01-02"], "instrument": ["S1"], "score": [1]})
        with self.assertRaises(ValueError):
            report.label_diagnostics(bad)


class YearlyTableTest(unittest.TestCase):
    def test_subset_does_not_require_other_candidates(self):
        verdict = {"folds": [{"year": 2023, "fold": 8, "annual_net_excess": 0.01,
                              "net_ir": 0.2, "mdd": -0.1, "unfilled_buy_rate": 0.05}]}
        table = report.yearly_table({"p27_csi500_alstm158": verdict})
        self.assertIn("P27 CSI500", table)
        self.assertEqual(len(table.splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
