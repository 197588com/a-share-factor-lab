# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

from finalize_candidates import batch_multiplicity, leave_one_year_out, deployment_gates


class FinalizeCandidatesTest(unittest.TestCase):
    def test_leave_one_year_out_detects_single_year_driver(self):
        dates = pd.to_datetime(list(pd.date_range("2023-01-02", periods=40, freq="B")) +
                               list(pd.date_range("2024-01-02", periods=40, freq="B")) +
                               list(pd.date_range("2025-01-02", periods=40, freq="B")) +
                               list(pd.date_range("2026-01-02", periods=40, freq="B")))
        daily = pd.DataFrame({"datetime": dates, "excess": np.repeat([-.001, .01, -.001, -.001], 40)})
        result = leave_one_year_out(daily)
        self.assertFalse(result["all_leave_one_out_means_positive"])
        self.assertLess(result["min_leave_one_out_mean"], 0)

    def test_batch_multiplicity_is_order_independent(self):
        entries = [{"candidate_id": "old", "kind": "factor", "window": "wf_folds8-11"},
                   {"candidate_id": "p27_csi500_alstm158", "kind": "structure", "window": "wf_folds8-11"},
                   {"candidate_id": "control", "kind": "control", "window": "wf_folds8-11"}]
        current = ["p27_csi500_alstm158", "p30_open_gross", "p30_open_net_masked"]
        result = batch_multiplicity(entries, current)
        self.assertEqual(result["n_trials"], 5)
        self.assertEqual(result["bonferroni_n"], 4)
        self.assertAlmostEqual(result["bonferroni_threshold"], 0.0125)

    def test_monthly_budget_fails_before_any_candidate_can_be_recorded(self):
        from datetime import date
        month = date.today().strftime("%Y-%m")
        entries = [{"candidate_id": f"old_signal_{i}", "kind": "signal", "window": "wf_folds8-11",
                    "date": f"{month}-01"} for i in range(2)]
        with self.assertRaises(RuntimeError):
            batch_multiplicity(entries, ["p30_open_gross"])

    def test_deployment_gate_requires_every_preregistered_condition(self):
        summary = {"overall": {"annual_net_excess": .1, "net_ir": 1.0}}
        stress = {"overall": {"annual_net_excess": .05}}
        daily = pd.DataFrame({"datetime": pd.to_datetime(["2023-01-02", "2024-01-02", "2025-01-02", "2026-01-02"]),
                              "excess": [.01, .01, .01, .01]})
        gates = deployment_gates(summary, stress, daily, dsr=.96, cluster_p=.001,
                                 bonferroni_threshold=.01, execution_health=True)
        self.assertTrue(all(gates.values()))
        gates["net_ir_gt_0_8"] = False
        self.assertFalse(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
