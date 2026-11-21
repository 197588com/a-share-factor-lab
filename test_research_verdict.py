# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

from research_verdict import annual_cluster_paired, annual_cluster_positive, executable_dsr


def daily_with_excess(values_by_year):
    frames = []
    for year, values in values_by_year.items():
        dates = pd.date_range(f"{year}-01-02", periods=len(values), freq="B")
        frames.append(pd.DataFrame({"datetime": dates, "excess": values}))
    return pd.concat(frames, ignore_index=True)


class ResearchVerdictTest(unittest.TestCase):
    def test_paired_year_cluster_uses_four_natural_years(self):
        reference = daily_with_excess({y: np.linspace(-0.001, 0.001, 40) for y in (2023, 2024, 2025, 2026)})
        candidate = reference.copy()
        candidate["excess"] += candidate["datetime"].dt.year.map({2023: .001, 2024: .0012, 2025: .0008, 2026: .0011})
        result = annual_cluster_paired(candidate, reference)
        self.assertTrue(result["evaluable"])
        self.assertEqual(result["years"], 4)
        self.assertEqual(result["positive_years"], 4)
        self.assertLess(result["p_one_sided_positive"], 0.05)

    def test_year_cluster_fails_closed_when_year_missing(self):
        daily = daily_with_excess({2023: [.001] * 40, 2024: [.001] * 40, 2025: [.001] * 40})
        result = annual_cluster_positive(daily)
        self.assertFalse(result["evaluable"])
        self.assertIsNone(result["p_one_sided_positive"])

    def test_dsr_uses_real_trial_sharpes_and_declares_coverage(self):
        daily = daily_with_excess({2023: np.linspace(-.002, .003, 60), 2024: np.linspace(-.001, .004, 60)})
        reference = daily_with_excess({2023: np.linspace(-.003, .002, 60), 2024: np.linspace(-.002, .003, 60)})
        with self.assertRaises(ValueError):
            executable_dsr(daily, n_trials=12, trial_sharpes=[])
        result = executable_dsr(daily, n_trials=12,
                                trial_sharpes=[float(daily.excess.mean() / daily.excess.std(ddof=1)),
                                               float(reference.excess.mean() / reference.excess.std(ddof=1))])
        self.assertEqual(result["n_trials"], 12)
        self.assertEqual(result["trial_sr_count"], 2)
        self.assertEqual(result["trial_sr_coverage"], 2 / 12)
        self.assertFalse(result["single_series_variance_fallback"])

    def test_dsr_self_contained_matches_frozen_reference(self):
        x = np.linspace(-.002, .003, 120)
        daily = pd.DataFrame({"excess": x})
        result = executable_dsr(daily, 12, [float(x.mean() / x.std(ddof=1)), .01, .02])
        self.assertAlmostEqual(result["observed_sharpe"], 0.3421009468727611, places=14)
        self.assertAlmostEqual(result["deflated_benchmark_sr0"], 0.31451283378831985, places=14)
        self.assertAlmostEqual(result["psr_vs_zero"], 0.9998874164606869, places=14)
        self.assertAlmostEqual(result["deflated_sharpe_ratio"], 0.6169537876609323, places=14)


if __name__ == "__main__":
    unittest.main()
