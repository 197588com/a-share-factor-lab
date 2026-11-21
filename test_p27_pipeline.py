# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

import train_models as tm
from p27_pipeline import build_segment_panels


class P27PipelineTest(unittest.TestCase):
    @staticmethod
    def frame(dates, instruments, member=True):
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime(dates), instruments], names=["datetime", "instrument"])
        n = len(idx)
        data = {}
        for j in range(tm.D_FEAT):
            data[("feature", f"F{j}")] = np.arange(n, dtype=np.float32) + j
        data[("label", "LABEL0")] = np.arange(n, dtype=np.float32) / 100
        frame = pd.DataFrame(data, index=idx)
        if member:
            return frame, frame.copy()
        filtered = frame.loc[[idx[0]]].copy()
        return frame, filtered

    def test_pit_nonmember_has_feature_but_no_label(self):
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        raw = {}
        filtered = {}
        for seg in ("train", "valid", "test"):
            raw[seg], filtered[seg] = self.frame(dates, ["SH600000", "SZ000001"], member=False)
        panels, _, meta = build_segment_panels(raw, filtered)
        F, L = panels["train"]
        kept_date = filtered["train"].index[0][0]
        kept_inst = filtered["train"].index[0][1]
        t = meta["segments"]["train"]["dates"].index(kept_date.date())
        i = meta["insts"].index(kept_inst)
        other = 1 - i
        self.assertTrue(np.isfinite(F[t]).all())
        self.assertTrue(np.isfinite(L[t, i]))
        self.assertTrue(np.isnan(L[t, other]))

    def test_test_segment_reuses_exactly_seven_history_days(self):
        raw, filtered = {}, {}
        starts = {"train": "2022-01-03", "valid": "2023-01-02", "test": "2024-01-01"}
        for seg, start in starts.items():
            dates = pd.date_range(start, periods=10, freq="B")
            raw[seg], filtered[seg] = self.frame(dates, ["SH600000"])
        panels, sample_indices, meta = build_segment_panels(raw, filtered)
        test_dates = meta["segments"]["test"]["dates"]
        self.assertEqual(len(test_dates), 17)
        self.assertEqual(test_dates[:7], meta["segments"]["valid"]["dates"][-7:])
        np.testing.assert_array_equal(sample_indices["test"], np.arange(7, 17))
        F, L = panels["test"]
        dataset = tm.WindowDS(F, L, np.zeros((len(test_dates), tm.D_GATE), dtype=np.float32),
                              sample_indices["test"], False)
        self.assertIn((7, 0), dataset.samples)
        x, _ = dataset[dataset.samples.index((7, 0))]
        self.assertEqual(x.shape, (8, tm.D_FEAT))
        self.assertTrue(np.isfinite(x.numpy()).all())


if __name__ == "__main__":
    unittest.main()


class P27EffectiveCoverageTest(unittest.TestCase):
    def test_coverage_requires_complete_eight_day_feature_window(self):
        from p27_pipeline import valid_sample_counts
        F = np.ones((12, 2, 3), dtype=np.float32)
        L = np.ones((12, 2), dtype=np.float32)
        F[3, 1, 0] = np.nan
        counts = valid_sample_counts(F, L, np.arange(7, 12))
        self.assertEqual(counts, [1, 1, 1, 1, 2])
        L[11, 1] = np.nan
        counts = valid_sample_counts(F, L, np.arange(7, 12))
        self.assertEqual(counts, [1, 1, 1, 1, 1])


class P27FormalBoundaryTest(unittest.TestCase):
    def test_last_two_formal_signal_dates_have_no_label(self):
        from p27_pipeline import mask_incomplete_formal_labels
        dates = pd.to_datetime(["2026-06-25", "2026-06-26", "2026-06-29", "2026-06-30"])
        idx = pd.MultiIndex.from_product([dates, ["SH600000"]], names=["datetime", "instrument"])
        frame = pd.DataFrame({("feature", "F0"): 1.0, ("label", "LABEL0"): 0.01}, index=idx)
        masked = mask_incomplete_formal_labels(frame)
        self.assertTrue(np.isfinite(masked.loc[(pd.Timestamp("2026-06-26"), "SH600000"), ("label", "LABEL0")]))
        self.assertTrue(np.isnan(masked.loc[(pd.Timestamp("2026-06-29"), "SH600000"), ("label", "LABEL0")]))
        self.assertTrue(np.isnan(masked.loc[(pd.Timestamp("2026-06-30"), "SH600000"), ("label", "LABEL0")]))
        self.assertTrue(np.isfinite(masked[("feature", "F0")]).all())
