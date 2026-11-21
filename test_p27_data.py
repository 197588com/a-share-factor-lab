# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from p27_data import health_checks


class P27DataTest(unittest.TestCase):
    def test_late_listed_stock_does_not_fail_history_proxy(self):
        counts = pd.Series(500, index=pd.date_range("2023-01-02", periods=100, freq="B"))
        prices = pd.DataFrame({
            "instrument": ["OLD", "NEW"], "rows": [100, 30],
            "start": ["2015-01-05", "2025-01-02"], "end": ["2026-06-30", "2026-06-30"],
            "bad_ohlc": [0, 0], "negative_volume": [0, 0],
        })
        checks = health_checks(counts, prices, [], [], expected_days=100)
        self.assertTrue(all(checks.values()))
        self.assertNotIn("history_starts_no_later_than_listing", checks)

    def test_imputed_membership_days_do_not_count_as_source_snapshots(self):
        counts = pd.Series(500, index=pd.date_range("2023-01-02", periods=100, freq="B"))
        prices = pd.DataFrame({"bad_ohlc": [0], "negative_volume": [0]})
        checks = health_checks(counts, prices, [], [], expected_days=100,
                               source_snapshot_days=97)
        self.assertFalse(checks["constituent_days_ge_98pct"])
        checks = health_checks(counts, prices, [], [], expected_days=100,
                               source_snapshot_days=98)
        self.assertTrue(checks["constituent_days_ge_98pct"])


if __name__ == "__main__":
    unittest.main()


class P27MembershipTest(unittest.TestCase):
    def test_missing_day_uses_only_previous_snapshot(self):
        from p27_data import assemble_membership
        early = pd.DataFrame({"instrument": ["SH600000"], "name": ["旧成分"]})
        late = pd.DataFrame({"instrument": ["SZ000001"], "name": ["新成分"]})
        snapshots = {pd.Timestamp("2024-01-02"): early, pd.Timestamp("2024-01-04"): late}
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
        result = assemble_membership(snapshots, dates)
        self.assertNotIn(pd.Timestamp("2024-01-01"), set(result["date"]))
        jan3 = result[result["date"].eq(pd.Timestamp("2024-01-03"))].iloc[0]
        self.assertEqual(jan3["instrument"], "SH600000")
        self.assertEqual(jan3["snapshot_date"], pd.Timestamp("2024-01-02"))
        self.assertTrue(jan3["is_imputed"])
        jan4 = result[result["date"].eq(pd.Timestamp("2024-01-04"))].iloc[0]
        self.assertEqual(jan4["instrument"], "SZ000001")
        self.assertFalse(jan4["is_imputed"])


class P27LimitPriceTest(unittest.TestCase):
    def test_limit_price_uses_exchange_half_up_rounding(self):
        from p27_data import inferred_limit
        frame = pd.DataFrame({
            "instrument": ["SH600000"], "date": [pd.Timestamp("2024-01-03")],
            "isST": [0], "preclose": [10.05],
        })
        self.assertEqual(float(inferred_limit(frame, True).iloc[0]), 11.06)
        self.assertEqual(float(inferred_limit(frame, False).iloc[0]), 9.05)


class P27Gem302LimitTest(unittest.TestCase):
    def test_302_prefix_uses_twenty_percent_after_gem_reform(self):
        from p27_data import inferred_limit
        frame = pd.DataFrame({
            "instrument": ["SZ302132", "SZ302132"],
            "date": pd.to_datetime(["2020-08-21", "2020-08-24"]),
            "isST": [0, 0], "preclose": [100.0, 100.0],
        })
        self.assertEqual(float(inferred_limit(frame, True).iloc[0]), 110.0)
        self.assertEqual(float(inferred_limit(frame, True).iloc[1]), 120.0)
        self.assertEqual(float(inferred_limit(frame, False).iloc[1]), 80.0)


class P27NormalPriceRoundingTest(unittest.TestCase):
    def test_normal_price_half_up_differs_from_bankers_rounding(self):
        from p27_data import inferred_limit
        frame = pd.DataFrame({
            "instrument": ["SH600000"], "date": [pd.Timestamp("2024-01-03")],
            "isST": [0], "preclose": [1.15],
        })
        self.assertEqual(float(inferred_limit(frame, True).iloc[0]), 1.27)


class P27LimitCacheRefreshTest(unittest.TestCase):
    def test_refresh_replaces_legacy_bankers_rounding(self):
        import tempfile
        from pathlib import Path
        from p27_data import refresh_limits
        frame = pd.DataFrame({
            "instrument": ["SH600000"], "date": [pd.Timestamp("2023-02-07")],
            "isST": [0], "preclose": [7.25], "up_limit": [7.98], "down_limit": [6.52],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sh600000.parquet"
            frame.to_parquet(path, index=False)
            count = refresh_limits(Path(tmp))
            result = pd.read_parquet(path)
        self.assertEqual(count, 1)
        self.assertEqual(float(result["down_limit"].iloc[0]), 6.53)


class P27AtomicCacheTest(unittest.TestCase):
    def test_interrupted_parquet_write_preserves_previous_valid_cache(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        from p27_data import atomic_to_parquet
        old = pd.DataFrame({"value": [1]})
        new = pd.DataFrame({"value": [2]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "price.parquet"
            atomic_to_parquet(old, path)
            def interrupted(frame, target, index=False):
                Path(target).write_bytes(b"PAR1-partial")
                raise RuntimeError("interrupted")
            with mock.patch.object(pd.DataFrame, "to_parquet", interrupted):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    atomic_to_parquet(new, path)
            result = pd.read_parquet(path)
            self.assertEqual(result["value"].tolist(), [1])
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_interrupted_csv_write_preserves_previous_valid_cache(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        from p27_data import atomic_to_csv
        old = pd.DataFrame({"value": [1]})
        new = pd.DataFrame({"value": [2]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "price.csv"
            atomic_to_csv(old, path)
            def interrupted(frame, target, index=False, **kwargs):
                Path(target).write_text("partial", encoding="utf-8")
                raise RuntimeError("interrupted")
            with mock.patch.object(pd.DataFrame, "to_csv", interrupted):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    atomic_to_csv(new, path)
            result = pd.read_csv(path)
            self.assertEqual(result["value"].tolist(), [1])
            self.assertFalse(list(Path(tmp).glob("*.tmp")))


class P27OhlcHealthTest(unittest.TestCase):
    def test_only_raw_confirmed_suspension_explains_fully_empty_ohlc(self):
        from p27_data import classify_ohlc_rows
        frame = pd.DataFrame({
            "date": ["2025-02-17", "2025-02-18", "2025-02-19"],
            "open": [None, None, None], "high": [None, None, None],
            "low": [None, 9.0, None], "close": [None, None, None],
            "volume": [None, 0.0, 100.0],
        })
        raw = pd.DataFrame({
            "date": pd.to_datetime(["2025-02-17", "2025-02-18", "2025-02-19"]),
            "tradestatus": [0, 1, 0], "volume": [None, 0.0, None],
        })
        bad, suspended = classify_ohlc_rows(frame, raw)
        self.assertEqual(bad, 2)
        self.assertEqual(suspended, 1)


class P27ExecutionCoverageTest(unittest.TestCase):
    def test_raw_prices_required_only_for_formal_execution_members(self):
        from p27_data import execution_instruments
        membership = pd.DataFrame({
            "date": pd.to_datetime(["2021-09-30", "2023-01-03", "2026-06-30", "2026-07-02"]),
            "instrument": ["SH600068", "SH600000", "SZ000001", "SZ000002"],
        })
        required = execution_instruments(membership)
        self.assertEqual(required, ["SH600000", "SZ000001"])
        self.assertNotIn("SH600068", required)
        self.assertNotIn("SZ000002", required)
