# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class P27ActionCrossValidationTest(unittest.TestCase):
    def test_missing_or_economically_different_reference_event_fails(self):
        from p27_cross_validate import validate_actions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            reference = root / "reference"
            candidate.mkdir(); reference.mkdir()
            pd.DataFrame({
                "instrument": ["SH600000", "SH600000"],
                "ex_date": pd.to_datetime(["2024-07-18", "2021-07-16"]),
                "pay_date": pd.to_datetime(["2024-07-18", "2021-07-16"]),
                "cash_div": [0.32, 0.20], "stk_div": [0.0, 0.0],
            }).to_parquet(candidate / "sh600000.parquet", index=False)
            pd.DataFrame({
                "instrument": ["SH600000", "SH600000"],
                "ex_date": pd.to_datetime(["2024-07-18", "2025-07-16"]),
                "pay_date": pd.to_datetime(["2024-07-18", "2025-07-16"]),
                "cash_div": [0.32, 0.41], "stk_div": [0.0, 0.0],
            }).to_parquet(reference / "sh600000_actions.parquet", index=False)
            summary, detail = validate_actions(candidate, reference)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["missing_in_candidate"], 1)
        self.assertEqual(summary["cash_mismatch_over_0_011"], 0)
        self.assertEqual(len(detail), 2)


    @staticmethod
    def _write_raw(raw_dir, instrument="SH600000", previous=23.69, reference=23.66):
        raw_dir.mkdir(exist_ok=True)
        pd.DataFrame({
            "date": pd.to_datetime(["2024-11-21", "2024-11-22"]),
            "close": [previous, reference], "preclose": [previous, reference],
        }).to_parquet(raw_dir / f"{instrument.lower()}.parquet", index=False)

    def test_right_only_event_is_imputed_and_old_cache_is_retained(self):
        from p27_cross_validate import validate_actions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"; reference = root / "reference"; raw = root / "raw"
            candidate.mkdir(); reference.mkdir()
            pd.DataFrame({
                "instrument": ["SH600482"], "ex_date": pd.to_datetime(["2022-07-01"]),
                "pay_date": pd.to_datetime(["2022-07-01"]), "cash_div": [0.2], "stk_div": [0.0],
            }).to_parquet(candidate / "sh600482.parquet", index=False)
            pd.DataFrame({
                "instrument": ["SH600482"], "ex_date": pd.to_datetime(["2024-11-22"]),
                "pay_date": pd.to_datetime(["2024-11-22"]), "cash_div": [0.03138], "stk_div": [0.0],
            }).to_parquet(reference / "sh600482_actions.parquet", index=False)
            self._write_raw(raw, "SH600482")
            summary, _ = validate_actions(candidate, reference, raw)
            cached = pd.read_parquet(candidate / "sh600482.parquet")
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["missing_in_candidate"], 0)
            self.assertEqual(len(cached), 2)
            repaired = cached[cached["ex_date"].eq(pd.Timestamp("2024-11-22"))].iloc[0]
            self.assertAlmostEqual(repaired["cash_div"], 0.03)
            self.assertEqual(repaired["cash_source"], "exchange_reference_imputed")
            second, _ = validate_actions(candidate, reference, raw)
            self.assertTrue(second["passed"])
            self.assertEqual(len(pd.read_parquet(candidate / "sh600482.parquet")), 2)

    def test_left_only_event_requires_exchange_verification(self):
        from p27_cross_validate import validate_actions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"; reference = root / "reference"; raw = root / "raw"
            candidate.mkdir(); reference.mkdir()
            columns = ["instrument", "ex_date", "pay_date", "cash_div", "stk_div"]
            pd.DataFrame({
                "instrument": ["SH600027"], "ex_date": pd.to_datetime(["2024-11-22"]),
                "pay_date": pd.to_datetime(["2024-11-22"]), "cash_div": [0.15], "stk_div": [0.0],
            }).to_parquet(candidate / "sh600027.parquet", index=False)
            pd.DataFrame(columns=columns).to_parquet(reference / "sh600027_actions.parquet", index=False)
            self._write_raw(raw, "SH600027", previous=6.08, reference=5.93)
            summary, detail = validate_actions(candidate, reference, raw)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["reference_missing_exchange_verified"], 1)
            self.assertEqual(detail.iloc[0]["adjudication"], "reference_missing_exchange_verified")

    def test_pure_stock_nat_pay_dates_match_but_missing_cash_pay_date_fails(self):
        from p27_cross_validate import validate_actions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"; reference = root / "reference"
            candidate.mkdir(); reference.mkdir()
            left = pd.DataFrame({
                "instrument": ["SH688005"], "ex_date": pd.to_datetime(["2025-03-06"]),
                "pay_date": [pd.NaT], "cash_div": [0.0], "stk_div": [0.49],
            })
            right = left.copy()
            left.to_parquet(candidate / "sh688005.parquet", index=False)
            right.to_parquet(reference / "sh688005_actions.parquet", index=False)
            summary, _ = validate_actions(candidate, reference)
            self.assertTrue(summary["passed"])
            left["cash_div"] = 0.2; right["cash_div"] = 0.2
            right["pay_date"] = pd.Timestamp("2025-03-06")
            left.to_parquet(candidate / "sh688005.parquet", index=False)
            right.to_parquet(reference / "sh688005_actions.parquet", index=False)
            summary, _ = validate_actions(candidate, reference)
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["pay_date_mismatch"], 1)

    def test_three_way_cash_conflict_is_not_repaired(self):
        from p27_cross_validate import validate_actions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"; reference = root / "reference"; raw = root / "raw"
            candidate.mkdir(); reference.mkdir()
            left = pd.DataFrame({
                "instrument": ["SH601966"], "ex_date": pd.to_datetime(["2024-11-22"]),
                "pay_date": pd.to_datetime(["2024-11-22"]), "cash_div": [0.07], "stk_div": [0.0],
            })
            right = left.copy(); right["cash_div"] = 0.084
            left.to_parquet(candidate / "sh601966.parquet", index=False)
            right.to_parquet(reference / "sh601966_actions.parquet", index=False)
            self._write_raw(raw, "SH601966", previous=14.97, reference=14.85)
            summary, _ = validate_actions(candidate, reference, raw)
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["cash_mismatch_over_0_011"], 1)
            self.assertEqual(pd.read_parquet(candidate / "sh601966.parquet").iloc[0]["cash_div"], 0.07)


    def test_official_evidence_combines_only_verified_implementation_subset(self):
        import json
        from p27_cross_validate import apply_official_evidence, validate_actions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"; reference = root / "reference"
            candidate.mkdir(); reference.mkdir()
            left = pd.DataFrame({
                "instrument": ["SH601966"], "ex_date": pd.to_datetime(["2025-07-10"]),
                "pay_date": pd.to_datetime(["2025-07-10"]), "cash_div": [0.07], "cash_div_tax": [0.07],
                "stk_div": [0.0], "cash_source": ["baostock_implementation"],
                "reference_implied_cash": [0.08], "cash_reconciliation_gap": [-0.01],
                "raw_cash_sum": [0.07],
            })
            right = left.copy(); right["cash_div"] = 0.084; right["cash_div_tax"] = 0.084
            left.to_parquet(candidate / "sh601966.parquet", index=False)
            right.to_parquet(reference / "sh601966_actions.parquet", index=False)
            evidence = root / "evidence.json"
            evidence.write_text(json.dumps({"schema": "p27_official_action_evidence_v1", "events": [{
                "instrument": "SH601966", "ex_date": "2025-07-10", "pay_date": "2025-07-10",
                "cash_div": 0.084, "stk_div": 0.0, "candidate_components": [0.07],
                "official_components": [0.014, 0.07], "announcement": "实施公告",
                "url": "https://example.com/official", "verbatim": "每股现金红利0.084元",
            }]}), encoding="utf-8")
            audit = apply_official_evidence(candidate, reference, evidence)
            self.assertTrue(audit["passed"])
            repaired = pd.read_parquet(candidate / "sh601966.parquet").iloc[0]
            self.assertEqual(repaired["cash_div"], 0.084)
            self.assertEqual(repaired["raw_cash_sum"], 0.07)
            self.assertAlmostEqual(repaired["cash_reconciliation_gap"], 0.004)
            summary, _ = validate_actions(candidate, reference)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["official_implementation_combined"], 1)


if __name__ == "__main__":
    unittest.main()
