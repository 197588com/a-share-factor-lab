# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from p27_execute import attach_participation


class FakeStore:
    def row(self, code, date):
        return pd.Series({"amount": 1_000_000.0})


class P27ExecuteTest(unittest.TestCase):
    def test_filled_order_participation_uses_raw_daily_amount(self):
        orders = pd.DataFrame({
            "exec_date": ["2024-01-03", "2024-01-03"], "code": ["SH600000", "SH600000"],
            "side": ["BUY", "BUY"], "status": ["FILLED", "BLOCKED"],
            "value": [100_000.0, 0.0],
        })
        enriched, stats = attach_participation(orders, FakeStore())
        self.assertAlmostEqual(enriched.loc[0, "participation"], 0.10)
        self.assertTrue(pd.isna(enriched.loc[1, "participation"]))
        self.assertEqual(stats["filled_with_amount"], 1)
        self.assertAlmostEqual(stats["participation_max"], 0.10)
        self.assertAlmostEqual(stats["participation_over_5pct"], 1.0)


class P27ExecutionDataGateTest(unittest.TestCase):
    def test_gate_requires_complete_actions_and_passed_cross_validation(self):
        import json
        import tempfile
        from pathlib import Path
        from p27_execute import require_execution_data_gates
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("meta", "metrics", "actions"):
                (root / sub).mkdir()
            (root / "meta" / "data_health.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            (root / "meta" / "limit_cross_validation.json").write_text(
                json.dumps({"mismatch_rows": 0}), encoding="utf-8")
            (root / "meta" / "actions_manifest.json").write_text(
                json.dumps({"failures": {}}), encoding="utf-8")
            (root / "metrics" / "coverage_fold8.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8")
            action_path = root / "actions" / "sh600000.parquet"
            pd.DataFrame().to_parquet(action_path, index=False)
            validator = lambda candidate_dir: ({"passed": True}, pd.DataFrame())
            result = require_execution_data_gates(["SH600000"], [8], root=root, action_validator=validator)
            self.assertTrue(result["passed"])
            action_path.unlink()
            with self.assertRaises(AssertionError):
                require_execution_data_gates(["SH600000"], [8], root=root, action_validator=validator)


class P27CorporateActionHealthTest(unittest.TestCase):
    def test_unverified_or_conflicting_action_on_holding_fails_closed(self):
        from p27_execute import corporate_action_health
        actions = pd.DataFrame({
            "code": ["SH600000"], "cash_source": ["baostock_implementation_reference_conflict"],
            "cash_div_per_old_share": [0.3], "cash_reconciliation_gap": [0.05],
        })
        health = corporate_action_health(actions)
        self.assertFalse(health["passed"])
        self.assertEqual(health["bad_source_events"], 1)

    def test_reference_imputation_is_visible_but_allowed(self):
        from p27_execute import corporate_action_health
        actions = pd.DataFrame({
            "code": ["SH600000"], "cash_source": ["exchange_reference_imputed"],
            "cash_div_per_old_share": [0.3], "cash_reconciliation_gap": [0.0],
        })
        health = corporate_action_health(actions)
        self.assertTrue(health["passed"])
        self.assertEqual(health["imputed_events"], 1)

    def test_stock_plan_gap_is_diagnostic_but_pure_cash_gap_fails(self):
        from p27_execute import corporate_action_health
        actions = pd.DataFrame({
            "code": ["SH688005", "SH600000"],
            "cash_source": ["baostock_implementation_stock_plan", "baostock_implementation"],
            "cash_div_per_old_share": [0.0, 0.2], "cash_reconciliation_gap": [0.25, 0.0],
        })
        health = corporate_action_health(actions)
        self.assertTrue(health["passed"])
        self.assertEqual(health["stock_plan_events"], 1)
        actions.loc[1, "cash_reconciliation_gap"] = 0.05
        self.assertFalse(corporate_action_health(actions)["passed"])


if __name__ == "__main__":
    unittest.main()
