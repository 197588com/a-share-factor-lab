# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from execution_data import normalize_actions


def daily(prev, pre, close=None):
    return pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        "close": [prev, prev if close is None else close],
        "pre_close": [prev, pre],
    })


def action(cash, stock=0.0, end="20231231", ann="20240101"):
    return {"div_proc": "实施", "ex_date": "20240102", "pay_date": "20240102",
            "cash_div": cash, "cash_div_tax": cash, "stk_div": stock,
            "stk_bo_rate": stock, "stk_co_rate": 0.0, "record_date": "20240101",
            "end_date": end, "ann_date": ann, "imp_ann_date": ann}


class NormalizeActionTest(unittest.TestCase):
    def test_same_day_distinct_cash_plans_are_combined(self):
        out = normalize_actions(pd.DataFrame([action(.113, end="special"), action(.442)]),
                                "SH601898", daily(13.52, 12.97))
        self.assertAlmostEqual(out.iloc[0]["cash_div"], .555)
        self.assertEqual(out.iloc[0]["cash_source"], "implementation_cash_subset")
        self.assertEqual(out.iloc[0]["n_plans"], 2)

    def test_duplicate_amount_is_not_double_counted(self):
        out = normalize_actions(pd.DataFrame([action(1.65, end="a"), action(1.65, end="b", ann="20240102")]),
                                "SZ300760", daily(274.60, 272.95))
        self.assertAlmostEqual(out.iloc[0]["cash_div"], 1.65)
        self.assertEqual(out.iloc[0]["n_plans"], 1)

    def test_missing_plan_is_imputed_from_reference(self):
        out = normalize_actions(pd.DataFrame([action(.113)]), "SH601898", daily(13.52, 12.97))
        self.assertAlmostEqual(out.iloc[0]["cash_div"], .55)
        self.assertEqual(out.iloc[0]["cash_source"], "exchange_reference_imputed")

    def test_higher_implementation_cash_is_not_overwritten(self):
        out = normalize_actions(pd.DataFrame([action(.8533)]), "SH600900", daily(22.0, 21.18))
        self.assertAlmostEqual(out.iloc[0]["cash_div"], .8533)
        self.assertEqual(out.iloc[0]["cash_source"], "implementation_cash_reference_conflict")

    def test_stock_plan_does_not_fabricate_negative_cash(self):
        out = normalize_actions(pd.DataFrame([action(.45, stock=.4)]),
                                "SZ300308", daily(166.96, 119.64))
        self.assertAlmostEqual(out.iloc[0]["cash_div"], .45)
        self.assertAlmostEqual(out.iloc[0]["stk_div"], .4)
        self.assertEqual(out.iloc[0]["cash_source"], "implementation_stock_plan")

    def test_stock_and_special_cash_are_combined_once(self):
        rows = [action(3.07, stock=.5), action(1.23, stock=0.0, end="special")]
        out = normalize_actions(pd.DataFrame(rows), "SH600188", daily(33.87, 19.71))
        self.assertAlmostEqual(out.iloc[0]["cash_div"], 4.30)
        self.assertAlmostEqual(out.iloc[0]["stk_div"], .5)


if __name__ == "__main__":
    unittest.main()
