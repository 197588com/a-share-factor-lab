# -*- coding: utf-8 -*-
"""P25 冻结信号的独立成本压力回放；原费用先做逐字段复现，不覆盖 P25。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import execution_replay as er
import shadow_execution as se

WS = Path(__file__).resolve().parent
P25 = WS / "data" / "backtest" / "execution_replay"
OUT = WS / "data" / "backtest" / "execution_replay_cost_x2"


def run(stress=1.0):
    scores = {fold: pd.read_parquet(WS / "data" / "scores" / f"alstm_fold{fold}.parquet")
              for fold in range(8, 12)}
    instruments = sorted(set().union(*[set(s["instrument"].str.upper()) for s in scores.values()]))
    store = er.RawStore(instruments)
    originals = {k: er.PARAMS[k] for k in ("slip", "comm", "stamp", "transfer")}
    for key in originals:
        er.PARAMS[key] = originals[key] * stress
    daily_all, orders_all, fold_rows = [], [], []
    try:
        for fold in range(8, 12):
            baseline = pd.read_parquet(P25 / f"daily_fold{fold}.parquet")
            # 直接读冻结快照，保留同分记录顺序；压力实验只改变成本参数。
            score = pd.read_parquet(P25 / f"mask_kept_qlib_fold{fold}.parquet")
            events = pd.read_parquet(P25 / f"mask_events_qlib_fold{fold}.parquet")
            daily, orders, actions = se.run_account(score, baseline, store)
            health = se.health(daily, orders, len(baseline))
            if not health["passed"]:
                raise AssertionError(health)
            if stress == 1.0:
                frozen = pd.read_parquet(P25 / f"daily_fold{fold}.parquet")
                for col in ("cash", "mkt_value", "receivables", "nav", "strategy_return", "bench", "excess",
                            "fee", "slippage", "blocked", "held", "skipped"):
                    if not np.allclose(daily[col], frozen[col], rtol=0, atol=1e-8):
                        raise AssertionError(f"P25 fold{fold} 原费用复现不一致：{col}")
                frozen_orders = pd.read_parquet(P25 / f"orders_fold{fold}.parquet")
                if len(orders) != len(frozen_orders):
                    raise AssertionError(f"P25 fold{fold} 订单数复现不一致")
                for col in ("exec_date", "signal_date", "code", "side", "status"):
                    if not orders[col].equals(frozen_orders[col]):
                        raise AssertionError(f"P25 fold{fold} 订单序列复现不一致：{col}")
                for col in ("qty", "px", "value", "fee"):
                    if not np.allclose(orders[col], frozen_orders[col], rtol=0, atol=1e-8):
                        raise AssertionError(f"P25 fold{fold} 订单数值复现不一致：{col}")
            daily["fold"] = fold
            daily_all.append(daily)
            orders_all.append(orders)
            fold_rows.append({"fold": fold, **se.metric(daily), **se.order_stats(orders)})
            if stress != 1.0:
                daily.to_parquet(OUT / f"daily_fold{fold}.parquet", index=False)
                orders.to_parquet(OUT / f"orders_fold{fold}.parquet", index=False)
                events.to_parquet(OUT / f"mask_events_fold{fold}.parquet", index=False)
                actions.to_parquet(OUT / f"actions_fold{fold}.parquet", index=False)
        daily = pd.concat(daily_all, ignore_index=True).sort_values("datetime").reset_index(drop=True)
        orders = pd.concat(orders_all, ignore_index=True)
        result = {"candidate": "p25_frozen_signal_cost_sensitivity", "stress": stress,
                  "overall": se.metric(daily), "folds": fold_rows, "orders": se.order_stats(orders),
                  "params": {k: er.PARAMS[k] for k in originals},
                  "is_new_research_candidate": False, "frozen_p25_overwritten": False}
        if stress != 1.0:
            OUT.mkdir(parents=True, exist_ok=True)
            daily.to_parquet(OUT / "daily_all.parquet", index=False)
            orders.to_parquet(OUT / "orders_all.parquet", index=False)
            (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            pd.DataFrame(fold_rows).to_csv(OUT / "fold_summary.csv", index=False)
        return result
    finally:
        er.PARAMS.update(originals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", type=float, choices=[1.0, 2.0], default=1.0)
    args = parser.parse_args()
    if args.stress != 1.0:
        OUT.mkdir(parents=True, exist_ok=True)
    result = run(args.stress)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
