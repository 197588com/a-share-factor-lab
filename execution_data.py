# -*- coding: utf-8 -*-
"""P25 原始执行行情与公司行动缓存；只写工作区，不触碰生产数据库。"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts

from token_loader import tushare_token

WS = Path(__file__).resolve().parent
OUT = WS / "data" / "backtest" / "execution_replay"
RAW = OUT / "raw"
START = "20230101"
END = "20260630"


def ts_code(instrument: str) -> str:
    instrument = instrument.upper()
    return f"{instrument[2:]}.{instrument[:2]}"


def _atomic_parquet(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _best_cash_subset(values, target):
    """在全部实施公告现金金额中找最接近参考昨收隐含总额的子集。"""
    states = {0.0: ()}
    for i, value in enumerate(values):
        value = round(max(float(value), 0.0), 8)
        for subtotal, path in list(states.items()):
            new = round(subtotal + value, 8)
            states.setdefault(new, path + (i,))
    best = min(states, key=lambda x: abs(x - target))
    return best, states[best]


def normalize_actions(actions: pd.DataFrame, instrument: str, daily: pd.DataFrame):
    """现金实施记录与除权参考昨收交叉复原；送转数量只信实施明细。"""
    if actions is None or actions.empty or "div_proc" not in actions or "ex_date" not in actions:
        return pd.DataFrame()
    actions = actions[actions["div_proc"].eq("实施")].copy()
    actions = actions[actions["ex_date"].notna()].copy()
    if actions.empty:
        return pd.DataFrame()
    actions["instrument"] = instrument.upper()
    actions["ex_date"] = pd.to_datetime(actions["ex_date"], format="%Y%m%d", errors="coerce")
    actions["pay_date"] = pd.to_datetime(actions["pay_date"], format="%Y%m%d", errors="coerce")
    actions = actions[actions["ex_date"] <= pd.Timestamp(END)]
    if actions.empty:
        return pd.DataFrame()
    actions = actions.sort_values(["ex_date", "imp_ann_date", "ann_date"])
    daily_idx = daily.copy()
    daily_idx["date"] = pd.to_datetime(daily_idx["date"])
    daily_idx = daily_idx.set_index("date").sort_index()
    records = []
    for (_, ex_date), g in actions.groupby(["instrument", "ex_date"], sort=True):
        r = g.iloc[-1].copy()
        for col in ["cash_div", "cash_div_tax", "stk_div", "stk_bo_rate", "stk_co_rate"]:
            g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
        has_stock = bool(g["stk_div"].gt(0).any())
        implied = np.nan
        if ex_date in daily_idx.index:
            i = daily_idx.index.get_loc(ex_date)
            if isinstance(i, (int, np.integer)) and i > 0:
                prev_close = float(daily_idx.iloc[i - 1]["close"])
                pre_close_ex = float(daily_idx.iloc[i]["pre_close"])
                if has_stock:
                    # 先由实施明细确定送转，再仅把差额作为交叉校验，不反推修改现金。
                    econ_cols = [c for c in ["cash_div", "cash_div_tax", "stk_div", "stk_bo_rate",
                                             "stk_co_rate", "record_date", "ex_date", "pay_date"] if c in g]
                    selected = g.drop_duplicates(econ_cols, keep="last")
                    cash = float(selected["cash_div"].sum())
                    stock = float(selected["stk_div"].sum())
                    implied = prev_close - pre_close_ex * (1 + stock)
                    source = "implementation_stock_plan"
                else:
                    stock = 0.0
                    implied = prev_close - pre_close_ex
                    cash, chosen = _best_cash_subset(g["cash_div"].tolist(), implied)
                    if abs(cash - implied) <= 0.011:
                        source = "implementation_cash_subset"
                    elif implied - cash > 0.011:
                        # 只有参考昨收显示“实施明细少了现金”才补缺口。
                        cash = implied
                        source = "exchange_reference_imputed"
                    elif implied >= 0:
                        # 实施金额更高时不覆盖，保留明细并暴露数据商字段冲突。
                        source = "implementation_cash_reference_conflict"
                    else:
                        source = "implementation_fallback_negative_reference"
                for col in ["stk_bo_rate", "stk_co_rate"]:
                    r[col] = float((selected if has_stock else g)[col].sum()) if col in g else 0.0
                r["cash_div"] = cash
                r["cash_div_tax"] = cash
                r["stk_div"] = stock
                r["cash_source"] = source
                r["reference_implied_cash"] = implied
                r["cash_reconciliation_gap"] = cash - implied
                r["raw_cash_sum"] = float(g["cash_div"].sum())
                r["n_raw_rows"] = int(len(g))
                r["n_plans"] = int(len(selected)) if has_stock else int(len(chosen))
                r["pay_date"] = g["pay_date"].max()
                r["n_pay_dates"] = int(g["pay_date"].dropna().nunique())
                r["plan_end_dates"] = "|".join(g["end_date"].astype(str).unique()) if "end_date" in g else ""
                records.append(r)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("ex_date").reset_index(drop=True)


def fetch_one(pro, instrument: str, sleep: float, refresh_actions=False):
    code = ts_code(instrument)
    raw_path = RAW / f"{instrument.lower()}.parquet"
    action_path = RAW / f"{instrument.lower()}_actions.parquet"
    if raw_path.exists():
        daily = pd.read_parquet(raw_path)
    else:
        daily = pro.daily(ts_code=code, start_date=START, end_date=END)
        time.sleep(sleep)
        if daily is None:
            daily = pd.DataFrame()
        limits = pro.stk_limit(ts_code=code, start_date=START, end_date=END)
        time.sleep(sleep)
        if limits is None:
            limits = pd.DataFrame()
        if not daily.empty:
            daily = daily.merge(
                limits[["trade_date", "up_limit", "down_limit"]] if not limits.empty else pd.DataFrame(columns=["trade_date", "up_limit", "down_limit"]),
                on="trade_date", how="left")
    if not daily.empty and "adj_factor" not in daily.columns:
        factors = pro.adj_factor(ts_code=code, start_date=START, end_date=END)
        time.sleep(sleep)
        daily = daily.merge(factors[["trade_date", "adj_factor"]], on="trade_date", how="left")
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["trade_date"], format="%Y%m%d")
        daily["instrument"] = instrument.upper()
        daily = daily.sort_values("date").drop_duplicates("date", keep="last")
    _atomic_parquet(daily, raw_path)

    actions_cached = action_path.exists() and not refresh_actions
    if actions_cached:
        actions = pd.read_parquet(action_path)
    else:
        actions = normalize_actions(pro.dividend(ts_code=code), instrument, daily)
        time.sleep(sleep)
    _atomic_parquet(actions, action_path)
    return f"daily={len(daily)} actions={len(actions)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.12)
    ap.add_argument("--refresh-actions", action="store_true")
    ap.add_argument("--instruments", default="", help="逗号分隔，仅用于定向重试")
    args = ap.parse_args()
    scores = pd.concat([pd.read_parquet(WS / "data" / "scores" / f"alstm_fold{i}.parquet", columns=["instrument"])
                        for i in range(8, 12)], ignore_index=True)
    instruments = sorted(scores["instrument"].str.upper().unique())
    if args.instruments:
        requested = {x.strip().upper() for x in args.instruments.split(",") if x.strip()}
        instruments = [x for x in instruments if x in requested]
    if args.limit:
        instruments = instruments[:args.limit]
    RAW.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api(tushare_token())
    failures = {}
    for i, instrument in enumerate(instruments, 1):
        try:
            result = fetch_one(pro, instrument, args.sleep, refresh_actions=args.refresh_actions)
            print(f"[{i}/{len(instruments)}] {instrument} {result}", flush=True)
        except Exception as exc:
            failures[instrument] = str(exc)
            print(f"[{i}/{len(instruments)}] FAIL {instrument}: {exc}", flush=True)
    manifest = {
        "source": "Tushare daily + stk_limit + adj_factor + dividend(实施，按计划去重并同除权日合并)",
        "start": START, "end": END, "instruments": len(instruments),
        "failures": failures, "actions_refreshed": bool(args.refresh_actions),
        "raw_files": len(list(RAW.glob("*.parquet"))),
    }
    manifest_name = "raw_manifest_retry.json" if args.instruments else "raw_manifest.json"
    (OUT / manifest_name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
