# -*- coding: utf-8 -*-
"""P27 BaoStock 公司行动缓存，按评分涉及标的断点续传。"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS / ".deps"))
ROOT = WS / "data" / "p27_universe_expansion"
OUT = ROOT / "actions"
OUT.mkdir(parents=True, exist_ok=True)


def rows_of(rs):
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
    return pd.DataFrame(rows, columns=rs.fields)


def action_frame(records, instrument):
    if not records:
        return pd.DataFrame(columns=["instrument", "ex_date", "pay_date", "cash_div", "cash_div_tax",
                                     "stk_div", "stk_bo_rate", "stk_co_rate", "cash_source"])
    d = pd.concat(records, ignore_index=True)
    for c in ["dividCashPsBeforeTax", "dividStocksPs", "dividReserveToStockPs"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["ex_date"] = pd.to_datetime(d["dividOperateDate"], errors="coerce")
    d["pay_date"] = pd.to_datetime(d["dividPayDate"], errors="coerce")
    d = d[d["ex_date"].notna() & d["ex_date"].le(pd.Timestamp("2026-06-30"))].copy()
    if d.empty:
        return pd.DataFrame(columns=["instrument", "ex_date", "pay_date", "cash_div", "cash_div_tax",
                                     "stk_div", "stk_bo_rate", "stk_co_rate", "cash_source"])
    # BaoStock 将送股与转增分列，经济总股数比例为两者之和。
    d["cash_div"] = d["dividCashPsBeforeTax"]
    d["cash_div_tax"] = d["dividCashPsBeforeTax"]
    d["stk_div"] = d["dividStocksPs"] + d["dividReserveToStockPs"]
    d["stk_bo_rate"] = d["dividStocksPs"]
    d["stk_co_rate"] = d["dividReserveToStockPs"]
    # 与 P25 对齐：纯送转没有现金应收，支付日保留 NaT；有现金而缺支付日才回退除权日。
    d.loc[d["cash_div"].gt(0) & d["pay_date"].isna(), "pay_date"] = d["ex_date"]
    d.loc[d["cash_div"].eq(0) & d["stk_div"].gt(0), "pay_date"] = pd.NaT
    d["instrument"] = instrument
    d["cash_source"] = "baostock_implementation"
    d["raw_cash_sum"] = d["cash_div"]
    d["n_raw_rows"] = 1
    d["n_plans"] = 1
    # 同一除权日若有多条计划，现金和股票比例相加；支付日取最晚并披露。
    grouped = []
    for (code, ex), g in d.groupby(["instrument", "ex_date"], sort=True):
        row = g.iloc[-1].copy()
        for c in ["cash_div", "cash_div_tax", "stk_div", "stk_bo_rate", "stk_co_rate", "raw_cash_sum"]:
            row[c] = float(g[c].sum())
        row["pay_date"] = g["pay_date"].max()
        row["n_raw_rows"] = int(len(g))
        row["n_plans"] = int(len(g))
        row["n_pay_dates"] = int(g["pay_date"].nunique())
        grouped.append(row)
    return pd.DataFrame(grouped).sort_values("ex_date").reset_index(drop=True)


def exchange_reference(raw, ex_date, stock=0.0):
    """从原始行情独立复原除权参考价隐含现金；缺参考时明确返回 None。"""
    if raw is None or raw.empty:
        return None
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    ex = pd.Timestamp(ex_date)
    if ex not in frame.index:
        return None
    loc = frame.index.get_loc(ex)
    if not isinstance(loc, (int, np.integer)) or loc <= 0:
        return None
    prev_close = frame.iloc[loc - 1].get("close")
    preclose = frame.iloc[loc].get("preclose")
    if not (pd.notna(prev_close) and pd.notna(preclose) and float(prev_close) > 0 and float(preclose) > 0):
        return None
    implied = float(prev_close) - float(preclose) * (1.0 + float(stock))
    return {"previous_close": float(prev_close), "ex_reference_preclose": float(preclose),
            "reference_implied_cash": implied}


def reconcile_with_raw(actions, raw):
    """纯现金可用参考昨收补记；送转只信实施明细，参考价差额仅作诊断。"""
    out = actions.copy()
    for col in ("reference_implied_cash", "cash_reconciliation_gap"):
        if col not in out:
            out[col] = np.nan
    if "pay_date" not in out:
        out["pay_date"] = pd.NaT
    out["pay_date"] = pd.to_datetime(out["pay_date"], errors="coerce")
    if out.empty:
        return out
    for idx, row in out.iterrows():
        ex = pd.Timestamp(row["ex_date"])
        cash = float(row.get("cash_div", 0.0)) if pd.notna(row.get("cash_div")) else 0.0
        stock = float(row.get("stk_div", 0.0)) if pd.notna(row.get("stk_div")) else 0.0
        if cash > 0 and pd.isna(out.at[idx, "pay_date"]):
            out.loc[idx, "pay_date"] = ex
        if cash == 0 and stock > 0:
            out.loc[idx, "pay_date"] = pd.NaT
        out.loc[idx, ["reference_implied_cash", "cash_reconciliation_gap"]] = np.nan
        reference = exchange_reference(raw, ex, stock)
        if reference is None:
            out.loc[idx, "cash_source"] = "baostock_unverified_missing_reference"
            continue
        implied = reference["reference_implied_cash"]
        out.loc[idx, "reference_implied_cash"] = implied
        if stock > 0:
            # 与 P25 implementation_stock_plan 完全一致：不按隐含现金修改实施股数或现金。
            out.loc[idx, "cash_source"] = "baostock_implementation_stock_plan"
        elif abs(cash - implied) <= 0.011:
            out.loc[idx, "cash_source"] = "baostock_implementation"
        elif implied >= 0 and implied - cash > 0.011:
            cash = implied
            out.loc[idx, "cash_div"] = cash
            out.loc[idx, "cash_div_tax"] = cash
            out.loc[idx, "cash_source"] = "exchange_reference_imputed"
            if pd.isna(out.at[idx, "pay_date"]):
                out.loc[idx, "pay_date"] = ex
        elif implied >= 0:
            out.loc[idx, "cash_source"] = "baostock_implementation_reference_conflict"
        else:
            out.loc[idx, "cash_source"] = "baostock_implementation_negative_reference"
        out.loc[idx, "cash_reconciliation_gap"] = cash - implied
    return out


def fetch_one(bs, instrument, years, refresh=False):
    out = OUT / f"{instrument.lower()}.parquet"
    if out.exists() and not refresh:
        return "cached"
    records = []
    for year in years:
        rs = bs.query_dividend_data(code=f"{instrument[:2].lower()}.{instrument[2:]}",
                                    year=str(year), yearType="report")
        records.append(rows_of(rs))
        time.sleep(0.015)
    frame = action_frame(records, instrument)
    raw_path = ROOT / "raw_daily" / f"{instrument.lower()}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"缺公司行动交叉核验原始行情 {raw_path}")
    raw = pd.read_parquet(raw_path, columns=["date", "close", "preclose"])
    frame = reconcile_with_raw(frame, raw)
    from p27_data import atomic_to_parquet
    atomic_to_parquet(frame, out)
    return f"actions={len(frame)}"


def replay_cached_actions():
    """无网络重放全部缓存；用于规则升级后清除陈旧来源和参考价诊断。"""
    from p27_data import atomic_to_parquet
    counts = {}
    files = sorted(OUT.glob("*.parquet"))
    events = 0
    for path in files:
        raw_path = ROOT / "raw_daily" / path.name
        if not raw_path.exists():
            raise FileNotFoundError(f"缺公司行动重放原始行情 {raw_path}")
        actions = pd.read_parquet(path)
        raw = pd.read_parquet(raw_path, columns=["date", "close", "preclose"])
        frame = reconcile_with_raw(actions, raw)
        atomic_to_parquet(frame, path)
        events += len(frame)
        if "cash_source" in frame:
            for source, count in frame["cash_source"].value_counts().items():
                counts[str(source)] = counts.get(str(source), 0) + int(count)
    result = {"passed": True, "files": len(files), "events": events, "cash_sources": counts,
              "network_used": False, "atomic_replay": True}
    (ROOT / "meta" / "actions_reconcile_replay.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruments", default="")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--reconcile-only", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("shard 必须满足 0 <= shard < shards")
    if args.reconcile_only:
        replay_cached_actions()
        return
    if args.instruments:
        instruments = [x.strip().upper() for x in args.instruments.split(",") if x.strip()]
    else:
        files = sorted((ROOT / "scores").glob("csi500_alstm_fold*.parquet"))
        if not files:
            raise FileNotFoundError("尚无 P27 评分，无法确定公司行动下载范围")
        scores = pd.concat([pd.read_parquet(p, columns=["instrument"]) for p in files], ignore_index=True)
        instruments = sorted(scores["instrument"].str.upper().unique())
    instruments = instruments[args.shard::args.shards]
    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    failures = {}
    try:
        for i, instrument in enumerate(instruments, 1):
            for attempt in range(3):
                try:
                    # 2023 年实施的分红通常属于 2022 报告期，向前多取两年再按除权日过滤。
                    msg = fetch_one(bs, instrument, range(2021, 2027), args.refresh)
                    print(f"[{i}/{len(instruments)}] {instrument} {msg}", flush=True)
                    break
                except Exception as exc:
                    if attempt == 2:
                        failures[instrument] = str(exc)
                        print(f"FAIL {instrument}: {exc}", flush=True)
                    else:
                        time.sleep(2 ** attempt)
    finally:
        bs.logout()
    manifest = {"instruments": len(instruments), "failures": failures,
                "files": len(list(OUT.glob("*.parquet"))),
                "shard": args.shard, "shards": args.shards}
    manifest_name = "actions_manifest.json" if args.shards == 1 else f"actions_manifest_{args.shard}_of_{args.shards}.json"
    (ROOT / "meta" / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"公司行动失败 {len(failures)} 项")


if __name__ == "__main__":
    main()
