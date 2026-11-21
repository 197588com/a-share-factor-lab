# -*- coding: utf-8 -*-
"""P27 历史成分与行情缓存。只写独立目录，不读取／修改生产数据库。

BaoStock 0.9.4 隔离在 .deps，避免覆盖项目 numpy/pandas。
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS / ".deps"))
ROOT = WS / "data" / "p27_universe_expansion"
SNAP = ROOT / "constituents" / "csi500"
QFQ = ROOT / "csv"
RAW = ROOT / "raw_daily"
META = ROOT / "meta"
for p in (SNAP, QFQ, RAW, META):
    p.mkdir(parents=True, exist_ok=True)
START = "2015-01-01"
RAW_START = "2022-12-01"
END = "2026-06-30"


def calendar():
    d = pd.read_csv(WS / "data" / "csv" / "sh000300.csv", usecols=["date"])
    x = pd.to_datetime(d["date"])
    return x[(x >= "2015-01-01") & (x <= END)].dt.strftime("%Y-%m-%d").tolist()


def rows_of(rs):
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
    return pd.DataFrame(rows, columns=rs.fields)


def qlib_code(code):
    market, number = code.split(".")
    return f"{market.lower()}{number}"


def bs_code(instrument):
    instrument = instrument.lower()
    return f"{instrument[:2]}.{instrument[2:]}"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_to_parquet(frame, path):
    """同目录临时文件 + 原子替换；中断不会截断既有有效缓存。"""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_to_csv(frame, path, **kwargs):
    """CSV 与 Parquet 使用同一原子缓存契约。"""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(tmp, index=False, **kwargs)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def fetch_constituents(bs, retry=False, shard=0, shards=1):
    dates = [d for d in calendar() if d >= "2015-12-01"][shard::shards]
    failures = {}
    for i, d in enumerate(dates, 1):
        out = SNAP / f"{d}.parquet"
        if out.exists() and not retry:
            continue
        try:
            frame = rows_of(bs.query_zz500_stocks(date=d))
            if len(frame) != 500:
                raise ValueError(f"成分数 {len(frame)} != 500")
            frame = frame.rename(columns={"updateDate": "update_date", "code": "bs_code", "code_name": "name"})
            frame["date"] = d
            frame["instrument"] = frame["bs_code"].map(qlib_code).str.upper()
            frame["source"] = "BaoStock query_zz500_stocks"
            frame.to_parquet(out, index=False)
        except Exception as exc:
            failures[d] = str(exc)
        if i % 50 == 0 or i == len(dates):
            print(f"constituents shard={shard} {i}/{len(dates)} failures={len(failures)}", flush=True)
        time.sleep(0.015)
    manifest = {
        "requested": len(dates), "files": len(list(SNAP.glob("*.parquet"))),
        "failures": failures, "start": dates[0], "end": dates[-1], "shard": shard, "shards": shards,
    }
    (META / f"constituent_fetch_manifest_{shard}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"历史成分缺口 {len(failures)} 日；可用 --retry 重试")
    expected = len([d for d in calendar() if d >= "2015-12-01"])
    if len(list(SNAP.glob("*.parquet"))) == expected:
        build_membership()


def assemble_membership(snapshots, trading_dates):
    """缺日只沿用此前最近快照；首日之前绝不向过去倒填未来成分。"""
    snapshots = {pd.Timestamp(d): f for d, f in snapshots.items()}
    previous = None
    frames = []
    for date in sorted(pd.to_datetime(trading_dates)):
        if date in snapshots:
            previous = (date, snapshots[date])
        if previous is None:
            continue
        snapshot_date, source = previous
        frame = source.copy()
        frame["date"] = date
        frame["snapshot_date"] = snapshot_date
        frame["is_imputed"] = bool(date != snapshot_date)
        frames.append(frame)
    if not frames:
        raise ValueError("交易日之前没有可用历史成分快照")
    return pd.concat(frames, ignore_index=True)


def build_membership():
    files = sorted(SNAP.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("无历史成分快照")
    snapshots = {pd.Timestamp(p.stem): pd.read_parquet(p) for p in files}
    trading_dates = pd.to_datetime([d for d in calendar() if d >= "2015-12-01"])
    frame = assemble_membership(snapshots, trading_dates)
    frame = frame.sort_values(["date", "instrument"]).drop_duplicates(["date", "instrument"], keep="last")
    frame.to_parquet(META / "csi500_membership_daily.parquet", index=False)
    ranges = frame.groupby("instrument")["date"].agg(["min", "max"]).reset_index()
    ranges["start"] = ranges["min"].dt.strftime("%Y-%m-%d")
    ranges["end"] = ranges["max"].dt.strftime("%Y-%m-%d")
    ranges[["instrument", "start", "end"]].to_csv(
        META / "csi500_fetch_ranges.txt", sep="\t", header=False, index=False)
    imputed = frame[frame["is_imputed"]]
    summary = {
        "rows": len(frame), "days": int(frame["date"].nunique()),
        "instruments": int(frame["instrument"].nunique()),
        "count_min": int(frame.groupby("date").size().min()),
        "count_median": float(frame.groupby("date").size().median()),
        "count_max": int(frame.groupby("date").size().max()),
        "source_snapshot_days": int(len(files)),
        "imputed_days": int(imputed["date"].nunique()),
        "max_staleness_calendar_days": int((frame["date"] - frame["snapshot_date"]).dt.days.max()),
        "forward_fill_rule": "previous_snapshot_only_no_backward_fill",
        "start": str(frame["date"].min().date()), "end": str(frame["date"].max().date()),
        "sha256": sha(META / "csi500_membership_daily.parquet"),
    }
    (META / "constituent_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def fetch_price_one(bs, instrument, kind="qfq"):
    is_qfq = kind == "qfq"
    out_dir = QFQ if is_qfq else RAW
    out = out_dir / (f"{instrument.lower()}.csv" if is_qfq else f"{instrument.lower()}.parquet")
    if out.exists() and out.stat().st_size > 100:
        return "cached"
    fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,tradestatus,isST"
    rs = bs.query_history_k_data_plus(
        bs_code(instrument), fields,
        start_date=START if is_qfq else RAW_START,
        end_date=END, frequency="d", adjustflag="2" if is_qfq else "3")
    d = rows_of(rs)
    if d.empty:
        raise ValueError("empty price history")
    numeric = ["open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus", "isST"]
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date", keep="last")
    d["instrument"] = instrument.upper()
    if is_qfq:
        # 对齐现有 Tushare/Qlib CSV 单位：volume=手，amount=千元。
        out_df = pd.DataFrame({
            "date": d["date"].dt.strftime("%Y-%m-%d"),
            "open": d["open"], "high": d["high"], "low": d["low"], "close": d["close"],
            "volume": d["volume"] / 100.0, "amount": d["amount"] / 1000.0,
            "factor": 1.0,
        })
        atomic_to_csv(out_df, out, float_format="%.8f")
    else:
        d["up_limit"] = inferred_limit(d, True)
        d["down_limit"] = inferred_limit(d, False)
        d["limit_source"] = "baostock_preclose_board_st_decimal_half_up_0.01"
        atomic_to_parquet(d, out)
    return f"rows={len(d)}"


def inferred_limit(d, up):
    code = d["instrument"].str[2:]
    date = d["date"]
    ratio = pd.Series(0.10, index=d.index)
    ratio.loc[d["isST"].eq(1)] = 0.05
    star = code.str.startswith("688")
    gem20 = code.str.startswith("30") & date.ge(pd.Timestamp("2020-08-24"))
    ratio.loc[star | gem20] = 0.20
    raw = d["preclose"] * (1 + ratio if up else 1 - ratio)
    # 交易所最小报价单位 0.01 元，十进制 half-up；避免 numpy/pandas 银行家舍入。
    return np.floor(raw * 100 + 0.5 + 1e-9) / 100


def refresh_limits(raw_dir=RAW):
    """仅刷新旧缓存的推断涨跌停，保留原始 OHLC 与公司行动字段。"""
    paths = sorted(Path(raw_dir).glob("*.parquet"))
    for path in paths:
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        frame["up_limit"] = inferred_limit(frame, True)
        frame["down_limit"] = inferred_limit(frame, False)
        frame["limit_source"] = "baostock_preclose_board_st_decimal_half_up_0.01"
        atomic_to_parquet(frame, path)
    return len(paths)


def fetch_prices(bs, limit=0, kind="both", only="", shard=0, shards=1):
    member = META / "csi500_membership_daily.parquet"
    if not member.exists():
        raise FileNotFoundError("先运行 constituents")
    m = pd.read_parquet(member, columns=["instrument"])
    instruments = sorted(m["instrument"].unique())
    # 指数也进入独立 provider，供基准和跨源核验；不属于股票宇宙。
    instruments = sorted(set(instruments) | {"SH000300", "SH000905"})
    if only:
        wanted = {x.strip().upper() for x in only.split(",") if x.strip()}
        instruments = [x for x in instruments if x in wanted]
    if limit:
        instruments = instruments[:limit]
    instruments = instruments[shard::shards]
    failures = {}
    for i, instrument in enumerate(instruments, 1):
        for k in (("qfq", "raw") if kind == "both" else (kind,)):
            try:
                msg = fetch_price_one(bs, instrument, k)
                print(f"[{i}/{len(instruments)}] shard={shard} {instrument} {k} {msg}", flush=True)
            except Exception as exc:
                failures[f"{instrument}:{k}"] = str(exc)
                print(f"FAIL {instrument} {k}: {exc}", flush=True)
            time.sleep(0.015)
    manifest = {"requested": len(instruments), "kind": kind, "failures": failures,
                "shard": shard, "shards": shards,
                "qfq_files": len(list(QFQ.glob("*.csv"))), "raw_files": len(list(RAW.glob("*.parquet")))}
    (META / f"price_fetch_manifest_{kind}_{shard}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"行情下载失败 {len(failures)} 项；重跑同一命令可断点续传")


def classify_ohlc_rows(frame, raw=None):
    """完整空 OHLC  only 在原始行情证实停牌时解释；其余缺失均为异常。"""
    missing = frame[["open", "high", "low", "close"]].isna()
    partial = missing.any(axis=1) & ~missing.all(axis=1)
    full = missing.all(axis=1)
    suspended = pd.Series(False, index=frame.index)
    if raw is not None and not raw.empty and full.any():
        status = raw.drop_duplicates("date", keep="last").set_index("date")["tradestatus"]
        dates = pd.to_datetime(frame["date"])
        is_suspended = dates.map(status).eq(0)
        no_volume = frame["volume"].isna() | frame["volume"].eq(0)
        suspended = full & is_suspended & no_volume
    bad = partial | (full & ~suspended)
    return int(bad.sum()), int(suspended.sum())


def health_checks(counts, prices, missing_qfq, missing_raw, expected_days,
                  source_snapshot_days=None):
    """只检查可观察数据质量；补日不能冒充真实快照，上市早晚本身不是缺陷。"""
    if source_snapshot_days is None:
        source_snapshot_days = len(counts)
    return {
        "constituent_days_ge_98pct": bool(source_snapshot_days >= 0.98 * expected_days),
        "all_snapshots_500": bool(counts.eq(500).all()),
        "no_qfq_missing": not missing_qfq,
        "no_raw_missing": not missing_raw,
        "no_bad_ohlc": bool(prices["bad_ohlc"].sum() == 0),
        "no_negative_volume": bool(prices["negative_volume"].sum() == 0),
    }


def execution_instruments(membership, start="2023-01-03", end=END):
    """原始执行行情只要求覆盖正式执行期成分；此前已退市股票不伪报缺失。"""
    dates = pd.to_datetime(membership["date"])
    formal = membership[dates.between(pd.Timestamp(start), pd.Timestamp(end))]
    return sorted(formal["instrument"].str.upper().unique())


def audit():
    # 四个采集分片完成后由单一进程重建，避免并发写成分总表。
    build_membership()
    refreshed_limits = refresh_limits()
    m = pd.read_parquet(META / "csi500_membership_daily.parquet")
    counts = m.groupby("date").size()
    unique = sorted(m["instrument"].unique())
    required_raw = execution_instruments(m)
    missing_qfq = [x for x in unique if not (QFQ / f"{x.lower()}.csv").exists()]
    missing_raw = [x for x in required_raw if not (RAW / f"{x.lower()}.parquet").exists()]
    coverage = []
    for instrument in unique:
        p = QFQ / f"{instrument.lower()}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, usecols=["date", "open", "high", "low", "close", "volume"])
        raw_path = RAW / f"{instrument.lower()}.parquet"
        raw = pd.read_parquet(raw_path, columns=["date", "tradestatus"]) if (
            raw_path.exists() and d[["open", "high", "low", "close"]].isna().any(axis=None)) else None
        bad_ohlc, suspended_ohlc = classify_ohlc_rows(d, raw)
        coverage.append({"instrument": instrument, "rows": len(d),
                         "start": d["date"].min(), "end": d["date"].max(),
                         "bad_ohlc": bad_ohlc, "suspended_ohlc": suspended_ohlc,
                         "negative_volume": int((d["volume"] < 0).sum())})
    c = pd.DataFrame(coverage)
    c.to_csv(META / "price_coverage_by_instrument.csv", index=False)
    expected_days = len([x for x in calendar() if x >= "2015-12-01"])
    source_snapshot_days = len(list(SNAP.glob("*.parquet")))
    checks = health_checks(counts, c, missing_qfq, missing_raw, expected_days,
                           source_snapshot_days=source_snapshot_days)
    out = {"checks": checks, "passed": bool(all(checks.values())),
           "constituent_days": int(len(counts)), "source_snapshot_days": source_snapshot_days,
           "expected_snapshot_days": expected_days,
           "source_snapshot_coverage": float(source_snapshot_days / expected_days),
           "instruments": len(unique), "formal_execution_instruments": len(required_raw),
           "refreshed_limit_files": refreshed_limits,
           "missing_qfq": missing_qfq, "missing_raw": missing_raw,
           "price_rows": int(c["rows"].sum()) if len(c) else 0}
    (META / "data_health.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    if not out["passed"]:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["constituents", "prices", "audit"])
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kind", choices=["qfq", "raw", "both"], default="both")
    ap.add_argument("--only", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("shard 必须满足 0 <= shard < shards")
    if args.cmd == "audit":
        audit()
        return
    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        if args.cmd == "constituents":
            fetch_constituents(bs, args.retry, args.shard, args.shards)
        else:
            fetch_prices(bs, args.limit, args.kind, args.only, args.shard, args.shards)
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
