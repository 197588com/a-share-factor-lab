# -*- coding: utf-8 -*-
"""minute_recorder.py — 分钟/集合竞价行情自录入库（DataRecorder 模式）

借鉴 vn.py DataRecorder 的思路：不依赖供应商事后拉取，把属于我们的两项
独立 Tushare 权限（A股历史分钟 stk_mins + 集合竞价 stk_auction）每日落本地
DuckDB，逐步攒出自主可控的高频库。

用法：
  python3 minute_recorder.py daily [--limit N] [--force]   # 最近交易日（默认今日，非交易日则最新交易日）
  python3 minute_recorder.py --date 20261105 [--limit N] [--force]
  python3 minute_recorder.py backfill --start 20250101 --end 20261105   # 历史回补（分批跑）

设计要点：
- 幂等：load_log 记录每只票每天每类数据的成功状态，重复运行自动跳过
- 限速：每股 sleep，尊重 Tushare 单分钟调用上限
- 原子：单票数据先 DELETE 再 INSERT（force 或重跑不产生重复）
- 事件：完成后发布 hf_minute_done，供巡检/台账对账
"""
import os, sys, time, json, argparse, datetime as dt

import duckdb
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
import sys as _sys, os as _os
for _root in {_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}:
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
from token_loader import tushare_token as _tushare_token
TOKEN = _tushare_token()
DB = os.path.join(WS, "data", "hf", "hf.duckdb")
MEMBERS = os.path.join(WS, "data", "csi300_members.csv")
SLEEP = 0.30  # 每股间隔（2 次调用），~110 calls/min，留足限速余量
MAX_RETRY = 3

DDL = """
CREATE TABLE IF NOT EXISTS m1 (
  ts_code VARCHAR, trade_time TIMESTAMP,
  open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
  vol DOUBLE, amount DOUBLE
);
CREATE TABLE IF NOT EXISTS auction (
  ts_code VARCHAR, trade_date DATE,
  vol DOUBLE, price DOUBLE, amount DOUBLE, pre_close DOUBLE,
  turnover_rate DOUBLE, volume_ratio DOUBLE, float_share DOUBLE
);
CREATE TABLE IF NOT EXISTS load_log (
  trade_date DATE, ts_code VARCHAR, kind VARCHAR,
  rows INTEGER, status VARCHAR, loaded_at TIMESTAMP
);
"""


def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = duckdb.connect(DB)
    con.execute(DDL)
    return con


def universe():
    df = pd.read_csv(MEMBERS, dtype=str)
    latest = df["trade_date"].max()
    codes = sorted(df[df["trade_date"] == latest]["con_code"].unique())
    return codes, latest


def latest_trade_day(pro, want=None):
    if want:
        return want
    today = dt.date.today().strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date="20200101", end_date=today)
    opens = cal[cal["is_open"] == 1]["cal_date"]
    return opens.max() if len(opens) else None


def _done(con, trade_date, ts_code, kind):
    r = con.execute(
        "SELECT COUNT(*) FROM load_log WHERE trade_date=? AND ts_code=? AND kind=? AND status='ok'",
        [pd.Timestamp(trade_date), ts_code, kind]).fetchone()[0]
    return r > 0


def _log(con, trade_date, ts_code, kind, rows, status):
    con.execute(
        "DELETE FROM load_log WHERE trade_date=? AND ts_code=? AND kind=?",
        [pd.Timestamp(trade_date), ts_code, kind])
    con.execute(
        "INSERT INTO load_log VALUES (?,?,?,?,?,?)",
        [pd.Timestamp(trade_date), ts_code, kind, int(rows), status, dt.datetime.now()])


def _fetch(pro, fn, ts_code, trade_date, **kw):
    for attempt in range(MAX_RETRY):
        try:
            return fn(ts_code=ts_code, trade_date=trade_date, **kw)
        except Exception as e:
            if attempt == MAX_RETRY - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def record_day(pro, con, trade_date, codes, force=False):
    td = pd.Timestamp(trade_date)
    stats = {"m1_ok": 0, "m1_rows": 0, "auc_ok": 0, "auc_rows": 0,
             "m1_skip": 0, "auc_skip": 0, "fail": []}
    for i, code in enumerate(codes):
        try:
            skipped = False
            if not force and _done(con, td, code, "m1"):
                stats["m1_skip"] += 1; skipped = True
            else:
                df = _fetch(pro, pro.stk_mins, code, trade_date, freq="1min")
                n = 0 if df is None else len(df)
                if n:
                    df["trade_time"] = pd.to_datetime(df["trade_time"])
                    # stk_mins 返回含 trade_date 在内的滚动窗口（约 7 个交易日），
                    # 整窗覆盖写：既保证幂等，也让近期缺口自愈
                    con.execute("DELETE FROM m1 WHERE ts_code=? AND trade_time BETWEEN ? AND ?",
                                [code, df["trade_time"].min(), df["trade_time"].max()])
                    con.register("_df", df[["ts_code", "trade_time", "open", "high",
                                            "low", "close", "vol", "amount"]])
                    con.execute("INSERT INTO m1 SELECT * FROM _df")
                _log(con, td, code, "m1", n, "ok")
                stats["m1_ok"] += 1; stats["m1_rows"] += n
            time.sleep(SLEEP)
            if not force and _done(con, td, code, "auction"):
                stats["auc_skip"] += 1
            else:
                skipped = False
                da = _fetch(pro, pro.stk_auction, code, trade_date)
                n = 0 if da is None else len(da)
                if n:
                    da["trade_date"] = pd.to_datetime(da["trade_date"])
                    con.execute("DELETE FROM auction WHERE ts_code=? AND trade_date=?",
                                [code, td])
                    con.register("_da", da[["ts_code", "trade_date", "vol", "price",
                                            "amount", "pre_close", "turnover_rate",
                                            "volume_ratio", "float_share"]])
                    con.execute("INSERT INTO auction SELECT * FROM _da")
                _log(con, td, code, "auction", n, "ok")
                stats["auc_ok"] += 1; stats["auc_rows"] += n
        except Exception as e:
            stats["fail"].append(f"{code}: {str(e)[:80]}")
            try:
                _log(con, td, code, "m1", 0, "fail")
            except Exception:
                pass
        time.sleep(0.05 if skipped else SLEEP)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(codes)} m1={stats['m1_ok']}({stats['m1_rows']}r) "
                  f"auc={stats['auc_ok']} fail={len(stats['fail'])}", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="daily", choices=["daily", "backfill"])
    ap.add_argument("--date", default=None, help="YYYYMMDD")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import tushare as ts
    pro = ts.pro_api(TOKEN)
    codes, member_date = universe()
    if a.limit:
        codes = codes[: a.limit]
    print(f"universe: CSI300 成分 x{len(codes)} (members@{member_date})", flush=True)

    con = connect()
    t0 = time.time()
    if a.mode == "daily":
        td = latest_trade_day(pro, a.date)
        print(f"record {td} ...", flush=True)
        st = record_day(pro, con, td, codes, force=a.force)
        days = [td]
    else:
        cal = pro.trade_cal(exchange="SSE", start_date=a.start, end_date=a.end)
        days = [d for d in cal[cal["is_open"] == 1]["cal_date"]]
        st = {"m1_ok": 0, "m1_rows": 0, "auc_ok": 0, "auc_rows": 0,
              "m1_skip": 0, "auc_skip": 0, "fail": []}
        for d in days:
            print(f"backfill {d} ...", flush=True)
            s = record_day(pro, con, d, codes, force=a.force)
            for k in ["m1_ok", "m1_rows", "auc_ok", "auc_rows", "m1_skip", "auc_skip"]:
                st[k] += s[k]
            st["fail"].extend(s["fail"])
    con.close()
    el = time.time() - t0

    summary = {
        "mode": a.mode, "days": days, "universe": len(codes),
        "elapsed_s": round(el, 1),
        "m1": {"loaded": st["m1_ok"], "rows": st["m1_rows"], "skip": st["m1_skip"]},
        "auction": {"loaded": st["auc_ok"], "rows": st["auc_rows"], "skip": st["auc_skip"]},
        "fail": st["fail"][:20], "n_fail": len(st["fail"]),
    }
    out = os.path.join(WS, "data", "hf", f"record_{days[-1]}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False)[:600], flush=True)

    try:
        sys.path.insert(0, WS)
        import event_bus
        event_bus.publish("hf_minute_done", {
            "days": days, "m1_rows": st["m1_rows"], "auc_rows": st["auc_rows"],
            "n_fail": len(st["fail"]), "elapsed_s": round(el, 1)})
    except Exception as e:
        print(f"事件发布失败（非致命）: {e}", flush=True)
    return 0 if not st["fail"] else 2


if __name__ == "__main__":
    sys.exit(main())
