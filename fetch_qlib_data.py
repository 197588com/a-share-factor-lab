# -*- coding: utf-8 -*-
"""从 Tushare 拉取 CSI300 成分股日线，输出 qlib dump 所需 CSV（前复权口径）。"""
import os, sys, time
import pandas as pd
import tushare as ts

import sys as _sys, os as _os
for _root in {_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}:
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
from token_loader import tushare_token as _tushare_token
TOKEN = _tushare_token()
pro = ts.pro_api(TOKEN)

WS = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(WS, "data", "csv")
os.makedirs(CSV_DIR, exist_ok=True)


def qlib_code(ts_code: str) -> str:
    code, mkt = ts_code.split(".")
    return ("sh" if mkt == "SH" else "sz") + code


def get_csi300_members():
    """历史成分（月度），返回 {qlib_code: (start_date, end_date)} 近似区间。"""
    cache = os.path.join(WS, "data", "csi300_members.csv")
    if os.path.exists(cache):
        w = pd.read_csv(cache, dtype=str)
    else:
        frames = []
        for year in range(2008, 2027):
            df = pro.index_weight(index_code="000300.SH", start_date=f"{year}0101", end_date=f"{year}1231")
            frames.append(df)
            time.sleep(0.35)
        w = pd.concat(frames, ignore_index=True)
        w.to_csv(cache, index=False)
    w["trade_date"] = pd.to_datetime(w["trade_date"], format="%Y%m%d")
    w = w.sort_values("trade_date")
    ranges = {}
    for code, g in w.groupby("con_code"):
        months = sorted(g["trade_date"].unique())
        start = pd.Timestamp(months[0])
        end = pd.Timestamp(months[-1]) + pd.offsets.MonthEnd(0)
        # 成分调整次月生效：起点取下月首日近似
        ranges[qlib_code(code)] = ((start + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
                                   end.strftime("%Y-%m-%d"))
    return ranges


def fetch_one(ts_code: str) -> bool:
    out = os.path.join(CSV_DIR, f"{qlib_code(ts_code)}.csv")
    if os.path.exists(out):
        return True
    try:
        d = pro.daily(ts_code=ts_code, start_date="20080101", end_date="20261231")
        if d is None or d.empty:
            open(out, "w").close()  # 空标的占位
            return True
        f = pro.adj_factor(ts_code=ts_code, start_date="20080101", end_date="20261231")
        d = d.merge(f[["trade_date", "adj_factor"]], on="trade_date", how="left")
        d["adj_factor"] = d["adj_factor"].ffill().bfill()
        latest = d.sort_values("trade_date")["adj_factor"].iloc[-1]
        k = d["adj_factor"] / latest  # 前复权因子，最新日=1
        d["open"] = (d["open"] * k).round(4)
        d["high"] = (d["high"] * k).round(4)
        d["low"] = (d["low"] * k).round(4)
        d["close"] = (d["close"] * k).round(4)
        out_df = pd.DataFrame({
            "date": pd.to_datetime(d["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d"),
            "open": d["open"], "high": d["high"], "low": d["low"], "close": d["close"],
            "volume": d["vol"], "amount": d["amount"], "factor": 1.0,
        }).sort_values("date")
        out_df.to_csv(out, index=False)
        return True
    except Exception as e:
        print(f"FAIL {ts_code}: {e}", flush=True)
        return False


def main():
    ranges = get_csi300_members()
    # 映射回 tushare 代码
    def ts_code_of(qc):
        return qc[2:] + (".SH" if qc.startswith("sh") else ".SZ")

    codes = sorted(ranges.keys())
    start, count = int(sys.argv[1]), int(sys.argv[2])
    batch = codes[start:start + count]
    done = 0
    for qc in batch:
        ok = fetch_one(ts_code_of(qc))
        done += ok
        time.sleep(0.31)  # 限速，双接口约 200次/分钟以内
    print(f"BATCH DONE {start}..{start+len(batch)} ok={done}/{len(batch)}")

    # 写 instruments 文件（每次更新，均为全量区间表）
    inst_path = os.path.join(WS, "data", "csi300.txt")
    with open(inst_path, "w") as fh:
        for qc, (s, e) in sorted(ranges.items()):
            fh.write(f"{qc}\t{s}\t{e}\n")
    print("instruments written:", len(ranges))


if __name__ == "__main__":
    main()
