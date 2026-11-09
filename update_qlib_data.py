# -*- coding: utf-8 -*-
"""增量更新：把 CSV 行情补齐到 tushare 最新交易日，并检测窗口内是否有除权事件。

除权检测逻辑：对比窗口内 adj_factor 与 CSV 末行隐含因子，若发生变化
（分红/送转），对该股做全量重抓，保证前复权口径一致。
"""
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
SINCE = "20261001"


def ts_code_of(qc: str) -> str:
    return qc[2:] + (".SH" if qc.startswith("sh") else ".SZ")


def load_csv(qc):
    p = os.path.join(CSV_DIR, f"{qc}.csv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    return pd.read_csv(p, dtype={"date": str})


def fetch_window(ts_code, start, end):
    d = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    if d is None or d.empty:
        return None, None
    f = pro.adj_factor(ts_code=ts_code, start_date=start, end_date=end)
    d = d.merge(f[["trade_date", "adj_factor"]], on="trade_date", how="left")
    return d, f


def apply_adj(d, latest_factor):
    k = d["adj_factor"] / latest_factor
    out = pd.DataFrame({
        "date": pd.to_datetime(d["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d"),
        "open": (d["open"] * k).round(4), "high": (d["high"] * k).round(4),
        "low": (d["low"] * k).round(4), "close": (d["close"] * k).round(4),
        "volume": d["vol"], "amount": d["amount"], "factor": 1.0,
    }).sort_values("date")
    return out


def update_one(qc, full_refetch=False):
    ts_code = ts_code_of(qc)
    old = load_csv(qc)
    try:
        if full_refetch or old is None:
            d = pro.daily(ts_code=ts_code, start_date="20080101", end_date="20261231")
            if d is None or d.empty:
                return True
            f = pro.adj_factor(ts_code=ts_code, start_date="20080101", end_date="20261231")
            d = d.merge(f[["trade_date", "adj_factor"]], on="trade_date", how="left")
            d["adj_factor"] = d["adj_factor"].ffill().bfill()
            new = apply_adj(d, d.sort_values("trade_date")["adj_factor"].iloc[-1])
        else:
            d, f = fetch_window(ts_code, SINCE, "20261231")
            if d is None:
                return True
            new = apply_adj(d, d.sort_values("trade_date")["adj_factor"].iloc[-1])
            have = set(old["date"])
            add = new[~new["date"].isin(have)]
            if add.empty:
                return True
            # 除权检测：旧末行收盘价的复权因子对比
            # 容差仅为 4 位小数舍入级（相对 1e-3 或绝对 0.005 元，取大者）——
            # 旧版 2% 门控会漏掉 <2% 的常规分红（A 股常见 0.3%~1.5%），导致历史前复权价陈旧
            last_date = old["date"].iloc[-1]
            fa = f.set_index("trade_date")["adj_factor"]
            if last_date.replace("-", "") in fa.index:
                k_old_row = old["close"].iloc[-1]
                d_all = pro.daily(ts_code=ts_code, start_date=last_date.replace("-", ""), end_date=last_date.replace("-", ""))
                raw_close = d_all["close"].iloc[0] if d_all is not None and not d_all.empty else None
                tol = max(abs(raw_close) * 1e-3, 0.005) if raw_close else None
                if raw_close and abs(raw_close - k_old_row) > tol:
                    return update_one(qc, full_refetch=True)  # 发生除权，全量重抓
            new = pd.concat([old, add], ignore_index=True).sort_values("date")
        new.to_csv(os.path.join(CSV_DIR, f"{qc}.csv"), index=False)
        return True
    except Exception as e:
        print(f"FAIL {qc}: {e}", flush=True)
        return False


def main():
    codes = [ln.split("\t")[0].strip() for ln in open(os.path.join(WS, "data", "csi300.txt"))]
    start, count = int(sys.argv[1]), int(sys.argv[2])
    batch = codes[start:start + count]
    done = 0
    for qc in batch:
        done += update_one(qc)
        time.sleep(0.15)
    print(f"UPDATE DONE {start}..{start+len(batch)} ok={done}/{len(batch)}")


if __name__ == "__main__":
    main()
