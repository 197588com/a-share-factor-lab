# -*- coding: utf-8 -*-
"""update_index_data.py — 沪深300/中证500 指数日线增量更新（csv_index 单一事实源）。

补齐巡检链缺口：daily_chain 原来只复制 csv_index 从不刷新它，导致基准指数
自 2026-11-03 起停更、台账 excess 全空。本脚本从 Tushare index_daily 增量追加，
幂等（按日期去重排序），供巡检链每日调用，也可手动跑。
"""
import os, sys, datetime as dt

import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
import sys as _sys, os as _os
for _root in {_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}:
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
from token_loader import tushare_token as _tushare_token
TOKEN = _tushare_token()
OUT = os.path.join(WS, "data", "csv_index")
INDEXES = {"sh000300": "000300.SH", "sh000905": "000905.SH"}
COLS = ["date", "open", "high", "low", "close", "volume"]


def update_one(pro, name, ts_code):
    p = os.path.join(OUT, f"{name}.csv")
    old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=COLS)
    last = old["date"].max() if len(old) else "1990-01-01"
    start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    end = dt.date.today().strftime("%Y%m%d")
    if start > end:
        print(f"{name}: 已最新 ({last})")
        return 0
    df = pro.index_daily(ts_code=ts_code, start_date=start, end_date=end)
    n = len(df)
    if n:
        df = df.rename(columns={"trade_date": "date", "vol": "volume"})[COLS]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        new = (pd.concat([old, df], ignore_index=True)
               .drop_duplicates("date").sort_values("date"))
        new.to_csv(p, index=False)
        print(f"{name}: +{n} 行, 最新 {new['date'].max()} NEW_ROWS")
    else:
        print(f"{name}: 区间内无新数据 ({start}~{end})")
    return n


def main():
    import tushare as ts
    pro = ts.pro_api(TOKEN)
    total = 0
    for name, code in INDEXES.items():
        total += update_one(pro, name, code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
