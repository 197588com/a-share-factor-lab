# -*- coding: utf-8 -*-
"""P1 数据质量校验：每次入库后自动跑的检查集（规则可迭代，失败即报警）。"""
import os, sys
import numpy as np
import pandas as pd
import tushare as ts

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys, os as _os
for _root in {_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}:
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
from token_loader import tushare_token as _tushare_token
TOKEN = _tushare_token()

CHECKS = []

def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco

@check("行情库最新日期 = Tushare 最近交易日")
def chk_freshness():
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    df = D.features(D.instruments("csi300"), ["$close"], start_time="2026-10-01", end_time="2026-12-31")
    qlib_last = df.index.get_level_values("datetime").max().date()
    pro = ts.pro_api(TOKEN)
    today = pd.Timestamp.today().strftime("%Y%m%d")
    tushare_last = pro.trade_cal(exchange="SSE", start_date="20260101", end_date=today, is_open="1")["cal_date"].max()
    tushare_last = pd.to_datetime(tushare_last, format="%Y%m%d").date()
    return qlib_last == tushare_last, f"qlib={qlib_last} tushare(最近已开盘日)={tushare_last}"

@check("价格正值（无零/负价格脏数据）")
def chk_positive_price():
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    df = D.features(D.instruments("csi300"), ["$close"], start_time="2010-01-01", end_time="2026-12-31")
    bad = int((df["$close"] <= 0).sum())
    return bad == 0, f"非正价格记录数={bad}"

@check("成分股覆盖数 >= 300")
def chk_coverage():
    inst = pd.read_csv(os.path.join(WS, "data", "csi300.txt"), sep="\t", header=None)
    return len(inst) >= 300, f"历史成分并集={len(inst)}"

@check("日历连续（长假断档属正常，其他月份不得断档>10天）")
def chk_calendar():
    cal = pd.read_csv(os.path.expanduser("~/.qlib/qlib_data/cn_data/calendars/day.txt"), header=None)[0]
    dt = pd.to_datetime(cal)
    gaps = dt.diff().dt.days.dropna()
    # 断档起点落在长假窗口（春节 1-2 月、国庆 9 月末-10 月）属正常；其余月份出现 >10 天断档才算异常
    big_idx = gaps[gaps > 10].index
    abnormal = [str(dt[i - 1].date()) for i in big_idx if not (dt[i - 1].month in (1, 2) or (dt[i - 1].month == 9 and dt[i - 1].day >= 20) or dt[i - 1].month == 10)]
    return len(abnormal) == 0, f"异常断档={len(abnormal)}（长假断档 {len(big_idx)} 处）" + (f": {abnormal}" if abnormal else "")

def main():
    fails = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"检查异常: {e}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name} | {detail}")
        fails += (not ok)
    print(f"RESULT: {len(CHECKS)-fails}/{len(CHECKS)} passed")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
