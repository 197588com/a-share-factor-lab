# -*- coding: utf-8 -*-
"""当前市场状态诊断：规则版 regime + 已验证因子 vol_60d 的快照。"""
import qlib
import numpy as np
import pandas as pd
from qlib.config import REG_CN
from qlib.data import D


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    # ---- 1. 市场状态（与 regime_labels rule_v1 同规则）----
    df = D.features(["sh000300"], ["$close"], start_time="2008-01-01", end_time="2026-12-31")
    close = df["$close"]
    if isinstance(close.index, pd.MultiIndex):
        close = close.xs("sh000300", level="instrument")
    close = close.sort_index()
    ret = close.pct_change()
    ma60, ma250 = close.rolling(60).mean(), close.rolling(250).mean()
    vol20 = ret.rolling(20).std()
    vol_thr = vol20.loc["2008":"2012"].quantile(0.75)

    def label(d):
        c, f, s, v = close[d], ma60[d], ma250[d], vol20[d]
        if np.isnan(f) or np.isnan(s):
            return None
        if c > s and f > s: return "bull"
        if c < s and f < s: return "bear"
        return "volatile" if v > vol_thr else "shock"

    recent = close.index[-15:]
    print("=== 近15个交易日 regime（规则版 v1）===")
    for d in recent:
        c, f, s, v = close[d], ma60[d], ma250[d], vol20[d]
        print(f"{d.date()}  close={c:7.2f}  MA60={f:7.2f}  MA250={s:7.2f}  vol20={v*100:5.2f}%  -> {label(d)}")

    d0 = recent[-1]
    print(f"\n当前: close={close[d0]:.2f} vs MA60 {ma60[d0]:.2f} ({(close[d0]/ma60[d0]-1)*100:+.1f}%) "
          f"vs MA250 {ma250[d0]:.2f} ({(close[d0]/ma250[d0]-1)*100:+.1f}%), vol20={vol20[d0]*100:.2f}% vs 阈值{vol_thr*100:.2f}%")
    yr_ago = close.index[close.index <= d0 - pd.Timedelta(days=365)][-1]
    print(f"近一年涨跌: {(close[d0]/close[yr_ago]-1)*100:+.1f}%")

    # ---- 2. vol_60d 因子当前快照（CSI300 成分，低波=正向）----
    today = str(d0.date())
    f = D.features(D.instruments("csi300"), ["Std($close/Ref($close,1)-1,60)", "$close"],
                   start_time=today, end_time=today)
    f.columns = ["vol60", "close"]
    snap = f.dropna().copy()
    snap["vol60_pct"] = snap["vol60"] * 100
    snap = snap.sort_values("vol60")
    print(f"\n=== vol_60d 因子快照 @ {today}（池内当日有数据 {len(snap)} 只）===")
    print("\n-- 波动率最低 10 只（因子看多，低波异象）--")
    print(snap.head(10)[["vol60_pct", "close"]].to_string())
    print("\n-- 波动率最高 10 只（因子看空）--")
    print(snap.tail(10)[["vol60_pct", "close"]].to_string())
    print(f"\n池内 vol60 中位数 {snap['vol60_pct'].median():.2f}%，最高 {snap['vol60_pct'].max():.2f}%，最低 {snap['vol60_pct'].min():.2f}%")


if __name__ == "__main__":
    main()
