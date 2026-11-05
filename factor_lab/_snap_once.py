# -*- coding: utf-8 -*-
import qlib
import pandas as pd
from qlib.config import REG_CN
from qlib.data import D


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    f = D.features(D.instruments("csi300"), ["Std($close/Ref($close,1)-1,60)", "$close"],
                   start_time="2026-10-20", end_time="2026-11-03")
    f.columns = ["vol60", "close"]
    snap = f.xs(pd.Timestamp("2026-11-03"), level="datetime").dropna().copy()
    snap["vol60_pct"] = snap["vol60"] * 100
    snap = snap.sort_values("vol60")
    print(f"11-03 池内有效 {len(snap)} 只")
    print("\n-- 波动率最低 10 只（因子看多）--")
    print(snap.head(10)[["vol60_pct", "close"]].round(4).to_string())
    print("\n-- 波动率最高 10 只（因子看空）--")
    print(snap.tail(10)[["vol60_pct", "close"]].round(4).to_string())
    print(f"\n中位 {snap['vol60_pct'].median():.2f}%  最高 {snap['vol60_pct'].max():.2f}%  最低 {snap['vol60_pct'].min():.2f}%")


if __name__ == "__main__":
    main()
