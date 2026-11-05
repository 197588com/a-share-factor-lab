# -*- coding: utf-8 -*-
"""margin_factor.py — 两融另类因子准入评估（factor_decay.py 同口径）。

因子族 margin（3 个候选，T 日收盘后公布，对 T+1→T+2 标签无泄漏）：
  MBR   融资买入额 / 成交额          —— 杠杆资金进攻强度
  MBAL  融资余额 / 5日均值 - 1       —— 杠杆资金存量动量
  MSRT  融券余额 / 融资余额          —— 做空/做多对峙强度

  python margin_factor.py fetch    # 拉全历史两融明细（断点续传，缓存 data/margin/）
  python margin_factor.py build    # 合成因子面板 -> data/factor_decay/margin_feat.parquet
  python margin_factor.py report   # 逐年 RankIC -> data/factor_decay/margin_ic.csv，
                                   # 并把 margin 族并入 family_ic_matrix.csv（准入基线）
"""
import os, sys, time
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(WS, "data", "margin")
FD = os.path.join(WS, "data", "factor_decay")
os.makedirs(MD, exist_ok=True)

import sys as _sys, os as _os
for _root in {_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}:
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
from token_loader import tushare_token as _tushare_token
TOKEN = _tushare_token()
FACTORS = ["MBR", "MBAL", "MSRT"]


def live_codes():
    """csi300.txt 中当前仍在册的成分（与 qlib csi300 活体口径对齐）。"""
    out = []
    for ln in open(os.path.join(WS, "data", "csi300.txt")):
        p = ln.split("\t")
        if len(p) >= 3 and p[2].strip() >= "2026-01-01":
            out.append(p[0].strip())
    return out


def qc_of_ts(ts):
    return ("sh" if ts.endswith(".SH") else "sz") + ts[:6]


def fetch():
    import tushare as ts
    pro = ts.pro_api(TOKEN)
    codes = live_codes()
    todo = [c for c in codes
            if not os.path.exists(os.path.join(MD, f"{c}.csv"))]
    print(f"live={len(codes)} todo={len(todo)}", flush=True)
    for i, qc in enumerate(todo):
        ts_code = qc[2:] + (".SH" if qc.startswith("sh") else ".SZ")
        try:
            d = pro.margin_detail(ts_code=ts_code, start_date="20150101", end_date="20261231")
            if d is not None and len(d):
                d.to_csv(os.path.join(MD, f"{qc}.csv"), index=False)
        except Exception as e:
            print(f"FAIL {qc}: {e}", flush=True)
        if i % 25 == 24:
            print(f"{i+1}/{len(todo)}", flush=True)
        time.sleep(0.13)
    print("MARGIN FETCH DONE", flush=True)


def build():
    frames = []
    for fn in sorted(os.listdir(MD)):
        if not fn.endswith(".csv"):
            continue
        qc = fn[:-4]
        px = os.path.join(WS, "data", "csv", f"{qc}.csv")
        if not os.path.exists(px):
            continue
        m = pd.read_csv(os.path.join(MD, fn), dtype={"trade_date": str})
        if not len(m):
            continue
        k = pd.read_csv(px, dtype={"date": str}, usecols=["date", "amount"])
        m["date"] = pd.to_datetime(m["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        m = m.merge(k, on="date", how="inner")
        m = m.sort_values("date")
        m["MBR"] = m["rzmre"] / m["amount"].replace(0, np.nan)
        m["MBAL"] = m["rzye"] / m["rzye"].rolling(5, min_periods=3).mean() - 1
        m["MSRT"] = m["rqye"] / (m["rzye"] + 1.0)
        frames.append(m[["date", "MBR", "MBAL", "MSRT"]].assign(
            instrument=qc).set_index(["date", "instrument"]))
    feat = pd.concat(frames).sort_index()
    feat.index.set_names(["datetime", "instrument"], inplace=True)
    feat = feat.reset_index()
    feat["datetime"] = pd.to_datetime(feat["datetime"])
    feat = feat.set_index(["datetime", "instrument"])
    feat.to_parquet(os.path.join(FD, "margin_feat.parquet"))
    print(f"margin panel {feat.shape} -> margin_feat.parquet", flush=True)


def rank_ic_by_year(year):
    feat = pd.read_parquet(os.path.join(FD, "margin_feat.parquet"))
    lab = pd.read_parquet(os.path.join(FD, "labels.parquet"))
    d0, d1 = f"{year}-01-01", f"{year}-12-31"
    feat = feat[(feat.index.get_level_values("datetime") >= d0)
                & (feat.index.get_level_values("datetime") <= d1)]
    df = feat.join(lab, how="inner")
    dd = df.index.get_level_values("datetime")
    rows = []
    for col in FACTORS:
        sub = pd.DataFrame({"d": dd, "f": df[col], "y": df["y"]}).dropna()
        if len(sub) < 1000:
            continue
        r = sub.groupby("d")[["f", "y"]].rank()
        sd = sub["d"]
        g = r["f"].groupby(sd)
        fx = r["f"] - g.transform("mean")
        fy = r["y"] - r["y"].groupby(sd).transform("mean")
        cov = (fx * fy).groupby(sd).sum()
        ic = cov / np.sqrt((fx ** 2).groupby(sd).sum() * (fy ** 2).groupby(sd).sum())
        ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
        ic = ic[sd.groupby(sd).size().reindex(ic.index) >= 30]
        if len(ic) < 30:
            continue
        rows.append({"year": year, "feature": col, "family": "margin",
                     "IC": float(ic.mean()), "ICIR": float(ic.mean() / (ic.std() + 1e-12))})
    return pd.DataFrame(rows)


def report():
    years = list(range(2015, 2027))
    ic = pd.concat([rank_ic_by_year(y) for y in years if len(rank_ic_by_year(y))])
    ic.to_csv(os.path.join(FD, "margin_ic.csv"), index=False)
    print(ic.pivot(index="feature", columns="year", values="IC").round(4).to_string())
    # 并入族级矩阵（追加/替换 margin 行）
    fp = os.path.join(FD, "family_ic_matrix.csv")
    fam = pd.read_csv(fp, index_col=0)
    fam = fam.drop(index="margin", errors="ignore")
    mrow = ic.groupby("year")["IC"].mean()
    fam.loc["margin"] = [round(float(mrow.get(y, np.nan)), 6) for y in years]
    fam = fam.reindex(columns=[str(y) for y in years])
    fam.to_csv(fp)
    print("\nfamily_ic_matrix.csv 已并入 margin 行", flush=True)


def main():
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch()
    elif cmd == "build":
        build()
    elif cmd == "report":
        report()


if __name__ == "__main__":
    main()
