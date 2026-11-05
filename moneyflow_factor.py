# -*- coding: utf-8 -*-
"""moneyflow_factor.py — 个股资金流另类因子准入评估（factor_decay.py 同口径，第三单）。

因子族 mf（4 个候选，T 日收盘后公布，对 T+1→T+2 标签无泄漏）：
  MFN   20日主力净流入率  —— mean(net_mf_amount / 成交总额, 20d)
  ELGN  20日特大单净流入率 —— mean((buy_elg - sell_elg) / 成交总额, 20d)
  LGN   20日大单净流入率   —— mean((buy_lg - sell_lg) / 成交总额, 20d)
  SMN   20日小单净流入率   —— mean((buy_sm - sell_sm) / 成交总额, 20d)
（日内资金流 dense 日频，无需零填充；20 日滚动平滑降噪，与 margin/dragon 口径一致）

  python moneyflow_factor.py fetch    # 拉全历史（断点续传，缓存 data/moneyflow/）
  python moneyflow_factor.py build    # 合成因子面板 -> data/factor_decay/mf_feat.parquet
  python moneyflow_factor.py report   # 逐年 RankIC -> data/factor_decay/mf_ic.csv，
                                      # 并把 mf 族并入 family_ic_matrix.csv（准入基线）
"""
import os, sys, time
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(WS, "data", "moneyflow")
FD = os.path.join(WS, "data", "factor_decay")
os.makedirs(MD, exist_ok=True)

import sys as _sys, os as _os
for _root in {_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}:
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
from token_loader import tushare_token as _tushare_token
TOKEN = _tushare_token()
FACTORS = ["MFN", "ELGN", "LGN", "SMN"]


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
            d = pro.moneyflow(ts_code=ts_code, start_date="20140101", end_date="20261231")
            if d is not None and len(d):
                d.to_csv(os.path.join(MD, f"{qc}.csv"), index=False)
        except Exception as e:
            print(f"FAIL {qc}: {e}", flush=True)
        if i % 25 == 24:
            print(f"{i+1}/{len(todo)}", flush=True)
        time.sleep(0.13)
    print("MONEYFLOW FETCH DONE", flush=True)


def build():
    frames = []
    for fn in sorted(os.listdir(MD)):
        if not fn.endswith(".csv"):
            continue
        qc = fn[:-4]
        m = pd.read_csv(os.path.join(MD, fn), dtype={"trade_date": str})
        if not len(m):
            continue
        m["date"] = pd.to_datetime(m["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        m = m.sort_values("date")
        ba = m["buy_sm_amount"] + m["buy_md_amount"] + m["buy_lg_amount"] + m["buy_elg_amount"]
        sa = m["sell_sm_amount"] + m["sell_md_amount"] + m["sell_lg_amount"] + m["sell_elg_amount"]
        tot = (ba + sa).replace(0, np.nan)
        m["MFN"] = (m["net_mf_amount"] / tot).rolling(20, min_periods=10).mean()
        m["ELGN"] = ((m["buy_elg_amount"] - m["sell_elg_amount"]) / tot).rolling(20, min_periods=10).mean()
        m["LGN"] = ((m["buy_lg_amount"] - m["sell_lg_amount"]) / tot).rolling(20, min_periods=10).mean()
        m["SMN"] = ((m["buy_sm_amount"] - m["sell_sm_amount"]) / tot).rolling(20, min_periods=10).mean()
        frames.append(m[["date", "MFN", "ELGN", "LGN", "SMN"]].assign(
            instrument=qc).set_index(["date", "instrument"]))
    feat = pd.concat(frames).sort_index()
    feat.index.set_names(["datetime", "instrument"], inplace=True)
    feat = feat.reset_index()
    feat["datetime"] = pd.to_datetime(feat["datetime"])
    feat = feat.set_index(["datetime", "instrument"])
    feat.to_parquet(os.path.join(FD, "mf_feat.parquet"))
    print(f"mf panel {feat.shape} -> mf_feat.parquet", flush=True)


def rank_ic_by_year(year):
    feat = pd.read_parquet(os.path.join(FD, "mf_feat.parquet"))
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
        rows.append({"year": year, "feature": col, "family": "mf",
                     "IC": float(ic.mean()), "ICIR": float(ic.mean() / (ic.std() + 1e-12))})
    return pd.DataFrame(rows)


def report():
    years = list(range(2015, 2027))
    ic = pd.concat([rank_ic_by_year(y) for y in years if len(rank_ic_by_year(y))])
    ic.to_csv(os.path.join(FD, "mf_ic.csv"), index=False)
    print(ic.pivot(index="feature", columns="year", values="IC").round(4).to_string())
    # 并入族级矩阵（追加/替换 mf 行）
    fp = os.path.join(FD, "family_ic_matrix.csv")
    fam = pd.read_csv(fp, index_col=0)
    fam = fam.drop(index="mf", errors="ignore")
    mrow = ic.groupby("year")["IC"].mean()
    fam.loc["mf"] = [round(float(mrow.get(y, np.nan)), 6) for y in years]
    fam = fam.reindex(columns=[str(y) for y in years])
    fam.to_csv(fp)
    print("\nfamily_ic_matrix.csv 已并入 mf 行", flush=True)


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
