# -*- coding: utf-8 -*-
"""corr_enhance.py — corr 族纵向深挖第一刀：相关结构增强因子（factor_decay.py 同口径）。

背景：三单公开事件/资金数据全部未过准入门禁（margin/dragon/mf），corr 族（Alpha158
CORR/CORD，价量相关）是唯一 12 年稳定存活线（-0.003~-0.029，2026 仍 -0.0205）。
本脚本对"相关结构"做四路改造，看能否在存活线上再挖增量：

  CC20     Corr($close, Log($volume+1), 20)         —— 锚：复刻 Alpha158 CORR20 验证口径
  CC20TS   CC20 - rolling_mean(CC20, 250)           —— 时序中性化：相关结构对自身常态的偏离
  CCSPREAD CC5 - CC60                               —— 短长窗口相关结构背离
  RC20     Corr(diff($close), diff(Log($volume+1)), 20) —— 一阶差分相关（与水平相关近似正交）

  python corr_enhance.py fetch    # 拉原始 close/volume（2013 起，留 250 日预热）
  python corr_enhance.py build    # 合成因子面板 -> data/factor_decay/corrx_feat.parquet
  python corr_enhance.py report   # 逐年 RankIC -> data/factor_decay/corrx_ic.csv，
                                  # 并入 family_ic_matrix.csv（族名 corrx）
"""
import os, sys, time
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(WS, "data", "factor_decay")
RAW = os.path.join(FD, "corrx_raw.parquet")
FACTORS = ["CC20", "CC20TS", "CCSPREAD", "RC20"]


def qlib_ctx():
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    return qlib


def fetch():
    qlib_ctx()
    from qlib.data import D
    t0 = time.time()
    raw = D.features(D.instruments("csi300"), ["$close", "$volume"],
                     start_time="2013-01-01", end_time="2026-12-31")
    out = pd.concat({"close": raw["$close"].unstack("instrument"),
                     "volume": raw["$volume"].unstack("instrument")}, axis=1)
    out.to_parquet(RAW)
    print(f"raw {out.shape} ({time.time()-t0:.0f}s) -> corrx_raw.parquet", flush=True)


def _roll_corr(a, b, w):
    """两个 (date x inst) DataFrame 的滚动 Pearson 相关（向量化）。"""
    ma, mb = a.rolling(w).mean(), b.rolling(w).mean()
    cov = (a * b).rolling(w).mean() - ma * mb
    sa, sb = a.rolling(w).std(), b.rolling(w).std()
    return cov / (sa * sb).replace(0, np.nan)


def build():
    raw = pd.read_parquet(RAW)
    close, vol = raw["close"], raw["volume"]
    lv = np.log1p(vol)
    feats = {}
    feats["CC20"] = _roll_corr(close, lv, 20)
    feats["CCSPREAD"] = _roll_corr(close, lv, 5) - _roll_corr(close, lv, 60)
    feats["RC20"] = _roll_corr(close.diff(), lv.diff(), 20)
    feats["CC20TS"] = feats["CC20"] - feats["CC20"].rolling(250, min_periods=60).mean()
    feat = pd.DataFrame({k: v.stack() for k, v in feats.items()})
    feat.index.names = ["datetime", "instrument"]
    feat = feat.reset_index()
    feat["datetime"] = pd.to_datetime(feat["datetime"])
    feat = feat.set_index(["datetime", "instrument"]).sort_index()
    feat = feat[feat.index.get_level_values("datetime") >= "2015-01-01"]
    feat.to_parquet(os.path.join(FD, "corrx_feat.parquet"))
    print(f"corrx panel {feat.shape} -> corrx_feat.parquet", flush=True)


def rank_ic_by_year(year):
    feat = pd.read_parquet(os.path.join(FD, "corrx_feat.parquet"))
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
        rows.append({"year": year, "feature": col, "family": "corrx",
                     "IC": float(ic.mean()), "ICIR": float(ic.mean() / (ic.std() + 1e-12))})
    return pd.DataFrame(rows)


def report():
    years = list(range(2015, 2027))
    ic = pd.concat([rank_ic_by_year(y) for y in years if len(rank_ic_by_year(y))])
    ic.to_csv(os.path.join(FD, "corrx_ic.csv"), index=False)
    print(ic.pivot(index="feature", columns="year", values="IC").round(4).to_string())
    fp = os.path.join(FD, "family_ic_matrix.csv")
    fam = pd.read_csv(fp, index_col=0)
    fam = fam.drop(index="corrx", errors="ignore")
    mrow = ic.groupby("year")["IC"].mean()
    fam.loc["corrx"] = [round(float(mrow.get(y, np.nan)), 6) for y in years]
    fam = fam.reindex(columns=[str(y) for y in years])
    fam.to_csv(fp)
    print("\nfamily_ic_matrix.csv 已并入 corrx 行", flush=True)


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
