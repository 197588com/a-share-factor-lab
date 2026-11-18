# -*- coding: utf-8 -*-
"""ensemble_probe.py — DEE 启发的双模型异质集成最小探针（2026-11-18）。

这不是 DEE 完整复现：不实现 50 专家、bootstrap、特征子集与质量过滤；只用既有
ALSTM/MASTER 样本外评分检验“异质预测器等权聚合是否降低误差”的核心机制。

预注册（先于任何结果）：
  · 数据：folds 0-7（2015-2022），因 MASTER 仅已有这 8 折；不触碰 2026-07-01 后封存期。
  · 标签：rank_discrim 缓存的 y=Ref($close,-2)/Ref($close,-1)-1。
  · 聚合：各模型逐日截面 rank(pct=True)，DEE2=0.5×ALSTM+0.5×MASTER；不搜索权重。
  · 主指标：逐日 RankIC；报告与 ALSTM 的同日配对差、朴素配对 t 和逐年方向。
  · 进入完整 DEE 复现的必要条件： pooled 差>0、配对 t>0、至少 6/8 年方向非负。
    这只是计算资源闸门，不是生产晋升；正式结构候选仍须全 12 fold、Bonferroni、账本。
  · 本探针 kind=probe，不计候选 n；无论结果都不直接改线上模型。
输出：data/rank_discrim/ensemble_probe.json。
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import rankdata

WS = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(WS, "data", "rank_discrim")
SCORES = os.path.join(WS, "data", "scores")
FOLDS = list(range(8))
MODELS = ["alstm", "master", "dee2"]


def rank_ic(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan
    pr, yr = rankdata(p[ok]), rankdata(y[ok])
    if pr.std() < 1e-12 or yr.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(pr, yr)[0, 1])


def fold_daily(fid):
    with open(os.path.join(RD, f"fold_{fid}.pkl"), "rb") as fh:
        base = pickle.load(fh)
    base["d"] = pd.to_datetime(base["d"])
    master = pd.read_parquet(os.path.join(SCORES, f"master_fold{fid}.parquet"))
    master["date"] = pd.to_datetime(master["date"])
    master = master.rename(columns={"date": "d", "instrument": "code", "score": "p_master"})
    df = base.merge(master[["d", "code", "p_master"]], on=["d", "code"],
                    how="left", validate="one_to_one")
    rows = []
    for d, g in df.groupby("d", sort=True):
        if len(g) < 30:
            continue
        pa = g["p"].rank(pct=True).to_numpy()
        pm = g["p_master"].rank(pct=True).to_numpy()
        preds = {"alstm": g["p"].to_numpy(), "master": g["p_master"].to_numpy(),
                 "dee2": 0.5 * (pa + pm)}
        row = {"d": d, "fold": fid, "year": pd.Timestamp(d).year,
               "score_rank_corr": rank_ic(pa, pm)}
        row.update({m: rank_ic(p, g["y"].to_numpy()) for m, p in preds.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df):
    out = {"days": int(len(df))}
    for m in MODELS:
        out[m] = {"rankIC": round(float(df[m].mean()), 6),
                  "rankICIR": round(float(df[m].mean() / (df[m].std() + 1e-12) * np.sqrt(244)), 4)}
    diff = df.dee2 - df.alstm
    out["dee2_minus_alstm"] = {"mean": round(float(diff.mean()), 6),
                               "relative": round(float(diff.mean() / (abs(df.alstm.mean()) + 1e-12)), 4),
                               "naivePairedT": round(float(diff.mean() /
                                                    (diff.std() / np.sqrt(len(diff)) + 1e-12)), 3),
                               "positiveDayShare": round(float((diff > 0).mean()), 4)}
    out["scoreRankCorr"] = round(float(df.score_rank_corr.mean()), 4)
    return out


def main():
    daily = pd.concat([fold_daily(f) for f in FOLDS], ignore_index=True)
    folds = {str(f): summarize(g) for f, g in daily.groupby("fold")}
    years = {str(y): summarize(g) for y, g in daily.groupby("year")}
    pooled = summarize(daily)
    nonnegative_years = sum(v["dee2_minus_alstm"]["mean"] >= 0 for v in years.values())
    resource_gate = bool(pooled["dee2_minus_alstm"]["mean"] > 0
                         and pooled["dee2_minus_alstm"]["naivePairedT"] > 0
                         and nonnegative_years >= 6)
    out = {"scope": "DEE-inspired two-model mechanism probe; not full DEE reproduction",
           "folds": FOLDS, "weights": {"alstm": 0.5, "master": 0.5},
           "pooled": pooled, "byFold": folds, "byYear": years,
           "nonnegativeYears": nonnegative_years,
           "fullDeeResourceGate": resource_gate,
           "productionChange": False,
           "note": "2015-2022 only; no evidence about 2023-2026 until MASTER folds 8-11 are trained"}
    with open(os.path.join(RD, "ensemble_probe.json"), "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
