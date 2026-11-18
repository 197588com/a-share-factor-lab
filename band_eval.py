# -*- coding: utf-8 -*-
"""band_eval.py — 名次带×Dropout 组合评估（P28 探针，2026-11-18）。

动机（P15 证据）：顶部第 1 名是噪声（t=0.08），区分度在 rank 6-50（驼峰）。
第一版探针发现：每日重选口径日换手 0.5-0.7，成本 31%/年量级，全带净超额深负——
成本敏感度比静态年成本估算更高，Dropout 缓冲是首要检验对象。

口径勘误（均保留审计痕迹，不做事后晋升）：
  · 初版把 dropout 写成“跌出 hi+drop 才卖”；核对本机 qlib 源码后，本版改为
    TopkDropoutStrategy.generate_trade_decision 的真实算法：非持仓候选取前
    n_drop+topk-len(last)，与持仓合并按当期分数排序，卖出落在尾部 n_drop 的持仓，
    再从候选头部回补。top50/drop5 是生产策略的近似复刻。
  · 原预注册把 top10/drop0 误称“生产基准”；实际生产为 top50/drop5。
    原对照仍保留，另加生产对照；本探针只决定是否有资格正式立项，不直接晋级。

预注册（先于首版结果写死）：
  · 数据：data/rank_discrim/fold_*.pkl（与 P15 同一批推理缓存，不新增训练/调参；
    样本外 fold8-11 段全部在 2026-07-01 封存线之前，无 holdout 触碰）
  · 日超额：持仓当日标签 y 均值 − 当日全宇宙 y 均值（与 P15 分桶同定义）
  · 成本：卖出 0.0015 / 买入 0.0005（生产费率），按当日卖买占组合比例分别计费
  · 网格：带 ∈ {top10, 6-50, 2-50, top50} × drop ∈ {0, 5, 10}（drop=0=每日重选）
  · 判定：以样本外（fold8-11，2023-2026）净超额 IR 为主判据；
    相对原对照 top10/drop0 的改善须全期与样本外同号，才记“值得走三层闸门”
  · 账本：kind=probe，不计 Bonferroni n（P21 口径）；正式立项另走三层闸门 + 账本 +1
  · 输出：data/rank_discrim/band_eval.json + 终端摘要

已知近似：y=Ref($close,-2)/Ref($close,-1)-1，是 T+1 收盘→T+2 收盘标签；
生产回测是 T+1 开盘成交且基准为 CSI300 指数。因此本探针只支持同口径相对比较，
绝对年化值不得替代 P4/P21 生产回测，也不得拿来修复 DSR。
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(WS, "data", "rank_discrim")
SELL_COST, BUY_COST = 0.0015, 0.0005
OOS_FOLDS = [8, 9, 10, 11]
BANDS = {"top10": (1, 10), "6-50": (6, 50), "2-50": (2, 50), "top50": (1, 50)}
DROPS = [0, 5, 10]
PRODUCTION = "top50/drop5"
ORIGINAL_CONTROL = "top10/drop0"


def fold_series(fold_id, lo, hi, drop):
    """qlib TopkDropout 的同标签近似。返回毛/净日超额及单边日换手。"""
    with open(os.path.join(RD, f"fold_{fold_id}.pkl"), "rb") as fh:
        df = pickle.load(fh)
    dates = sorted(df.d.unique())
    n_hold = hi - lo + 1
    held = set()
    gross, net, tos = {}, {}, []
    for d in dates:
        g = df[df.d == d]
        g = g[np.isfinite(g.p) & np.isfinite(g.y)].sort_values("p", ascending=False)
        uni = float(g.y.mean())
        if not held:
            held = set(g.iloc[lo - 1:min(hi, len(g))].code)
            n_sell = 0
        elif drop == 0:  # 每日重选：卖出全部不在带内的持仓
            band = set(g.iloc[lo - 1:min(hi, len(g))].code)
            sold = held - band
            n_sell = len(sold)
            held -= sold
            if len(held) < n_hold:
                for c in g.iloc[lo - 1:min(hi, len(g))].code:
                    if len(held) >= n_hold:
                        break
                    if c not in held:
                        held.add(c)
        else:
            # qlib method_buy=top：候选来自带内非持仓，按分数从高到低。
            band_codes = list(g.iloc[lo - 1:min(hi, len(g))].code)
            candidates = [c for c in band_codes if c not in held]
            n_today = drop + n_hold - len(held)
            today = candidates[:n_today]
            # method_sell=bottom：持仓+候选合并排序，尾部 drop 名中的持仓才卖出。
            comb = sorted(held.union(today),
                          key=lambda c: float(g.loc[g.code == c, "p"].iloc[0])
                          if (g.code == c).any() else -np.inf,
                          reverse=True)
            tail = set(comb[-drop:])
            sold = held & tail
            n_sell = len(sold)
            held -= sold
            # qlib: buy = today[:len(sell)+topk-len(last)]；卖出后等价于补回 n_hold。
            n_buy = n_hold - len(held)
            held.update(today[:n_buy])
        cost = n_sell / n_hold * (SELL_COST + BUY_COST)  # 卖多少回补多少，双边同比例
        if len(gross) > 0:
            tos.append(n_sell / n_hold)
        day = g[g.code.isin(held)]
        r = float(day.y.mean()) - uni if len(day) else 0.0
        gross[pd.Timestamp(d)] = r
        net[pd.Timestamp(d)] = r - cost
    return pd.Series(gross), pd.Series(net), pd.Series(tos, dtype=float)


def stats(gross, net, turnover):
    return {"days": int(len(net)),
            "grossAnn": round(float(gross.mean() * 244), 4),
            "netAnn": round(float(net.mean() * 244), 4),
            "netIR": round(float(net.mean() / (net.std() + 1e-12) * np.sqrt(244)), 3),
            "turnoverDay": round(float(turnover.mean()), 4) if len(turnover) else 0.0}


def paired_diff(candidate, reference):
    joined = pd.concat([candidate.rename("candidate"), reference.rename("reference")], axis=1).dropna()
    diff = joined.candidate - joined.reference
    t = float(diff.mean() / (diff.std() / np.sqrt(len(diff)) + 1e-12))
    return {"days": int(len(diff)), "meanAnnDiff": round(float(diff.mean() * 244), 4),
            "naivePairedT": round(t, 3)}


def main():
    out = {"sellCost": SELL_COST, "buyCost": BUY_COST, "oosFolds": OOS_FOLDS,
           "productionReference": PRODUCTION, "originalControl": ORIGINAL_CONTROL,
           "absoluteValuesComparableToProduction": False,
           "label": "Ref($close,-2)/Ref($close,-1)-1", "cells": {}, "folds": {}}
    cache = {}
    for band, (lo, hi) in BANDS.items():
        for drop in DROPS:
            key = f"{band}/drop{drop}"
            per_fold = {f: fold_series(f, lo, hi, drop) for f in range(12)}
            cache[key] = per_fold
            g_all = pd.concat([v[0] for v in per_fold.values()]).sort_index()
            n_all = pd.concat([v[1] for v in per_fold.values()]).sort_index()
            t_all = pd.concat([v[2] for v in per_fold.values()])
            g_oos = pd.concat([per_fold[f][0] for f in OOS_FOLDS]).sort_index()
            n_oos = pd.concat([per_fold[f][1] for f in OOS_FOLDS]).sort_index()
            t_oos = pd.concat([per_fold[f][2] for f in OOS_FOLDS])
            out["cells"][key] = {"all": stats(g_all, n_all, t_all),
                                 "oos": stats(g_oos, n_oos, t_oos)}
            out["folds"][key] = {str(f): stats(v[0], v[1], v[2]) for f, v in per_fold.items()}
            print(f"{key}: 全期净IR={out['cells'][key]['all']['netIR']:.2f} "
                  f"样本外净IR={out['cells'][key]['oos']['netIR']:.2f} "
                  f"样本外净超额={out['cells'][key]['oos']['netAnn']:.2%} "
                  f"样本外日换手={out['cells'][key]['oos']['turnoverDay']:.3f}", flush=True)
    prod_oos = pd.concat([cache[PRODUCTION][f][1] for f in OOS_FOLDS]).sort_index()
    ctrl_oos = pd.concat([cache[ORIGINAL_CONTROL][f][1] for f in OOS_FOLDS]).sort_index()
    out["pairedOOS"] = {key: {"vsProduction": paired_diff(pd.concat(
                                    [cache[key][f][1] for f in OOS_FOLDS]).sort_index(), prod_oos),
                              "vsOriginalControl": paired_diff(pd.concat(
                                    [cache[key][f][1] for f in OOS_FOLDS]).sort_index(), ctrl_oos)}
                        for key in cache}
    with open(os.path.join(RD, "band_eval.json"), "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("BAND_JSON: " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
