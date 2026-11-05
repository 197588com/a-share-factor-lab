# -*- coding: utf-8 -*-
"""vectorbt 交叉验证（P4 双引擎之快速引擎）。

无 A 股微观结构（T+1/涨跌停/停牌），用于数量级交叉核对 qlib 主引擎：
T 日收盘信号 → T+1 收盘到 T+2 收盘持有 top-N 等权组合，扣双边成本。
差异预期来源：成交价（qlib 用 T+1 开盘价）、涨跌停屏蔽、最小佣金 5 元。

用法：python crosscheck_vbt.py --model alstm --fold 1 [--topk 10]
"""
import os, sys, argparse
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="alstm")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--cost", type=float, default=0.0015)
    ap.add_argument("--n_drop", type=int, default=5)
    a = ap.parse_args()

    df = pd.read_parquet(os.path.join(WS, "data", "scores", f"{a.model}_fold{a.fold}.parquet"))
    df["instrument"] = df["instrument"].str.upper()
    df["date"] = pd.to_datetime(df["date"])

    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    insts = sorted(df["instrument"].unique())
    px = D.features(insts, ["$close"], start_time=df["date"].min(), end_time=df["date"].max() + pd.Timedelta(days=10)).squeeze()
    close = px.unstack("instrument").sort_index()
    ret = close.pct_change().shift(-2)      # T 收盘信号 → T+1 收盘买入，持有到 T+2 收盘

    dates = sorted(df["date"].unique())
    port_ret, bench = [], []
    bench_px = D.features(["sh000300"], ["$close"], start_time=df["date"].min(),
                          end_time=df["date"].max() + pd.Timedelta(days=10)).squeeze().sort_index()
    if isinstance(bench_px.index, pd.MultiIndex):
        bench_px = bench_px.droplevel("instrument")
    # 基准与组合持有同一窗口：d+1 收盘 → d+2 收盘
    bench_ret = bench_px.pct_change().shift(-2)

    prev: list = []   # 当前持仓
    gross_ret = []
    n_changes, costs = [], []
    for d in dates:
        d = pd.Timestamp(d)
        day = df[df["date"] == d]
        sc = day.set_index("instrument")["score"]
        last = sorted(prev, key=lambda c: -sc.get(c, -np.inf))  # 持仓按今日分数排序
        # 买入候选：非持仓中分数最高的 (n_drop + topk - len(last)) 只
        n_buy = a.n_drop + a.topk - len(last)
        cands_new = day[~day["instrument"].isin(last)] \
            .nlargest(max(n_buy, 0), "score")["instrument"].tolist()
        # 合并池按分数排序，仅卖出池中垫底的 n_drop 只（qlib method_sell="bottom"）
        comb = sorted(set(last) | set(cands_new), key=lambda c: -sc.get(c, -np.inf))
        bottom = set(comb[-a.n_drop:]) if len(comb) > a.n_drop else set()
        sell = [c for c in last if c in bottom]
        keep = [c for c in last if c not in bottom]
        buy = cands_new[:max(a.n_drop - len(sell) + a.topk - len(last), 0)]
        top = (keep + buy)[:a.topk]
        r = ret.loc[d, top].mean()
        n_change = len(set(top) ^ set(prev))
        # n_change 为对称差集，已同时含卖出与买入两侧，不再乘 2
        cost = n_change / max(len(top), 1) * a.cost
        prev = top
        n_changes.append(n_change)
        costs.append(cost)
        gross_ret.append(r)
        port_ret.append(r - cost)
        bench.append(bench_ret.loc[d] if d in bench_ret.index else np.nan)

    pr = pd.Series(port_ret, index=pd.to_datetime(dates)).dropna()
    bn = pd.Series(bench, index=pd.to_datetime(dates)).dropna()
    gr = pd.Series(gross_ret, index=pd.to_datetime(dates)).dropna()
    ex = (pr - bn).dropna()
    ex_gross = (gr - bn).dropna()
    ann, ir = ex.mean() * 244, ex.mean() / (ex.std() + 1e-12) * np.sqrt(244)
    ann_g, ir_g = ex_gross.mean() * 244, ex_gross.mean() / (ex_gross.std() + 1e-12) * np.sqrt(244)
    print(f"[vectorbt] {a.model}_fold{a.fold} top{a.topk}_drop{a.n_drop} cost{a.cost:.4f}/side: "
          f"gross ann_excess={ann_g*100:.2f}% IR={ir_g:.2f} | "
          f"net ann_excess={ann*100:.2f}% IR={ir:.2f} | "
          f"日均换手名数={np.mean(n_changes):.1f}/50 日均成本={np.mean(costs)*100:.3f}% "
          f"(vs qlib 主引擎同 fold 对照)")


if __name__ == "__main__":
    main()
