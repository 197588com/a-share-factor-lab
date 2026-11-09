# -*- coding: utf-8 -*-
"""P4 回测：walk-forward 分数 → qlib TopkDropout（A股规则）+ 成本/容量分析。

主引擎（framework 1.5）：
- TopkDropout(topk=50, n_drop=5)，信号 = walk-forward 各 fold 模型测试窗分数
- T+1：次日开盘价成交（deal_price=open）；涨跌停禁交易（forbid_all_trade_at_limit，
  前提是数据带涨跌停字段——见报告诚实清单）
- 成本基准：open 0.05% / close 0.15%（A股摩擦典型值），另跑双边 15bp 敏感性

输出：data/backtest/wf_backtest.csv + 控制台汇总

用法：
  python backtest_p4.py --model alstm --folds 0-7 [--open-cost 0.0005 --close-cost 0.0015]
"""
import os, sys, argparse
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
SCORE_DIR = os.path.join(WS, "data", "scores")
OUT_DIR = os.path.join(WS, "data", "backtest")


def mask_limit_up(pred):
    """涨停不可买：双闸门——
    ① T 日收盘涨停（T 日涨幅代理，保守侧：收盘封板次日大概率高开/续板，挡）；
    ② 执行日 T+1 开盘即涨停（open ≥ T收盘×(1+涨停幅)-0.005，真实不可成交口径）。
    自转数据无 $change/$limit 字段，用 $close/$open 手工算；
    主板 ±9.8%、创业板(300)2020-08-24 后 ±19.8%、科创板(688)±19.8%、ST ±4.8%（无 name 表，主板近似）。
    只挡买入不挡卖出（卖出侧需交易所级撮合支持，列为诚实清单近似）。"""
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    insts = sorted(pred.index.get_level_values("instrument").unique())
    px = D.features(insts, ["$close", "$open"],
                    start_time=pred.index.get_level_values("datetime").min(),
                    end_time=pred.index.get_level_values("datetime").max())
    # D.features 返回 index(instrument, datetime)、列=字段；对齐到 pred 的 (datetime, instrument)
    close = px["$close"].swaplevel().sort_index()
    open_ = px["$open"].swaplevel().sort_index()
    chg = close.groupby(level="instrument").pct_change()
    # 执行日开盘 vs 信号日收盘（同一行 t：open 取 t+1 需 shift(-1)）
    nxt_open = open_.groupby(level="instrument").shift(-1)
    open_gap = (nxt_open / close - 1)
    df = pred.to_frame("score").join(chg.rename("chg"), how="left").join(open_gap.rename("ogap"), how="left")
    code = df.index.get_level_values("instrument").str[2:]
    board = np.where(code.str.startswith("68"), 0.198,
                     np.where(code.str.startswith("30"), 0.198, 0.098))
    date = df.index.get_level_values("datetime")
    gem20 = (code.str.startswith("30")) & (date >= pd.Timestamp("2020-08-24"))
    thr = np.where(gem20, 0.198, board)
    n_mask_t = int((df["chg"] >= thr).sum())
    df.loc[df["chg"] >= thr, "score"] = np.nan
    # T+1 开盘即涨停：open ≥ T收盘×(1+幅)−0.005 ⇔ ogap ≥ 幅 − 0.005/T收盘
    close_t = close.reindex(df.index)
    ogap_th = thr - 0.005 / close_t.to_numpy(dtype=float).clip(min=1e-9)
    n_mask_open = int((df["score"].notna() & (df["ogap"].to_numpy(dtype=float) >= ogap_th)).sum())
    df.loc[df["ogap"].to_numpy(dtype=float) >= ogap_th, "score"] = np.nan
    print(f"mask_limit_up: T日封板挡 {n_mask_t} 格，T+1开盘涨停挡 {n_mask_open} 格", flush=True)
    out = df["score"].dropna()
    return out


def run_bt(pred, start, end, open_cost, close_cost, name=None):
    import qlib
    from qlib.config import REG_CN
    from qlib.backtest import backtest
    from qlib.contrib.strategy import TopkDropoutStrategy
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    cal = pd.read_csv(os.path.expanduser("~/.qlib/qlib_data/cn_data/calendars/day.txt"),
                      header=None)[0].tolist()
    end = min(str(end), cal[-2])   # 测试窗超出日历尽头时收敛；末日下一步取 calendar[i+1]，需留一日余量
    strategy = TopkDropoutStrategy(topk=50, n_drop=5, signal=pred)
    from qlib.backtest.executor import SimulatorExecutor
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    pm, _ = backtest(start_time=start, end_time=end, strategy=strategy, executor=executor,
                     benchmark="SH000300", account=1e8,
                     exchange_kwargs={"freq": "day", "deal_price": "open",
                                      "open_cost": open_cost, "close_cost": close_cost,
                                      "min_cost": 5})
    report_df, positions = pm["1day"]
    ret, bench, cost = report_df["return"], report_df["bench"], report_df["cost"]
    excess = ret - bench - cost
    ann = excess.mean() * 244
    ir = excess.mean() / (excess.std() + 1e-12) * np.sqrt(244)
    nav = (1 + ret - cost).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    turnover = float(report_df["turnover"].mean()) if "turnover" in report_df else float("nan")
    if name:  # 日度收益序列持久化：DSR 正确输入（策略收益非 IC）与超额归因的事实源
        os.makedirs(os.path.join(OUT_DIR, "daily"), exist_ok=True)
        dp = os.path.join(OUT_DIR, "daily", f"{name}.parquet")
        report_df[["return", "bench", "cost"]].assign(
            excess=excess).reset_index().to_parquet(dp, index=False)
    return {"days": len(report_df), "ann_excess": float(ann), "ir": float(ir), "mdd": mdd,
            "turnover": turnover, "total_cost": float(cost.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["alstm", "master"])
    ap.add_argument("--folds", default="0-7")
    ap.add_argument("--open-cost", type=float, default=0.0005)
    ap.add_argument("--close-cost", type=float, default=0.0015)
    ap.add_argument("--tag", default=None, help="成本场景标签（默认 open/close 值）")
    a = ap.parse_args()
    tag = a.tag or f"oc{a.open_cost:.4f}_cc{a.close_cost:.4f}"
    import wf_pipeline as wf
    folds = {f["id"]: f for f in wf.generate_folds(wf.load_config())}
    ids = []
    for part in a.folds.split(","):
        if "-" in part:
            x, y = part.split("-"); ids.extend(range(int(x), int(y) + 1))
        else:
            ids.append(int(part))

    rows = []
    for i in ids:
        f = folds[i]
        pq = os.path.join(SCORE_DIR, f"{a.model}_fold{i}.parquet")
        df = pd.read_parquet(pq)
        df["instrument"] = df["instrument"].str.upper()   # 交易所代码大写，否则撮合全部拒单
        pred = df.set_index(["date", "instrument"])["score"]
        pred.index = pred.index.set_names(["datetime", "instrument"])
        pred.index = pd.MultiIndex.from_arrays([pd.to_datetime(pred.index.get_level_values(0)),
                                                pred.index.get_level_values(1)],
                                               names=["datetime", "instrument"])
        pred = mask_limit_up(pred)
        # 信号最后一日无 T+1 可成交（日历尽头），剔除避免撮合越界
        pred = pred.sort_index()
        last_dt = pred.index.get_level_values("datetime").max()
        cal = pd.read_csv(os.path.expanduser("~/.qlib/qlib_data/cn_data/calendars/day.txt"),
                          header=None)[0].tolist()
        if str(last_dt)[:10] >= cal[-1]:
            pred = pred[pred.index.get_level_values("datetime") < last_dt]
        r = run_bt(pred, f["test"][0], f["test"][1], a.open_cost, a.close_cost,
                   name=f"{a.model}_fold{i}_{tag}")
        r.update({"model": a.model, "fold": i, "test_year": f["test"][0][:4], "cost_tag": tag})
        rows.append(r)
        print(f"{a.model}_fold{i} ({r['test_year']}) [{tag}]: ann_excess={r['ann_excess']*100:.2f}% "
              f"IR={r['ir']:.2f} MDD={r['mdd']*100:.1f}% turnover={r['turnover']:.3f}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{a.model}_{tag}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved:", out)


if __name__ == "__main__":
    main()
