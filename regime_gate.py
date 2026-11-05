# -*- coding: utf-8 -*-
"""regime_gate.py — 组合层改进第一刀：regime 暴露门控回测（2023-2026 样本外）。

背景：P6 四单新因子全部未过终审，边际价值转向组合构建层。本脚本在既有
TopkDropout(top50,n_drop=5) walk-forward 回测（backtest_p4 同源）之上叠加
hmm_roll regime 暴露门控，回答：按市场状态调节仓位能否改善净超额/回撤。

方法（诚实近似清单）：
- 基线逐日收益直接取 qlib 撮合 report_df（与 backtest_p4 同一引擎、同一成本）
- 门控：w(t) = 前一日 regime 映射的暴露（T 日收市判断 → T+1 持仓，无未来函数）
- 门控日收益 = w×(策略收益−策略成本) − 调仓摩擦；现金部分收益记 0（无货基数据）
- 调仓摩擦 = |Δw|×单侧成本（降仓 0.15% 卖出侧，升仓 0.05% 买入侧）——近似，真实
  再入场所选个股与权重路径会有差异
- regime 日度翻转频繁（hmm_roll 特性），摩擦成本是门控的主要敌人，本脚本如实计量

场景：
  nogate      不门控（校准基线，应复现 backtest_p4 数字）
  bear_off    熊市清仓（bear=0，其余满仓）——最激进
  bear_half   熊市半仓（bear=0.5）
  risk_scale  梯度（bear=0.3 / shock=0.5 / quiet=1 / bull=1）

  python regime_gate.py            # folds 8-11 全场景
输出：data/backtest/regime_gate.csv + 控制台汇总
"""
import os, sys
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import wf_pipeline as wf
import backtest_p4 as bp

DB = os.path.join(WS, "data", "factor_lab.duckdb")
OPEN_COST, CLOSE_COST = 0.0005, 0.0015
FOLDS = [8, 9, 10, 11]

SCEN = {
    "nogate": None,
    "bear_off": {"bear": 0.0, "shock": 1.0, "quiet": 1.0, "bull": 1.0},
    "bear_half": {"bear": 0.5, "shock": 1.0, "quiet": 1.0, "bull": 1.0},
    "risk_scale": {"bear": 0.3, "shock": 0.5, "quiet": 1.0, "bull": 1.0},
    # 平滑版：regime 先过 5 日多数投票再映射，压掉日度翻转的磨损
    "bear_off_s5": {"bear": 0.0, "shock": 1.0, "quiet": 1.0, "bull": 1.0, "_smooth": 5},
    "risk_scale_s5": {"bear": 0.3, "shock": 0.5, "quiet": 1.0, "bull": 1.0, "_smooth": 5},
    "bear_half_s5": {"bear": 0.5, "shock": 1.0, "quiet": 1.0, "bull": 1.0, "_smooth": 5},
}


def regime_lut():
    import duckdb
    con = duckdb.connect(DB, read_only=True)
    out = {d: s for d, s in con.execute(
        "SELECT trade_date, regime FROM regime_labels WHERE source='hmm_roll'").fetchall()}
    con.close()
    return out


def daily_bt(pred, start, end):
    """与 backtest_p4.run_bt 同引擎，但返回逐日 report_df。"""
    import qlib
    from qlib.config import REG_CN
    from qlib.backtest import backtest
    from qlib.contrib.strategy import TopkDropoutStrategy
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    cal = pd.read_csv(os.path.expanduser("~/.qlib/qlib_data/cn_data/calendars/day.txt"),
                      header=None)[0].tolist()
    end = min(str(end), cal[-2])
    strategy = TopkDropoutStrategy(topk=50, n_drop=5, signal=pred)
    from qlib.backtest.executor import SimulatorExecutor
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    pm, _ = backtest(start_time=start, end_time=end, strategy=strategy, executor=executor,
                     benchmark="SH000300", account=1e8,
                     exchange_kwargs={"freq": "day", "deal_price": "open",
                                      "open_cost": OPEN_COST, "close_cost": CLOSE_COST,
                                      "min_cost": 5})
    return pm["1day"][0]


def load_pred(fid):
    df = pd.read_parquet(os.path.join(bp.SCORE_DIR, f"alstm_fold{fid}.parquet"))
    df["instrument"] = df["instrument"].str.upper()
    pred = df.set_index(["date", "instrument"])["score"]
    pred.index = pred.index.set_names(["datetime", "instrument"])
    pred.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(pred.index.get_level_values(0)),
         pred.index.get_level_values(1)], names=["datetime", "instrument"])
    pred = bp.mask_limit_up(pred).sort_index()
    last_dt = pred.index.get_level_values("datetime").max()
    cal = pd.read_csv(os.path.expanduser("~/.qlib/qlib_data/cn_data/calendars/day.txt"),
                      header=None)[0].tolist()
    if str(last_dt)[:10] >= cal[-1]:
        pred = pred[pred.index.get_level_values("datetime") < last_dt]
    return pred


def metrics(excess):
    nav = (1 + excess).cumprod()
    return {"ann_excess": float(excess.mean() * 244),
            "ir": float(excess.mean() / (excess.std() + 1e-12) * np.sqrt(244)),
            "mdd": float((nav / nav.cummax() - 1).min())}


def gate_report(df, lut, wmap):
    """df: report_df（index=date），返回门控后逐日超额序列与摩擦统计。"""
    ret, cost, bench = df["return"], df["cost"], df["bench"]
    if wmap is None:
        excess = ret - cost - bench
        return excess, {"flips": 0, "flip_cost": 0.0}
    # w(t)：用前一交易日 regime 决定 t 日持仓（无未来函数）
    dates = df.index
    reg = pd.Series([lut.get(d.date(), "quiet") for d in dates], index=dates)
    smooth = wmap.get("_smooth", None) if wmap else None
    wmap = {k: v for k, v in wmap.items() if not k.startswith("_")} if wmap else None
    if smooth:
        # 5 日多数投票：同权重的 regime 序列 rolling 后取众数（有序类别映射数值）
        rank = {"bear": 0, "shock": 1, "quiet": 2, "bull": 3}
        r = reg.map(rank)
        reg = r.rolling(smooth, min_periods=1).apply(lambda x: pd.Series(x).mode()[0], raw=True
                                                     ).map({v: k for k, v in rank.items()})
    w = reg.map(lambda r: wmap.get(r, 1.0))
    w_use = w.shift(1).fillna(1.0)
    dw = w_use.diff().fillna(w_use.iloc[0] if len(w_use) else 0.0)
    flip_cost = np.where(dw < 0, -dw * CLOSE_COST, dw * OPEN_COST)
    gated = w_use * (ret - cost) - pd.Series(flip_cost, index=dates) - bench
    return gated, {"flips": int((dw != 0).sum()),
                   "flip_cost": float(pd.Series(flip_cost, index=dates).sum())}


def main():
    lut = regime_lut()
    folds = {f["id"]: f for f in wf.generate_folds(wf.load_config())}
    rows = []
    for fid in FOLDS:
        f = folds[fid]
        pred = load_pred(fid)
        rep = daily_bt(pred, f["test"][0], f["test"][1])
        year = f["test"][0][:4]
        for name, wmap in SCEN.items():
            excess, flip = gate_report(rep, lut, wmap)
            m = metrics(excess)
            rows.append({"year": year, "fold": fid, "scenario": name, **m, **flip})
            print(f"{year} fold{fid} [{name:10s}]: ann_excess={m['ann_excess']*100:+.2f}% "
                  f"IR={m['ir']:.2f} MDD={m['mdd']*100:.1f}% flips={flip['flips']} "
                  f"flip_cost={flip['flip_cost']*100:.2f}pp", flush=True)
    out = os.path.join(bp.OUT_DIR, "regime_gate.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved:", out)

    # 汇总：逐年场景对照
    piv = pd.DataFrame(rows).pivot_table(index="scenario", columns="year",
                                         values="ann_excess") * 100
    print("\n年化净超额(%)对照：")
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
