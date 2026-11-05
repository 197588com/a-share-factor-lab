# -*- coding: utf-8 -*-
"""因子生命周期在线监控（framework_design.md 1.4a 红线实现）。

红线：
  R1  60日滚动 RankIC < 全期|RankIC|/3 且持续 >= 20 日      -> 衰减告警 -> 降级观察
  R2  regime 切换后 10 日内该 regime 下累计 RankIC 转负       -> 状态不适配告警
  R3  因子间 IC 序列相关性（拥挤度代理）> 0.9                -> 拥挤告警（代理口径）

用法：
  python factor_lab/monitor.py          # 全量监控报告 + 告警落库
  python factor_lab/monitor.py --strict # validated/promoted 因子触发红线时 exit 1

口径说明：ic_daily 目前覆盖 2010-2022（P1 测试窗），本脚本在历史上的最近窗口
做干跑验证管线；每日实盘监控随 P4 因子日更链路接入后自动转为实时。
"""
import os
import sys
import datetime as dt
import duckdb
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(WS, "data", "factor_lab.duckdb")
ROLL_WIN, IC_FRAC, STREAK_DAYS = 60, 1 / 3, 20
CROWD_THR = 0.9


def sync_states(con):
    """把最新测试结果的建议状态同步进注册表状态机。"""
    con.execute("""
        UPDATE factor_registry SET state = r.suggested_state, updated_at = ?
        FROM (SELECT DISTINCT ON (factor_id) factor_id, suggested_state
              FROM factor_test_results ORDER BY factor_id, run_at DESC) r
        WHERE factor_registry.factor_id = r.factor_id
          AND factor_registry.state <> r.suggested_state""", [dt.datetime.now()])


def load(con):
    ic = con.execute("SELECT factor_id, trade_date, rank_ic FROM ic_daily ORDER BY 1,2").df()
    reg = con.execute("""SELECT trade_date, regime FROM regime_labels
                         WHERE source='hmm_v2'""").df()
    states = dict(con.execute("SELECT factor_id, state FROM factor_registry").fetchall())
    return ic, reg, states


def check_r1(g, full_mean_abs):
    """R1: 滚动 IC 衰减。返回 (streak_days, cur_roll_ic)。"""
    roll = g.set_index("trade_date")["rank_ic"].rolling(ROLL_WIN).mean().dropna()
    if roll.empty:
        return 0, np.nan
    below = (roll.abs() < full_mean_abs * IC_FRAC).astype(int)
    # 末尾连续为 1 的天数
    streak = int(below.iloc[::-1].eq(1).cumprod().sum())
    return streak, float(roll.iloc[-1])


def check_r2(g, reg):
    """R2: 最近一次 regime 切换后的 10 日内累计 RankIC 是否转负。"""
    m = g.merge(reg, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    if len(m) < 12:
        return None, np.nan, 0
    changed = m["regime"].ne(m["regime"].shift())
    idx = changed[changed].index
    if len(idx) < 2:
        return None, np.nan, 0
    switch = idx[-1]                      # 最近一次切换点
    seg = m.iloc[switch:switch + 10]
    return m["regime"].iloc[switch], float(seg["rank_ic"].sum()), len(seg)


def main():
    strict = "--strict" in sys.argv
    con = duckdb.connect(DB)
    sync_states(con)
    ic, reg, states = load(con)
    full_mean = ic.groupby("factor_id")["rank_ic"].mean().abs()

    # R3: 因子间 IC 序列相关性（拥挤度代理）
    wide = ic.pivot(index="trade_date", columns="factor_id", values="rank_ic")
    corr = wide.corr()

    alerts = []
    rows = []
    for fid, g in ic.groupby("factor_id"):
        fm = float(full_mean[fid])
        streak, roll_ic = check_r1(g, fm)
        last_regime, seg_ic, seg_len = check_r2(g, reg)
        state = states.get(fid, "?")
        red = []
        if streak >= STREAK_DAYS and state in ("validated", "promoted"):
            red.append(f"R1 衰减 {streak}日")
        if seg_len >= 5 and seg_ic < 0 and state in ("validated", "promoted"):
            red.append(f"R2 regime[{last_regime}]切换后{seg_len}日IC转负({seg_ic:+.3f})")
        crowd_with = [c for c in corr.columns
                      if c != fid and abs(corr.loc[fid, c]) > CROWD_THR
                      and states.get(c) in ("validated", "promoted")]
        if crowd_with and state in ("validated", "promoted"):
            red.append(f"R3 拥挤(与{','.join(crowd_with)})")
        rows.append({
            "factor": fid, "state": state,
            "全期RankIC": round(float(g["rank_ic"].mean()), 4),
            f"滚动{ROLL_WIN}日": round(roll_ic, 4) if roll_ic == roll_ic else None,
            "衰减持续": streak,
            "当前regime": last_regime, "regime段IC": round(seg_ic, 3) if seg_ic == seg_ic else None,
            "红线": "; ".join(red) if red else "-",
        })
        for r in red:
            alerts.append((fid, r.split()[0], r, dt.datetime.now()))

    rep = pd.DataFrame(rows).sort_values(["红线", "全期RankIC"],
                                         ascending=[False, False])
    pd.set_option("display.width", 250)
    print(f"=== 因子健康监控 @ {dt.date.today()}（数据至 {ic['trade_date'].max()}）===")
    print(rep.to_string(index=False))

    con.execute("""CREATE TABLE IF NOT EXISTS factor_alerts(
        factor_id VARCHAR, red_line VARCHAR, detail VARCHAR, run_at TIMESTAMP)""")
    if alerts:
        # 同日同因子同红线去重，避免定时任务重复落库
        for a in alerts:
            con.execute("""DELETE FROM factor_alerts WHERE factor_id=? AND red_line=?
                           AND run_at::DATE = CURRENT_DATE""", [a[0], a[1]])
        con.executemany("INSERT INTO factor_alerts VALUES (?,?,?,?)", alerts)
    con.close()

    promoted_red = [a for a in alerts]
    print(f"\nalerts: {len(promoted_red)} 条已落库 factor_alerts")
    if strict and promoted_red:
        sys.exit(1)


if __name__ == "__main__":
    main()
