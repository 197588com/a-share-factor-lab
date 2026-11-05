# -*- coding: utf-8 -*-
"""regime 条件化因子表现档案（framework_design.md 1.4b 的数据层实现）。

把逐日 IC（ic_daily）按 HMM 状态（regime_labels source='hmm_v2'）重新分组，
产出每个因子在每个状态下的表现统计，落库 factor_regime_stats；
并打印"当前状态下"的因子排序——即市场状况 -> 盈利方向的映射输入。

用法：
  python factor_lab/regime_stats.py           # 全量重建档案 + 当前状态因子排序
  python factor_lab/regime_stats.py --state quiet  # 指定状态看排序
"""
import os
import sys
import datetime as dt
import duckdb
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(WS, "data", "factor_lab.duckdb")
SOURCE = "hmm_v2"


def build(con):
    ic = con.execute("SELECT factor_id, trade_date, rank_ic FROM ic_daily").df()
    reg = con.execute("""SELECT trade_date, regime FROM regime_labels
                         WHERE source=?""", [SOURCE]).df()
    m = ic.merge(reg, on="trade_date", how="inner")
    rows = []
    for (fid, state), g in m.groupby(["factor_id", "regime"]):
        icv = g["rank_ic"].astype(float)
        t = icv.mean() / (icv.std(ddof=1) / np.sqrt(len(icv))) if icv.std(ddof=1) > 0 else np.nan
        rows.append({
            "factor_id": fid, "state": state, "n_days": int(len(g)),
            "rank_ic_mean": float(icv.mean()), "rank_icir": float(icv.mean() / icv.std(ddof=1)) if icv.std(ddof=1) > 0 else np.nan,
            "rank_ic_tstat": float(t),
            "win_rate": float((np.sign(icv) == np.sign(icv.mean())).mean()),
        })
    st = pd.DataFrame(rows)
    con.execute("DROP TABLE IF EXISTS factor_regime_stats")
    con.execute("""CREATE TABLE factor_regime_stats(
        factor_id VARCHAR, state VARCHAR, n_days INTEGER,
        rank_ic_mean DOUBLE, rank_icir DOUBLE, rank_ic_tstat DOUBLE, win_rate DOUBLE)""")
    con.executemany("INSERT INTO factor_regime_stats VALUES (?,?,?,?,?,?,?)",
                    st.itertuples(index=False, name=None))
    print(f"factor_regime_stats: {len(st)} rows ({m['factor_id'].nunique()} factors x {m['regime'].nunique()} states)")
    return st


def main():
    con = duckdb.connect(DB)
    st = build(con)
    states = con.execute("""SELECT state, prob FROM regime_probs
                            WHERE source=? AND trade_date=(SELECT MAX(trade_date) FROM regime_probs
                            WHERE source=?) ORDER BY prob DESC""", [SOURCE, SOURCE]).fetchall()
    con.close()

    cur = None
    for a in sys.argv[1:]:
        if a.startswith("--state"):
            cur = a.split("=")[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
    if cur is None and states:
        cur = states[0][0]

    pd.set_option("display.width", 200)
    print(f"\n当前 HMM 状态: {states[0][0]} (p={states[0][1]:.2f})"
          + (f" | 备选: {states[1][0]} p={states[1][1]:.2f}" if len(states) > 1 else ""))
    sub = st[st["state"] == cur].sort_values("rank_ic_mean")
    names = {}
    try:
        con2 = duckdb.connect(DB, read_only=True)
        names = dict(con2.execute("SELECT factor_id, name FROM factor_registry").fetchall())
        con2.close()
    except Exception:
        pass
    sub = sub.copy()
    sub.insert(0, "因子", sub["factor_id"].map(lambda f: f"{f}({names.get(f, '?')})"))
    print(f"\n=== {cur} 状态下的因子 RankIC 排序（负值=该方向有定价力，看绝对值与 t 值）===")
    print(sub[["因子", "n_days", "rank_ic_mean", "rank_icir", "rank_ic_tstat", "win_rate"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
