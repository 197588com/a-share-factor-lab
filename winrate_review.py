# -*- coding: utf-8 -*-
"""winrate_review.py — 每周胜率校准（闭环触发器）。

判定标准（预先立约，2026-11-05）：
  · 样本 < 10 个信号日：insufficient，不出结论
  · 滚动 20 日胜率（窗口内全部交易日 < 0.5）：trigger——触发信号层规则审查
  · 否则：healthy / watch（最新一期日超额为负但滚动仍 ≥0.5）

产出：
  · data/live/winrate_review_<日期>.md 审查报告
  · stdout 末行 REVIEW_JSON（weekly_retrain_entry 合并进工件）

审查触发时报告包含：当前胜率 vs 三线基准（50% 掷硬币 / 52.2% top50 样本外 / 54.1% top50 全期）、
与历史基准的差距、逐名次胜率（如样本足够）、触发后的候选动作清单（不自动执行——审查是人机共同决策）。
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd
import duckdb

WS = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(WS, "data", "factor_lab.duckdb")
HIST = os.path.join(WS, "data", "live", "winrate_hist.json")
OUT_DIR = os.path.join(WS, "data", "live")

TRIGGER_DAYS = 20          # 滚动窗口
MIN_DAYS = 10              # 最小样本
PASS_LINE = 0.5223         # top50 样本外及格线
COIN = 0.5


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("""SELECT l.signal_date, l.horizon, l.excess, s.rank
                        FROM live_ledger l LEFT JOIN live_signals s
                          ON l.signal_date = s.signal_date AND l.code = s.code
                        WHERE l.settled AND l.excess IS NOT NULL""").fetchdf()
    hist = json.load(open(HIST)) if os.path.exists(HIST) else {}
    today = datetime.date.today().isoformat()

    h1 = df[df.horizon == 1].copy()
    status, reason = "insufficient", ""
    g = pd.Series(dtype=float)
    if not h1.empty:
        g = h1.groupby("signal_date").excess.mean().sort_index()
        win = (g > 0).astype(float)
        n = len(g)
        roll_all = win.rolling(TRIGGER_DAYS).mean()
        complete_window = n >= TRIGGER_DAYS
        below = roll_all.dropna() < COIN
        if n < MIN_DAYS:
            status = "insufficient"
            reason = f"样本 {n} 天 < {MIN_DAYS}，不出结论"
        elif complete_window and bool(below.iloc[-1]):
            status = "trigger"
            below_streak = 0
            for v in below.iloc[::-1]:
                if v:
                    below_streak += 1
                else:
                    break
            reason = (f"滚动{TRIGGER_DAYS}日胜率 {roll_all.iloc[-1]:.1%} < 50%"
                      f"（连续 {below_streak} 期）→ 触发信号层规则审查")
        elif g.iloc[-1] < 0:
            status = "watch"
            reason = f"最新一期（{g.index[-1].date()}）超额 {g.iloc[-1]:+.2%} 为负，滚动仍 ≥50%"
        else:
            status = "healthy"
            reason = f"滚动{TRIGGER_DAYS}日胜率 {roll_all.iloc[-1] if complete_window else float('nan'):.1%}" if complete_window \
                else f"样本 {n} 天（<{TRIGGER_DAYS}），暂以总胜率 {win.mean():.1%} 观察"

    wr = float((g > 0).mean()) if len(g) else None
    mean_exc = float(g.mean()) if len(g) else None

    # 逐名次胜率（样本足够才出）
    rank_tbl = ""
    rk = None
    if len(h1) >= 30:
        rr = h1.groupby("rank").excess.agg(["mean", lambda s: (s > 0).mean(), "count"])
        rr.columns = ["mean_exc", "win", "n"]
        rr = rr[rr["n"] >= 3]
        if len(rr):
            rk = {int(k): {"win": round(float(v.win), 4), "n": int(v.n),
                           "meanExc": round(float(v.mean_exc), 6)}
                  for k, v in rr.iterrows()}
            rank_tbl = "\n".join(f"| 第 {int(k)} 名 | {v.win:.1%} | {v.mean_exc:+.3%} | {int(v.n)} |"
                                 for k, v in rr.iterrows())

    review = {
        "date": today, "status": status, "reason": reason,
        "liveDays": int(len(g)), "h1WinRate": (round(wr, 4) if wr is not None else None),
        "h1MeanExcess": (round(mean_exc, 6) if mean_exc is not None else None),
        "passLine": PASS_LINE,
        "histTop10Win": hist.get("top10Win"), "histTop50Win": hist.get("top50Win"),
        "oosTop50Win": hist.get("oosTop50Win"),
        "byRank": rk,
    }

    lines = [f"# 胜率校准报告 · {today}", "",
             f"**状态：{status.upper()}** — {reason}", "",
             f"- 实盘信号日：{len(g)} 天，h=1 胜率 {wr if wr is None else f'{wr:.1%}'}"
             f"，均值超额 {mean_exc if mean_exc is None else f'{mean_exc:+.3%}'}",
             f"- 基准：掷硬币 50% / top50 样本外及格线 {PASS_LINE:.1%}"
             f" / top50 全期 {hist.get('top50Win', 0):.1%}", ""]
    if rank_tbl:
        lines += ["## 逐名次胜率（实盘）", "", "| 名次 | 胜率 | 均值超额 | n |", "|---|---|---|---|", rank_tbl, ""]
    if status == "trigger":
        lines += ["## 触发审查（候选动作，不自动执行）", "",
                  "1. 核对数据链路：信号/行情/结算是否异常（先排除工程假摔）",
                  "2. 对比历史基准：当前胜率偏离是水平移动还是正常波动带内",
                  "3. 逐名次检查：是否顶部名次系统性失效（P15 驼峰区是否左移）",
                  "4. 候选决策：a) 维持观察 b) 信号层规则调整（仓位/名次带）c) 新胜率模型立项"
                  "（二分类头，走三层闸门）", ""]
    path = os.path.join(OUT_DIR, f"winrate_review_{today}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    review["reportPath"] = path
    print("REVIEW_JSON: " + json.dumps(review, ensure_ascii=False))


if __name__ == "__main__":
    main()
