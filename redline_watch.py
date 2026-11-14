# -*- coding: utf-8 -*-
"""redline_watch.py — 红线持续观察 → 提前归因复查联动（P23，2026-11-14 预注册）。

立约（判定标准先于任何触发写死，事后不得移动球门）：
  观察源   data/factor_lab.duckdb → factor_alerts 表；
           data/live/patrol_alerts.jsonl（P23 起巡检每日落盘，含零告警日，主日历）
  触发口径 同一 (factor_id, red_line) 组合：
            本周命中 = 最近 5 个巡逻日中 ≥3 天出现
            上周命中 = 再往前 5 个巡逻日中 ≥3 天出现
            两周均命中 → 提前触发一次归因复查（不等每月 1 日月度例行）
  冷却     同一组合两次提前触发间隔 ≥14 天；月度例行不受影响、不计冷却
  留痕     history 记录 trigger=early_redline:<factor>:<red_line>，
           审查报告 md 标题带【提前触发】，与月度例行区分
  边界     只驱动审查报告与人机决策，不改模型结构；
           任何结构改动仍须重走三层闸门 + 候选账本 Bonferroni 预算

输出：末行 WATCH_JSON: {...}（供 weekly_retrain_entry.py 并入周度工件）。
"""
import json, os, subprocess, sys, datetime, glob

WS = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(WS, "data", "factor_lab.duckdb")
EVENTS = os.path.join(WS, "data", "live", "chain_events")
STATE = os.path.join(WS, "data", "live", "redline_watch.json")
REVIEW = os.path.join(WS, "attribution_review.py")

WEEK_N, WEEK_MIN, COOLDOWN_DAYS = 5, 3, 14


def load_alerts():
    """返回 (calendar, alerts)。
    主源 data/live/patrol_alerts.jsonl（P23 起巡检每日落盘，含零告警日）；
    历史回补 factor_alerts 表（仅覆盖有告警的日期；过渡期日历可能缺零告警日，
    窗口缩小只影响命中分母，11-17 起 jsonl 逐日完整后自然消除，已在 P23 留痕）。
    日历 = jsonl 日期 ∪ DB 告警日期 ∪ 近 7 个巡逻事件日，排序取近 10 个。"""
    import duckdb
    by_date = {}  # date -> set((factor, redLine))
    jl = os.path.join(WS, "data", "live", "patrol_alerts.jsonl")
    if os.path.exists(jl):
        for r in json.load(open(jl)):
            by_date.setdefault(r["date"], set()).update(
                (a["factor"], a["redLine"]) for a in r.get("alerts", []))
    con = duckdb.connect(DB, read_only=True)
    for d, f, r in con.execute(
            "SELECT run_at::DATE, factor_id, red_line FROM factor_alerts").fetchall():
        by_date.setdefault(str(d), set()).add((f, r))
    con.close()
    event_dates = [os.path.basename(p).split("__")[0]
                   for p in glob.glob(os.path.join(EVENTS, "*__patrol_done.json"))]
    cal = sorted(set(by_date) | set(event_dates))[-10:]
    alerts = {(d, f, r) for d in cal for f, r in by_date.get(d, set())}
    return cal, alerts


def main():
    today = datetime.date.today().isoformat()
    dates, alerts = load_alerts()
    if len(dates) < 4:  # 日历不足不判定
        print("WATCH_JSON: " + json.dumps({"asOf": today, "status": "insufficient",
                                           "reason": f"日历不足（{len(dates)}）"}, ensure_ascii=False))
        return
    this_w = dates[-WEEK_N:]
    last_w = dates[-2 * WEEK_N:-WEEK_N] if len(dates) >= 2 * WEEK_N else dates[:-WEEK_N]

    state = {}
    if os.path.exists(STATE):
        state = json.load(open(STATE))

    fired, watching = [], []
    combos = {(f, r) for _, f, r in alerts}
    for f, r in sorted(combos):
        n_this = sum((d, f, r) in alerts for d in this_w)
        n_last = sum((d, f, r) in alerts for d in last_w) if last_w else 0
        key = f"{f}:{r}"
        if n_this >= WEEK_MIN and n_last >= WEEK_MIN:
            lf = state.get(key)
            cooled = lf is None or (datetime.date.fromisoformat(today)
                                    - datetime.date.fromisoformat(lf)).days >= COOLDOWN_DAYS
            if cooled:
                env = dict(os.environ, EARLY_TRIGGER=f"early_redline:{key}")
                rr = subprocess.run([sys.executable, REVIEW], capture_output=True,
                                    text=True, timeout=900, env=env)
                try:
                    art = json.loads(rr.stdout)  # attribution_review main 输出整段 JSON
                except Exception:
                    art = {"status": "error", "stderr": rr.stderr[-300:]}
                fired.append({"combo": key, "prevFire": lf, "review": art.get("status"),
                              "headline": art.get("headline", "")})
                state[key] = today
            else:
                watching.append({"combo": key, "prevFire": lf, "cooldownDays": COOLDOWN_DAYS})
        elif n_this >= WEEK_MIN:
            watching.append({"combo": key, "note": f"本周 {n_this} 天命中，待上周确认（{n_last}/5）"})

    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(state, open(STATE, "w"), ensure_ascii=False, indent=1)

    out = {"asOf": today, "calendar": dates, "combosSeen": sorted(f"{f}:{r}" for f, r in combos),
           "fired": fired, "watching": watching,
           "status": "fired" if fired else ("watching" if watching else "clear")}
    print("WATCH_JSON: " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
