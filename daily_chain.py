# -*- coding: utf-8 -*-
"""每日收盘后全链巡检（framework_design.md 1.4d 的编排实现）。

链：交易日判断 -> 行情增量 -> 统一灌库 -> 数据质量门禁 -> regime 刷新(规则版+HMM)
    -> 因子监控(strict) -> 一句话健康摘要。

退出码：0 成功（含非交易日跳过/数据已最新）；1 任一环失败。
定时任务只需调用本脚本；手动跑：python daily_chain.py [--force]
"""
import os
import sys
import time
import glob
import subprocess
import datetime as dt

import pandas as pd
import tushare as ts

WS = os.path.dirname(os.path.abspath(__file__))
import sys as _sys, os as _os
for _root in {_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}:
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
from token_loader import tushare_token as _tushare_token
TOKEN = _tushare_token()
DUMP = os.path.join(WS, "qlib-repo", "scripts", "dump_bin.py")
QDIR = os.path.expanduser("~/.qlib/qlib_data/cn_data")
PY = sys.executable


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd, desc, timeout=1200):
    log(f"-> {desc}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=WS, capture_output=True, text=True, timeout=timeout)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-3:])
    if r.returncode != 0:
        log(f"!! {desc} 失败 (exit={r.returncode}, {time.time()-t0:.0f}s)\n{tail}")
        return False, tail
    log(f"   ok ({time.time()-t0:.0f}s)")
    return True, tail


def is_trade_day():
    pro = ts.pro_api(TOKEN)
    today = dt.date.today().strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=today, end_date=today)
    return bool(len(cal) and cal["is_open"].iloc[0] == 1)


def data_fresh():
    """qlib 行情是否已更新到最近交易日。"""
    r = subprocess.run([PY, "factor_lab/data_quality.py"], cwd=WS, capture_output=True, text=True)
    return r.returncode == 0 and "[FAIL]" not in r.stdout


def stale_tail():
    """个股 csv 尾部滞后数：最后日期落后于全库最新日的票数。

    data_quality 的 freshness 取全库 max 日期——297/300 更新即可通过，
    少量个股缺行会被静默跳过重试（2026-11-05 事故：3 只票缺 11-05，
    其中 601456 还是 top1 信号）。这里按个股粒度兜底。
    """
    latest = None
    last_dates = {}
    for f in glob.glob(os.path.join(WS, "data", "csv", "s*.csv")):
        try:
            tail = pd.read_csv(f)["date"].iloc[-1]
            last_dates[os.path.basename(f)] = str(tail)[:10]
        except Exception:
            continue
    if not last_dates:
        return 0, []
    latest = max(last_dates.values())
    latest_ts = pd.Timestamp(latest)
    # 只盯"近期还活跃"的票：滞后超过 10 个自然日的视为退市/长期停牌，
    # 由复牌后自然回补（fetch 窗口覆盖），不计入尾部滞后。
    cutoff = str((latest_ts - pd.Timedelta(days=10)).date())
    stale = sorted([f for f, d in last_dates.items()
                    if d < latest and d >= cutoff])
    if not stale:
        return 0, []
    # 停牌降频：滞后 >3 天的票用一次 tushare 查询确认是否仍停牌——
    # 仍在停牌则移出触发列表（日常零代价），已复牌则保留触发（立即回补）。
    # 2026-11-05 教训：中国稀土/TCL中环 每日强制全量更新多花 ~5 分钟。
    trigger, watching = [], []
    for f in stale:
        age = (latest_ts - pd.Timestamp(last_dates[f])).days
        if age <= 3:
            trigger.append(f)
            continue
        qc = f[:-4]
        ts_code = ("SH" + qc[2:] if qc.startswith("sh") else "SZ" + qc[2:])
        try:
            d = pro_daily(ts_code, last_dates[f].replace("-", ""))
            if d is None or d.empty:
                watching.append(f)  # 仍停牌，观察名单
            else:
                trigger.append(f)   # 已复牌有成交，必须回补
        except Exception:
            trigger.append(f)       # 查询失败保守处理：照常触发
    if watching:
        log(f"长期停牌观察 {len(watching)} 只（>3天，降频不触发更新）: "
            f"{', '.join(w[:8] for w in watching[:5])}")
    return len(trigger), trigger


def pro_daily(ts_code, start_date):
    """tushare 个股日线查询（独立小函数便于异常隔离）。"""
    import tushare as ts
    pro = ts.pro_api(TOKEN)
    return pro.daily(ts_code=ts_code, start_date=start_date,
                     end_date=dt.date.today().strftime("%Y%m%d"))


def main():
    force = "--force" in sys.argv
    log("daily chain start")

    if not is_trade_day():
        log("今日非交易日，跳过")
        return 0
    # 指数日线增量：每日必跑（2 次 API 调用，廉价）。不能只看个股 freshness——
    # 个股新鲜时指数可能停更（2026-11-05 事故：基准停在 11-03 两天）。
    # 指数有新行则必须重灌（输出 NEW_ROWS 标记）。
    ok, tail = run([PY, "update_index_data.py"], "指数日线增量(Tushare)")
    if not ok:
        return 1
    index_new = "NEW_ROWS" in tail
    n_stale, stale = stale_tail()
    if n_stale:
        log(f"个股尾部滞后 {n_stale} 只（{', '.join(s[:8] for s in stale[:5])}{'…' if n_stale > 5 else ''}），强制走更新")
    if not force and data_fresh() and not index_new and not n_stale:
        log("行情已是最新，跳过更新环节（regime/监控照常刷新）")
        updated = True
    else:
        ok, _ = run([PY, "update_qlib_data.py", "0", "850"], "行情增量更新(Tushare)")
        if not ok:
            return 1
        # 指数随个股同目录统一灌库（日历对齐教训：必须单次全量同跑）
        for f in ("sh000300.csv", "sh000905.csv"):
            src, dst = os.path.join(WS, "data", "csv_index", f), os.path.join(WS, "data", "csv", f)
            if os.path.exists(src):
                subprocess.run(["cp", src, dst], check=True)
        ok, _ = run([PY, DUMP, "dump_all", "--data_path", "data/csv", "--qlib_dir", QDIR,
                     "--freq", "day", "--date_field_name", "date",
                     "--include_fields", "open,high,low,close,volume,amount,factor"], "统一灌库(qlib bin)")
        if not ok:
            return 1
        ok, tail = run([PY, "factor_lab/data_quality.py"], "数据质量门禁")
        if not ok:
            log(tail)
            return 1

    # regime 双引擎刷新
    ok, _ = run([PY, "factor_lab/pipeline.py", "regime"], "regime 规则版刷新")
    if not ok:
        return 1
    ok, _ = run([PY, "factor_lab/regime_hmm.py", "run"], "regime HMM 刷新")
    if not ok:
        return 1

    # 因子监控：红线是"市场信号"而非"系统故障"，监控本身跑通即可，告警进摘要
    ok, tail = run([PY, "factor_lab/monitor.py"], "因子监控")
    if not ok:
        return 1
    ok, _ = run([PY, "factor_lab/regime_stats.py"], "regime 条件化因子档案")

    # 一句话摘要
    import duckdb
    con = duckdb.connect(os.path.join(WS, "data", "factor_lab.duckdb"), read_only=True)
    st = con.execute("""SELECT state, prob FROM regime_probs
                        WHERE source='hmm_v2' AND trade_date=(SELECT MAX(trade_date) FROM regime_probs
                        WHERE source='hmm_v2') ORDER BY prob DESC LIMIT 1""").fetchone()
    alerts = con.execute("SELECT COUNT(*) FROM factor_alerts WHERE run_at::DATE = CURRENT_DATE").fetchone()[0]
    con.close()
    log(f"摘要: 当前regime={st[0] if st else '?'}(p={st[1]:.2f}) 今日红线告警={alerts}条"
        f"{' <-- 需要关注' if alerts else ''}")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        # 事件总线：巡检完成是整条 cron 链的上游事件，信号/台账订阅它
        try:
            import event_bus
            event_bus.publish("patrol_done", {
                "finished_at": __import__("datetime").datetime.now().isoformat(timespec="seconds")})
        except Exception as e:
            print(f"event publish 失败（非致命）: {e}", flush=True)
    sys.exit(rc)
