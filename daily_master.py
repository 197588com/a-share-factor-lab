# -*- coding: utf-8 -*-
"""daily_master.py — 每日收盘总流水线（一键贯穿全部日频任务）。

把原先 5 个独立定时任务（巡检→信号→模拟账户→台账→胜率→分钟自录）按依赖顺序
在单次运行内依次执行，产出一个汇总工件：用户只需要看「每日总览」一张卡。

设计要点：
  - 各阶段复用现有 automation assets 入口的 run(ctx)，单一事实源不复制逻辑；
  - 阶段间沿用 event_bus 事件（patrol_done / signal_done / hf_minute_done），
    各入口自带 wait+超时降级，master 不另造调度假设；
  - 单阶段失败不阻断后续（记录 error 继续），汇总里显性列出异常；
  - 周重训（周五）频率不同，保持独立任务，不并入本流水线。
"""
import os
import sys
import json
import traceback
import importlib.util
import datetime as dt

WS = "/Users/yanglin/Documents/kimi/tasks/2026-11-03/13-01-52-a8ddd0ae"
BP = os.path.expanduser(
    "~/Library/Application Support/kimi-desktop/daimon-share/daimon/agents/main/blueprint/automations")

# (阶段名, 入口文件) —— 顺序即依赖顺序
STAGES = [
    ("patrol", f"{BP}/automation_ae1835c7-6492-4bb4-a24f-0db03857e3f9/assets/daily_patrol.py"),
    ("signal", f"{BP}/automation_371db7f2-093e-4009-afc9-3130aaa0f594/assets/automation.py"),
    ("sim",    f"{BP}/automation_014f7dbc-f98c-40a4-bf3c-88a3b0043adc/assets/sim_settle.py"),
    ("ledger", f"{BP}/automation_e0ec2dfc-c489-44ab-9a44-903dea7e822f/assets/automation.py"),
    ("winrate", f"{BP}/automation_1ed02c8e-bd1b-4527-87f1-a405c103bba8/assets/winrate_meter.py"),
    ("hf",     f"{BP}/automation_d49608be-44b6-4282-abc4-9fac03848e19/assets/automation.py"),
]

ORDER = [s[0] for s in STAGES]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(f"entry_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_stage(name, path):
    t0 = dt.datetime.now()
    try:
        mod = _load(name, path)
        r = mod.run(None) or {}
        art = r.get("artifact", r) if isinstance(r, dict) else r
        return {"ok": True, "sec": int((dt.datetime.now() - t0).total_seconds()),
                "artifact": art}
    except Exception as e:
        return {"ok": False, "sec": int((dt.datetime.now() - t0).total_seconds()),
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1200:]}


def build_final(sections):
    """汇总段：一句话总结 + 异常清单。"""
    anomalies = []
    sig = (sections.get("signal") or {}).get("artifact") or {}
    pat = (sections.get("patrol") or {}).get("artifact") or {}
    sim = (sections.get("sim") or {}).get("artifact") or {}
    wr = (sections.get("winrate") or {}).get("artifact") or {}
    for name in ORDER:
        s = sections.get(name) or {}
        if not s.get("ok"):
            anomalies.append(f"{name} 阶段失败：{s.get('error', '?')}")
    if sig.get("stale"):
        anomalies.append(f"信号迟滞：{sig.get('signalDate')}（今日应有新信号）")
    if pat and not pat.get("qualityPass", True):
        anomalies.append("数据质量检查未通过")
    alerts = pat.get("alerts") or []
    if alerts:
        anomalies.append(f"因子红线告警 {len(alerts)} 条：" +
                         "、".join(a.get("factor", "?") for a in alerts[:3]))
    blk = sim.get("blockedOrders") or 0
    if blk:
        anomalies.append(f"模拟账户 {blk} 笔订单受阻（涨停/跳空）")

    picks = sig.get("picks") or []
    top3 = "、".join(f"{p.get('name') or p.get('code')}" for p in picks[:3]) or "—"
    excess = sim.get("excessRet")
    line = (f"市场 {pat.get('regime', '?')}（p={pat.get('regimeProb', 0):.2f}）｜"
            f"信号 {sig.get('signalDate', '?')} top3：{top3}｜"
            f"模拟净值 {sim.get('nav', 0):,.0f}"
            + (f"（超额 {excess * 100:+.2f}%）" if excess is not None else "")
            + f"｜h1 胜率 {((wr.get('live') or {}).get('h1') or {}).get('winRate', 0) * 100:.1f}%")
    return {"headline": line, "anomalies": anomalies,
            "healthy": len(anomalies) == 0}


def run(ctx):
    os.chdir(WS)
    sys.path.insert(0, WS)
    sections = {}
    for name, path in STAGES:
        print(f"[master] stage={name} start", flush=True)
        sections[name] = run_stage(name, path)
        ok = sections[name]["ok"]
        print(f"[master] stage={name} ok={ok} sec={sections[name]['sec']}", flush=True)
        if not ok:
            print(f"[master] stage={name} error: {sections[name]['error']}", flush=True)

    # 工件瘦身：trace 只留在日志，不进工件（binding 传输体积）
    out_sections = {}
    for name in ORDER:
        s = sections[name]
        out_sections[name] = {k: v for k, v in s.items() if k != "trace"}
    final = build_final(sections)
    artifact = {
        "summary": ("每日总流水线完成：" + final["headline"]
                    + (f"；异常 {len(final['anomalies'])} 项" if final["anomalies"] else "；全链路健康")),
        "ranAt": dt.datetime.now().isoformat(timespec="seconds"),
        "pipelineVer": "master_v1",
        "stages": {n: {"ok": sections[n]["ok"], "sec": sections[n]["sec"]} for n in ORDER},
        **out_sections,
        "final": final,
    }
    # 事件总线：总流水线完成（后续周重训等可订阅）
    try:
        import event_bus
        event_bus.publish("daily_master_done", {
            "finished_at": artifact["ranAt"],
            "healthy": final["healthy"],
            "anomalies": len(final["anomalies"])})
    except Exception as e:
        print(f"daily_master_done 发布失败（非致命）: {e}", flush=True)
    return {"artifact": artifact}


if __name__ == "__main__":
    print(json.dumps(run(None)["artifact"]["final"], ensure_ascii=False, indent=2))
