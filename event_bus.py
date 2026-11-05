# -*- coding: utf-8 -*-
"""event_bus.py — 迷你事件引擎（跨进程文件事件总线）

设计借鉴 vn.py EventEngine 的发布-订阅解耦思想，适配本项目的 cron 多进程链路：
巡检/信号/台账是独立进程，无法共享内存事件队列，因此用文件系统做事件载体。

契约：
- publish(event, payload=None, date=None)  发布事件，幂等（同事件同日覆盖更新 payload）
- wait(event, timeout_s=1800, poll_s=15, date=None)  阻塞等待事件，返回 (ok, payload, elapsed_s)
- wait 超时不抛异常——调用方自行决定降级策略（事件引擎只解耦，不替业务做决策）
- list_events(date=None)  列出当日事件，供巡检对账

事件命名约定：patrol_done / signal_done / ledger_done / hf_minute_done
payload 建议含 finished_at、耗时、关键计数，便于台账对账。

单一事实源：本文件 + data/live/chain_events/ 目录。
"""
import os, json, time, datetime as dt

_WS = os.path.dirname(os.path.abspath(__file__))
EVENTS_DIR = os.path.join(_WS, "data", "live", "chain_events")


def _dir():
    os.makedirs(EVENTS_DIR, exist_ok=True)
    return EVENTS_DIR


def _path(event, date=None):
    d = date or dt.date.today().strftime("%Y-%m-%d")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in event)
    return os.path.join(_dir(), f"{d}__{safe}.json")


def publish(event, payload=None, date=None):
    """发布事件。payload 须可 json 序列化；自动附加 published_at。"""
    body = {"event": event, "date": date or dt.date.today().strftime("%Y-%m-%d"),
            "published_at": dt.datetime.now().isoformat(timespec="seconds"),
            "payload": payload or {}}
    p = _path(event, date)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False)
    os.replace(tmp, p)  # 原子写，避免消费者读到半截 JSON
    return p


def exists(event, date=None):
    return os.path.exists(_path(event, date))


def wait(event, timeout_s=1800, poll_s=15, date=None):
    """阻塞等待事件。返回 (ok, payload, elapsed_s)；超时返回 (False, None, elapsed_s)。"""
    t0 = time.time()
    while True:
        p = _path(event, date)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    body = json.load(f)
                return True, body.get("payload"), time.time() - t0
            except (json.JSONDecodeError, OSError):
                pass  # 写到一半，下一轮再读
        if time.time() - t0 >= timeout_s:
            return False, None, time.time() - t0
        time.sleep(poll_s)


def list_events(date=None):
    d = date or dt.date.today().strftime("%Y-%m-%d")
    out = []
    if os.path.isdir(EVENTS_DIR):
        for fn in sorted(os.listdir(EVENTS_DIR)):
            if fn.startswith(d + "__") and fn.endswith(".json"):
                try:
                    with open(os.path.join(EVENTS_DIR, fn), encoding="utf-8") as f:
                        out.append(json.load(f))
                except (json.JSONDecodeError, OSError):
                    out.append({"event": fn, "error": "unreadable"})
    return out


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "publish":
        publish(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})
        print("published", sys.argv[2])
    elif cmd == "list":
        for e in list_events(sys.argv[2] if len(sys.argv) > 2 else None):
            print(e["event"], e.get("published_at"), json.dumps(e.get("payload", {}), ensure_ascii=False)[:200])
