# -*- coding: utf-8 -*-
"""DSR 实盘复算入口：调用项目工作区的 dsr_live.run_review()。"""
import sys
import datetime

WS = "/Users/yanglin/Documents/kimi/tasks/2026-11-03/13-01-52-a8ddd0ae"


def run(ctx):
    sys.path.insert(0, WS)
    try:
        import dsr_live
        art, _ = dsr_live.run_review()
        return {"artifact": art}
    except Exception as e:
        return {"artifact": {"asOf": str(datetime.date.today()),
                             "status": "error",
                             "headline": f"复算运行失败：{type(e).__name__}: {e}"}}
