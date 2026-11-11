# -*- coding: utf-8 -*-
"""月度超额归因审查入口：调用项目工作区的 attribution_review.run_review()。"""
import json, os, sys

WS = "/Users/yanglin/Documents/kimi/tasks/2026-11-03/13-01-52-a8ddd0ae"


def run(ctx):
    sys.path.insert(0, WS)
    try:
        import attribution_review
        art, _ = attribution_review.run_review()
        return {"artifact": art}
    except Exception as e:  # 失败也出 artifact，状态 error，小组件可显示
        return {"artifact": {"asOf": str(__import__("datetime").date.today()),
                             "status": "error",
                             "headline": f"审查运行失败：{type(e).__name__}: {e}"}}
