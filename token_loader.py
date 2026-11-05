# -*- coding: utf-8 -*-
"""token_loader.py — 凭证统一加载（脱敏改造 2026-11-06）。

优先级：环境变量 TUSHARE_TOKEN → 项目根 .env 文件。
仓库内不再硬编码任何 token；.env 已 gitignore。
"""
import os

_ROOT_HINTS = (".",)


def _read_env_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln.startswith("TUSHARE_TOKEN="):
                    return ln.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def tushare_token():
    t = os.environ.get("TUSHARE_TOKEN")
    if t:
        return t
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), here,
              os.path.dirname(here), os.path.join(here, "..")):
        t = _read_env_file(os.path.join(d, ".env"))
        if t:
            return t
    return ""
