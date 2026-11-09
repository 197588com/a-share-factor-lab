# -*- coding: utf-8 -*-
"""holdout_guard.py — U1：一次性终审 holdout 封存守卫。

封存语义（final_review_audit.md U1）：
  2026-07-01（含）之后的已实现收益日期 = 一次性终审保留区。
  适用范围：候选评估 / 终审 / 因子准入 / 排名分解 / 任何模型选择类实验。
  不适用：日常信号生产与 sim_v2 台账结算（运维跟踪，不构成候选判决）。
  开封条件（预定）：模型结构级改动的毕业终审，开封一次后不再关闭。

用法：
  from holdout_guard import assert_window_allowed
  assert_window_allowed("2023-01-01", "2026-06-30")   # 允许
  assert_window_allowed("2026-01-01", "2026-09-30")   # 抛 SealedHoldoutError
"""
from datetime import date

HOLDOUT_SEALED_FROM = date(2026, 7, 1)


class SealedHoldoutError(RuntimeError):
    pass


def _d(s):
    return date.fromisoformat(str(s)[:10])


def is_sealed(d):
    return _d(d) >= HOLDOUT_SEALED_FROM


def assert_window_allowed(start, end, purpose="candidate"):
    """窗口 [start, end] 不得触碰封存区。

    purpose="candidate"（默认）：抛异常，硬拦截。
    purpose="audit"：不改判任何候选、只重算既有产物上的既有诊断时可放行，
        但必须显式传入并打印警告；新判决一律不得用 audit 模式。
    """
    if not (is_sealed(end) or is_sealed(start)):
        return
    msg = (f"窗口 {start}~{end} 触碰一次性终审保留区（封存自 {HOLDOUT_SEALED_FROM}）。")
    if purpose == "audit":
        print(f"[holdout_guard:audit] {msg} 本次运行为诊断复算，不产生新判决。")
        return
    raise SealedHoldoutError(
        msg + "该数据只能用于日常信号/台账运维，或按 U1 预定的开封条件使用。")


def allowed_test_ceiling():
    """当前候选评估允许的最新日期（封存日前一日）。"""
    return (HOLDOUT_SEALED_FROM.toordinal() - 1)


if __name__ == "__main__":
    assert_window_allowed("2023-01-01", "2026-06-30")
    try:
        assert_window_allowed("2026-01-01", "2026-09-30")
        raise SystemExit("guard broken: sealed window passed")
    except SealedHoldoutError as e:
        print(f"guard ok: {e}")
    print(f"sealed from {HOLDOUT_SEALED_FROM}; test ceiling ordinal {allowed_test_ceiling()}")
