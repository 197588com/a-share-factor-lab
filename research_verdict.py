# -*- coding: utf-8 -*-
"""P27/P30 统一统计证据：可执行收益 DSR 与自然年聚类检验。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

WS = Path(__file__).resolve().parent
YEARS = (2023, 2024, 2025, 2026)


def _year_cluster(values, days):
    values = values.reindex(YEARS)
    days = days.reindex(YEARS).fillna(0)
    missing = [int(y) for y in YEARS if pd.isna(values.loc[y])]
    if missing or bool((days < 30).any()):
        return {"evaluable": False, "years": int(values.notna().sum()), "missing_years": missing,
                "each_year_at_least_30_days": bool((days >= 30).all()),
                "mean": None, "std": None, "t": None, "df": None,
                "p_one_sided_positive": None, "positive_years": None}
    x = values.to_numpy(dtype=float)
    stat = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-15))
    return {"evaluable": True, "years": len(x), "missing_years": [],
            "each_year_at_least_30_days": True, "mean": float(x.mean()),
            "std": float(x.std(ddof=1)), "t": stat, "df": len(x) - 1,
            "p_one_sided_positive": float(student_t.sf(stat, len(x) - 1)),
            "positive_years": int((x > 0).sum())}


def annual_cluster_positive(daily):
    """相对自身基准的年度净超额；不把不同宽基的超额强行配对。"""
    d = daily.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d["year"] = d["datetime"].dt.year
    grouped = d.groupby("year")["excess"]
    return _year_cluster(grouped.mean(), grouped.size())


def annual_cluster_paired(candidate, reference):
    """候选减参考的同日超额差，以四个自然年作为独立聚类。"""
    c = candidate[["datetime", "excess"]].copy()
    r = reference[["datetime", "excess"]].copy()
    c["datetime"] = pd.to_datetime(c["datetime"])
    r["datetime"] = pd.to_datetime(r["datetime"])
    joined = c.merge(r, on="datetime", how="inner", suffixes=("_candidate", "_reference"),
                     validate="one_to_one")
    if joined.empty:
        return _year_cluster(pd.Series(dtype=float), pd.Series(dtype=float))
    joined["delta"] = joined["excess_candidate"] - joined["excess_reference"]
    joined["year"] = joined["datetime"].dt.year
    grouped = joined.groupby("year")["delta"]
    out = _year_cluster(grouped.mean(), grouped.size())
    out["paired_days"] = int(len(joined))
    return out


def executable_dsr(daily, n_trials, trial_sharpes):
    """只用可执行日度超额及真实试验 SR；禁止单序列渐近方差回退。"""
    sharpes = [float(x) for x in trial_sharpes if np.isfinite(x)]
    if n_trials < len(sharpes) or len(sharpes) < 2:
        raise ValueError("至少需要两个真实可执行试验 SR，且 n_trials 不得小于 SR 数")
    excess = pd.to_numeric(daily["excess"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(excess) < 30 or not np.isfinite(excess).all():
        raise ValueError("DSR 日度超额序列不足或包含非有限值")
    # Bailey & Lopez de Prado (2014) DSR；项目内实现，避免依赖未跟踪外部目录。
    observed = float(excess.mean() / excess.std(ddof=1))
    centered = excess - excess.mean()
    scale = excess.std(ddof=0)
    if not np.isfinite(observed) or scale == 0:
        raise ValueError("DSR 观测 SR 非有限或超额序列零方差")
    skew = float(np.mean((centered / scale) ** 3))
    kurtosis = float(np.mean((centered / scale) ** 4))  # 非超额峰度，正态=3
    sr_variance = float(np.var(np.asarray(sharpes, dtype=float), ddof=1))
    gamma = 0.5772156649015328606
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    sr0 = float(np.sqrt(max(sr_variance, 0.0)) * ((1 - gamma) * z1 + gamma * z2))
    denominator = max(1 - skew * observed + (kurtosis - 1) / 4 * observed ** 2, 1e-12)
    standard_error = float(np.sqrt(denominator / (len(excess) - 1)))
    psr_zero = float(norm.cdf((observed - 0.0) / standard_error))
    deflated = float(norm.cdf((observed - sr0) / standard_error))
    result = {"observed_sharpe": observed, "deflated_benchmark_sr0": sr0,
              "psr_vs_zero": psr_zero, "deflated_sharpe_ratio": deflated,
              "n_obs": int(len(excess)), "n_trials": int(n_trials),
              "skew": skew, "kurtosis": kurtosis, "passed": bool(deflated >= 0.95)}
    result.update({
        "series": "executable daily excess return; folds 8-11; formal window only",
        "observed_sr_ann": float(result["observed_sharpe"] * np.sqrt(244)),
        "trial_sharpes": sharpes, "trial_sr_count": len(sharpes),
        "trial_sr_coverage": float(len(sharpes) / n_trials),
        "trial_sr_coverage_note": "仅具有真实可执行收益的配置进入 SR 方差；数据闸失败项不伪造为 0。",
        "single_series_variance_fallback": False,
        "n_trials_includes_current_candidate": True,
    })
    return result
