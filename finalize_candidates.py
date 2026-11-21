# -*- coding: utf-8 -*-
"""P27/P30 整批复核、真实 DSR、年度聚类与候选账本终审。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import candidate_ledger as ledger
import research_verdict as rv
import shadow_execution as se

WS = Path(__file__).resolve().parent
P27 = WS / "data" / "p27_universe_expansion"
P30 = WS / "data" / "p30_execution_aware"
P25 = WS / "data" / "backtest" / "execution_replay"
CURRENT = ["p27_csi500_alstm158", "p30_open_gross", "p30_open_net_masked"]
KINDS = {"p27_csi500_alstm158": "structure", "p30_open_gross": "signal",
         "p30_open_net_masked": "signal"}


def batch_multiplicity(entries, current):
    """当前整批一次判决，排除重跑顺序依赖；落账前强制月度预算。"""
    from datetime import date
    current = list(current)
    base = [e for e in entries if e["candidate_id"] not in current]
    base_n = sum(1 for e in base if e["window"] == "wf_folds8-11" and e["kind"] in ledger.COUNTS_TOWARD_N)
    n = base_n + len(current)
    month = date.today().strftime("%Y-%m")
    existing = {e["candidate_id"] for e in entries}
    budget = {}
    for kind, maximum in ledger.MAX_PER_MONTH.items():
        used = sum(1 for e in entries if e["kind"] == kind and e.get("date", "").startswith(month))
        incoming = sum(1 for c in current if c not in existing and KINDS[c] == kind)
        budget[kind] = {"used": used, "incoming": incoming, "max": maximum}
        if used + incoming > maximum:
            raise RuntimeError(f"{kind} 月度预算超限：已用 {used} + 本批 {incoming} > {maximum}")
    return {"base_ledger_entries": len(base), "current_batch_candidates": len(current),
            "n_trials": len(base) + len(current), "n_trials_includes_current_batch": True,
            "bonferroni_base_n": base_n, "bonferroni_n": n,
            "bonferroni_threshold": ledger.ALPHA / max(n, 1), "monthly_budget": budget}


def leave_one_year_out(daily):
    d = daily.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    yearly = d.groupby(d["datetime"].dt.year)["excess"].mean().reindex(rv.YEARS)
    if yearly.isna().any():
        return {"all_leave_one_out_means_positive": False, "min_leave_one_out_mean": None,
                "leave_one_out_means": {}}
    values = {str(y): float(yearly.drop(y).mean()) for y in rv.YEARS}
    return {"all_leave_one_out_means_positive": bool(min(values.values()) > 0),
            "min_leave_one_out_mean": float(min(values.values())),
            "leave_one_out_means": values}


def deployment_gates(summary, stress, daily, dsr, cluster_p, bonferroni_threshold, execution_health):
    overall = summary["overall"]
    yearly = daily.copy()
    yearly["datetime"] = pd.to_datetime(yearly["datetime"])
    means = yearly.groupby(yearly["datetime"].dt.year)["excess"].mean().reindex(rv.YEARS)
    loyo = leave_one_year_out(daily)
    return {
        "annual_net_excess_positive": bool(overall["annual_net_excess"] > 0),
        "net_ir_gt_0_8": bool(overall["net_ir"] > 0.8),
        "dsr_gte_0_95": bool(dsr >= 0.95),
        "at_least_3_of_4_years_positive": bool((means > 0).sum() >= 3),
        "2025_2026_not_both_negative": bool(not (means.loc[2025] < 0 and means.loc[2026] < 0)),
        "double_cost_positive": bool(stress["overall"]["annual_net_excess"] > 0),
        "bonferroni_increment": bool(cluster_p is not None and cluster_p < bonferroni_threshold),
        "no_single_year_driver": bool(loyo["all_leave_one_out_means_positive"]),
        "executable_health": bool(execution_health),
    }


def paths(candidate, stress=1):
    if candidate == "p27_csi500_alstm158":
        root = P27 / ("execution" if stress == 1 else f"execution_cost_x{stress:g}")
        return (root / "summary.json", root / "capital_1000000" / "daily_all.parquet",
                root / "capital_1000000" / "orders_all.parquet")
    root = P30 / ("execution" if stress == 1 else f"execution_cost_x{stress:g}") / candidate
    return root / "summary.json", root / "daily_all.parquet", root / "orders_all.parquet"


def load_candidate(candidate, stress=1):
    summary_path, daily_path, orders_path = paths(candidate, stress)
    if not summary_path.exists() or not daily_path.exists():
        raise FileNotFoundError(f"{candidate} 执行产物不完整：{summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    daily = pd.read_parquet(daily_path)
    if orders_path.exists():
        orders = pd.read_parquet(orders_path)
    else:
        root = orders_path.parent
        orders = pd.concat([pd.read_parquet(p) for p in sorted(root.glob("orders_fold*.parquet"))],
                           ignore_index=True)
    return summary, daily, orders


def daily_sr(daily):
    x = daily["excess"].to_numpy(dtype=float)
    return float(np.mean(x) / np.std(x, ddof=1))


def finalize(record=False):
    entries = ledger._entries()
    multiplicity = batch_multiplicity(entries, CURRENT)
    p25_daily = pd.read_parquet(P25 / "daily_all.parquet")
    loaded = {c: load_candidate(c) for c in CURRENT}
    stressed = {c: load_candidate(c, 2)[0] for c in CURRENT}
    # 全部真实可执行配置共同估计 SR 方差；数据闸失败项绝不补 0。
    trial_sharpes = [daily_sr(p25_daily)] + [daily_sr(loaded[c][1]) for c in CURRENT]
    verdicts = {}
    for candidate in CURRENT:
        summary, daily, orders = loaded[candidate]
        health = se.health(daily, orders)
        cluster = (rv.annual_cluster_positive(daily) if candidate == "p27_csi500_alstm158"
                   else rv.annual_cluster_paired(daily, p25_daily))
        dsr = rv.executable_dsr(daily, multiplicity["n_trials"], trial_sharpes)
        gates = deployment_gates(summary, stressed[candidate], daily, dsr["deflated_sharpe_ratio"],
                                 cluster["p_one_sided_positive"], multiplicity["bonferroni_threshold"],
                                 health["passed"])
        failed = [k for k, passed in gates.items() if not passed]
        decision = "pending_independent_review" if not failed else "rejected_" + failed[0]
        result = {
            "candidate": candidate, "kind": KINDS[candidate], "decision": decision,
            "failed_gates": failed, "gates": gates, "overall": summary["overall"],
            "double_cost_overall": stressed[candidate]["overall"], "orders": summary["orders"],
            "annual_cluster": cluster, "leave_one_year_out": leave_one_year_out(daily),
            "dsr": dsr, "multiplicity": multiplicity, "execution_health": health,
            "folds": summary["folds"], "production_frozen": True,
            "promotion_requires": "一次性 holdout 与独立部署复核；本判决不解除生产冻结",
        }
        if candidate == "p27_csi500_alstm158":
            capacity_path = P27 / "capacity" / "summary.json"
            if not capacity_path.exists():
                raise FileNotFoundError("P27 三档容量诊断尚未完成，禁止终审")
            result["capacity"] = json.loads(capacity_path.read_text(encoding="utf-8"))
            result["participation"] = summary.get("participation")
            out = P27 / "metrics" / "final_verdict.json"
        else:
            result["participation"] = summary.get("participation")
            out = P30 / "metrics" / f"final_{candidate}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        verdicts[candidate] = result
        if record:
            ledger.record(candidate, KINDS[candidate], "wf_folds8-11", decision,
                          cluster["p_one_sided_positive"],
                          {"preregistration": "p27_universe_expansion.md" if candidate.startswith("p27") else "p30_execution_aware.md",
                           "batch_bonferroni_threshold": multiplicity["bonferroni_threshold"],
                           "batch_bonferroni_n": multiplicity["bonferroni_n"],
                           "dsr_n_trials": multiplicity["n_trials"]},
                          "失败闸门=" + (",".join(failed) if failed else "无；仅待独立复核，不解禁生产"))
    combined = WS / "data" / "p27_p30_final_verdicts.json"
    combined.write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({c: {"decision": v["decision"], "failed_gates": v["failed_gates"],
                          "annual_net_excess": v["overall"]["annual_net_excess"],
                          "net_ir": v["overall"]["net_ir"], "dsr": v["dsr"]["deflated_sharpe_ratio"],
                          "cluster_p": v["annual_cluster"]["p_one_sided_positive"]}
                     for c, v in verdicts.items()}, ensure_ascii=False, indent=2), flush=True)
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()
    finalize(args.record)


if __name__ == "__main__":
    main()
