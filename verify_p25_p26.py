# -*- coding: utf-8 -*-
"""P25/P26/P27/P30 交付前机器事实源交叉核验；不重训、不改判据。"""
import json
import math
from pathlib import Path

import candidate_ledger as ledger
import wf_pipeline

WS = Path(__file__).resolve().parent
P25 = WS / "data" / "backtest" / "execution_replay"
P25X2 = WS / "data" / "backtest" / "execution_replay_cost_x2"
P26 = WS / "data" / "backtest" / "p26_ivol_residual"
P27 = WS / "data" / "p27_universe_expansion"
P30 = WS / "data" / "p30_execution_aware"
CURRENT = ["p27_csi500_alstm158", "p30_open_gross", "p30_open_net_masked"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (actual, expected)


summary = load(P25 / "summary.json")
health = load(P25 / "health.json")
dsr = load(P25 / "dsr_execution_replay.json")
cost = load(P25 / "execution_cost_decomposition.json")
manifest = load(P25 / "raw_manifest.json")
stress = load(P25X2 / "summary.json")
p26 = load(P26 / "p26_ivol_residual_results.json")

# P25 冻结事实不得因后续候选或压力回放而改变。
assert summary["formal_window"] == ["2023-01-03", "2026-06-30"]
assert summary["overall"]["days"] == dsr["n_obs"] == 843
close(summary["overall"]["annual_net_excess"], 0.05312732638868238)
close(summary["overall"]["net_ir"], 0.6330998987944725)
close(summary["overall"]["mdd"], -0.20192428878904534)
close(summary["overall"]["mean_cash_weight"], 0.06737905127474049)
close(summary["overall"]["mean_cash_weight"], cost["overall"]["cost"]["mean_cash_weight"])
close(dsr["deflated_sharpe_ratio"], 0.365053493444333)
close(dsr["psr_vs_zero"], 0.8801615919991119)
assert dsr["n_trials"] == 9 and dsr["passed"] is False
assert health["passed"] is True and all(health["checks"].values())
assert health["corporate_action_independent_validation"]["events_checked"] == 1598
assert health["corporate_action_events_on_holdings"] == 201
assert health["corporate_action_imputed_on_holdings"] == 0
assert health["limit_fallback_orders"] == 0
assert manifest["instruments"] == 386 and manifest["raw_files"] == 772
assert manifest["failures"] == {} and manifest["actions_refreshed"] is True

# 独立双倍成本目录必须复现相同日期数，且不得覆盖冻结 P25。
assert stress["candidate"] == "p25_frozen_signal_cost_sensitivity"
assert stress["stress"] == 2.0 and stress["frozen_p25_overwritten"] is False
assert stress["overall"]["days"] == 843
close(stress["overall"]["annual_net_excess"], -0.015666136945512166)
close(stress["overall"]["net_ir"], -0.18645078484817365)
close(stress["overall"]["mdd"], -0.24593761757651367)
close(stress["overall"]["total_net_return"], 0.21047482979093646)
close(stress["params"]["slip"], 0.002)
close(stress["params"]["comm"], 0.0005)
close(stress["params"]["stamp"], 0.001)
close(stress["params"]["transfer"], 0.00002)
reproduction_log = (P25X2 / "reproduction_stress1.log").read_text(encoding="utf-8")
assert '"stress": 1.0' in reproduction_log and "Traceback" not in reproduction_log

# P26 保留数据闸失败，不伪造收益或 p 值。
assert p26["decision"] == "rejected_gate1"
assert p26["decision_reason"] == "data_health_failed"
assert p26["primary"]["evaluable"] is False
assert p26["primary"]["missing_years"] == [2015, 2016, 2017, 2018]
assert p26["primary"]["p_one_sided_negative"] is None
assert p26["health"]["no_sealed_label_outcome"] is True
assert p26["health"]["label_end"] == "2026-06-30"
assert p26["health"]["passed"] is False and p26["gate1_passed"] is False
assert p26["ledger_chain_ok_after"] is True

# P27 数据、跨源行情／公司行动与容量诊断必须先于绩效判决成立。
p27_health = load(P27 / "meta" / "data_health.json")
p27_limits = load(P27 / "meta" / "limit_cross_validation.json")
p27_actions = load(P27 / "meta" / "action_cross_validation.json")
p27_capacity = load(P27 / "capacity" / "summary.json")
assert p27_health["passed"] is True
assert p27_limits["mismatch_rows"] == 0
assert p27_limits["up_exact_rate"] == p27_limits["down_exact_rate"] == 1.0
assert p27_actions["passed"] is True
assert load(P27 / "meta" / "execution_data_gates.json")["passed"] is True
assert p27_actions["missing_in_candidate"] == p27_actions["missing_in_reference"] == 0
assert p27_actions["cash_mismatch_over_0_011"] == 0
assert p27_actions["stock_mismatch"] == p27_actions["pay_date_mismatch"] == 0
assert p27_actions["reference_missing_exchange_verified"] >= 0
assert p27_actions["exchange_reference_imputed"] >= 0
official = load(P27 / "meta" / "action_official_evidence_audit.json")
assert official["passed"] is True and official["status"] == "verified"
assert official["events"] == 1 and len(official["records"]) == 1
assert official["records"][0]["instrument"] == "SH601966"
assert official["records"][0]["ex_date"] == "2025-07-10"
assert official["records"][0]["status"] in {"applied", "already_verified"}
close(official["records"][0]["cash_div"], 0.084)
assert p27_actions["official_implementation_combined"] == 1
p27_gates = load(P27 / "meta" / "execution_data_gates.json")
assert p27_gates["passed"] is True and all(p27_gates["checks"].values())
assert p27_gates["official_evidence"]["status"] == "verified"
assert p27_gates["action_download_failures"] == {}
assert p27_gates["missing_action_files"] == []
assert [x["capital"] for x in p27_capacity["capacities"]] == [1000000.0, 10000000.0, 100000000.0]
assert all(x["participation"]["participation_is_diagnostic_only"] is True
           for x in p27_capacity["capacities"])
assert load(P27 / "meta" / "csi1000_data_gate.json")["decision"] == "rejected_gate0_data_health"

verdicts = load(WS / "data" / "p27_p30_final_verdicts.json")
assert list(verdicts) == CURRENT

# 不只信 summary：从逐日收益与逐订单独立重算执行指标、DSR 和年度聚类。
import pandas as pd
import research_verdict as rv
import shadow_execution as se
from finalize_candidates import load_candidate
from p30_execute import require_candidate_data_gates

p25_daily = pd.read_parquet(P25 / "daily_all.parquet")
for candidate in CURRENT:
    v = verdicts[candidate]
    assert v["production_frozen"] is True
    assert v["execution_health"]["passed"] is True
    assert v["multiplicity"]["n_trials"] == 14
    assert v["multiplicity"]["bonferroni_n"] == 7
    close(v["multiplicity"]["bonferroni_threshold"], 0.05 / 7)
    assert v["dsr"]["n_trials"] == 14
    assert v["dsr"]["trial_sr_count"] == 4
    assert v["dsr"]["single_series_variance_fallback"] is False
    assert v["dsr"]["n_obs"] == v["overall"]["days"]
    expected = "pending_independent_review" if not v["failed_gates"] else "rejected_" + v["failed_gates"][0]
    assert v["decision"] == expected
    if candidate.startswith("p30_"):
        # 终审再次从权重历史、标签、配置和评分原文件核验，而非只信执行摘要。
        assert require_candidate_data_gates(candidate, [8, 9, 10, 11])["passed"] is True
    for stress_level in (1, 2):
        execution_summary, daily, orders = load_candidate(candidate, stress_level)
        if candidate.startswith("p30_"):
            assert execution_summary["data_gates"]["passed"] is True
        dates = pd.to_datetime(daily["datetime"])
        assert len(daily) == execution_summary["overall"]["days"] == 843
        assert dates.is_monotonic_increasing and dates.is_unique
        assert dates.min() == pd.Timestamp("2023-01-03")
        assert dates.max() == pd.Timestamp("2026-06-30")
        assert se.health(daily, orders, expected_days=843)["passed"] is True
        assert se.order_stats(orders) == execution_summary["orders"]
        recomputed = se.metric(daily)
        for key in ("annual_net_excess", "net_ir", "mdd", "total_net_return",
                    "mean_one_way_turnover", "mean_cash_weight", "mean_positions"):
            close(recomputed[key], execution_summary["overall"][key])
        if stress_level == 1:
            close(v["overall"]["annual_net_excess"], execution_summary["overall"]["annual_net_excess"])
            base_daily = daily
        else:
            close(v["double_cost_overall"]["annual_net_excess"],
                  execution_summary["overall"]["annual_net_excess"])
            close(execution_summary["params"]["slip"], 0.002)
            close(execution_summary["params"]["comm"], 0.0005)
            close(execution_summary["params"]["stamp"], 0.001)
            close(execution_summary["params"]["transfer"], 0.00002)
    cluster = (rv.annual_cluster_positive(base_daily) if candidate == CURRENT[0]
               else rv.annual_cluster_paired(base_daily, p25_daily))
    close(cluster["p_one_sided_positive"], v["annual_cluster"]["p_one_sided_positive"])
    dsr_check = rv.executable_dsr(base_daily, 14, v["dsr"]["trial_sharpes"])
    close(dsr_check["deflated_sharpe_ratio"], v["dsr"]["deflated_sharpe_ratio"])
    close(dsr_check["psr_vs_zero"], v["dsr"]["psr_vs_zero"])

# 三档资金也必须能从各自逐日序列复算，成交额占比不参与成交回填。
for item in p27_capacity["capacities"]:
    capital = item["capital"]
    target = P27 / "capacity" / f"capital_{int(capital)}"
    daily = pd.read_parquet(target / "daily_all.parquet")
    orders = pd.read_parquet(target / "orders_all.parquet")
    assert se.health(daily, orders, expected_days=843)["passed"] is True
    assert se.order_stats(orders) == item["orders"]
    recomputed = se.metric(daily, capital=capital)
    for key in ("annual_net_excess", "net_ir", "mdd", "total_net_return",
                "mean_one_way_turnover", "mean_cash_weight", "mean_positions"):
        close(recomputed[key], item["overall"][key])

entries = ledger._entries()
assert len(entries) == 14 and ledger.verify_chain() is True
assert entries[-1]["candidate_id"] == "p30_open_net_masked"
assert [e["candidate_id"] for e in entries[-3:]] == CURRENT
threshold, n = ledger.threshold("wf_folds8-11")
assert n == 7
close(threshold, 0.05 / 7)
status = ledger.status()
assert status["budget_used_this_month"] == {"signal": 2, "structure": 2}

audit = wf_pipeline.audit(None, "alstm")
assert audit["canonical"] is True and audit["passed"] is False
assert audit["source"].endswith("data/backtest/execution_replay/dsr_execution_replay.json")
close(audit["deflated_sharpe_ratio"], dsr["deflated_sharpe_ratio"])

print(json.dumps({
    "p25": {"days": 843, "annual_net_excess": summary["overall"]["annual_net_excess"],
            "net_ir": summary["overall"]["net_ir"], "mdd": summary["overall"]["mdd"],
            "mean_cash_weight": summary["overall"]["mean_cash_weight"],
            "dsr": dsr["deflated_sharpe_ratio"], "health_passed": health["passed"],
            "double_cost_annual_net_excess": stress["overall"]["annual_net_excess"]},
    "p26": {"decision": p26["decision"], "reason": p26["decision_reason"],
            "missing_years": p26["primary"]["missing_years"], "p_value": None},
    "p27_p30": {c: {"decision": verdicts[c]["decision"],
                     "annual_net_excess": verdicts[c]["overall"]["annual_net_excess"],
                     "dsr": verdicts[c]["dsr"]["deflated_sharpe_ratio"]} for c in CURRENT},
    "ledger": {"entries": 14, "chain_ok": True, "wf_folds8-11_n": n,
               "next_threshold": threshold, "budget": status["budget_used_this_month"]},
    "canonical_audit": {"canonical": True, "passed": False},
}, ensure_ascii=False, indent=2))
