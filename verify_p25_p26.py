# -*- coding: utf-8 -*-
"""P25/P26 交付前的机器事实源交叉核验；不重训、不改判据。"""
import json
import math
from pathlib import Path

import candidate_ledger as ledger
import wf_pipeline

WS = Path(__file__).resolve().parent
P25 = WS / "data" / "backtest" / "execution_replay"
P26 = WS / "data" / "backtest" / "p26_ivol_residual"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (actual, expected)


summary = load(P25 / "summary.json")
health = load(P25 / "health.json")
dsr = load(P25 / "dsr_execution_replay.json")
cost = load(P25 / "execution_cost_decomposition.json")
manifest = load(P25 / "raw_manifest.json")
p26 = load(P26 / "p26_ivol_residual_results.json")

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

assert p26["decision"] == "rejected_gate1"
assert p26["decision_reason"] == "data_health_failed"
assert p26["primary"]["evaluable"] is False
assert p26["primary"]["missing_years"] == [2015, 2016, 2017, 2018]
assert p26["primary"]["p_one_sided_negative"] is None
assert p26["health"]["no_sealed_label_outcome"] is True
assert p26["health"]["label_end"] == "2026-06-30"
assert p26["health"]["passed"] is False and p26["gate1_passed"] is False
assert p26["ledger_chain_ok_after"] is True
assert len(ledger._entries()) == 10 and ledger.verify_chain() is True
assert ledger._entries()[-1]["candidate_id"] == "p26_ivol_alpha158_residual"
assert ledger._entries()[-1]["p_value"] is None
threshold, n = ledger.threshold("wf_folds0-11")
assert n == 5
close(threshold, 0.01)

audit = wf_pipeline.audit(None, "alstm")
assert audit["canonical"] is True and audit["passed"] is False
assert audit["source"].endswith("data/backtest/execution_replay/dsr_execution_replay.json")
close(audit["deflated_sharpe_ratio"], dsr["deflated_sharpe_ratio"])

print(json.dumps({
    "p25": {"days": 843, "annual_net_excess": summary["overall"]["annual_net_excess"],
            "net_ir": summary["overall"]["net_ir"], "mdd": summary["overall"]["mdd"],
            "mean_cash_weight": summary["overall"]["mean_cash_weight"],
            "dsr": dsr["deflated_sharpe_ratio"], "health_passed": health["passed"]},
    "p26": {"decision": p26["decision"], "reason": p26["decision_reason"],
            "missing_years": p26["primary"]["missing_years"], "p_value": None},
    "ledger": {"entries": 10, "chain_ok": True, "wf_folds0-11_n": n, "next_threshold": threshold},
    "canonical_audit": {"canonical": True, "passed": False},
}, ensure_ascii=False, indent=2))
