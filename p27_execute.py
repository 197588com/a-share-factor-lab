# -*- coding: utf-8 -*-
"""P27 中证500可执行回灌、双倍成本与三档资金容量诊断。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import execution_replay as er
import p27_pipeline as pp
import shadow_execution as se
import wf_pipeline as wf

WS = Path(__file__).resolve().parent
ROOT = WS / "data" / "p27_universe_expansion"
CAPITALS = (1_000_000.0, 10_000_000.0, 100_000_000.0)


def attach_participation(orders, store):
    """成交额占比只作容量诊断，不事后改变既定成交结果。"""
    enriched = orders.copy()
    enriched["daily_amount"] = np.nan
    enriched["participation"] = np.nan
    if enriched.empty:
        return enriched, {"filled_with_amount": 0, "participation_p95": None,
                          "participation_max": None, "participation_over_5pct": None,
                          "participation_is_diagnostic_only": True}
    for idx, order in enriched[enriched["status"].eq("FILLED")].iterrows():
        row = store.row(order["code"], pd.Timestamp(order["exec_date"]))
        if row is None or not pd.notna(row.get("amount")) or float(row["amount"]) <= 0:
            continue
        amount = float(row["amount"])
        enriched.loc[idx, "daily_amount"] = amount
        enriched.loc[idx, "participation"] = float(order["value"]) / amount
    values = enriched["participation"].dropna().to_numpy(dtype=float)
    stats = {
        "filled_with_amount": int(len(values)),
        "participation_p95": float(np.percentile(values, 95)) if len(values) else None,
        "participation_max": float(values.max()) if len(values) else None,
        "participation_over_5pct": float((values > 0.05).mean()) if len(values) else None,
        "participation_is_diagnostic_only": True,
    }
    return enriched, stats


def corporate_action_health(actions):
    """持仓公司行动必须可核验；送转参考价差额仅诊断，纯现金冲突仍失败。"""
    good = {"baostock_implementation", "exchange_reference_imputed", "baostock_implementation_stock_plan",
            "official_implementation_combined"}
    if actions is None or actions.empty:
        return {"passed": True, "events": 0, "bad_source_events": 0, "imputed_events": 0,
                "stock_plan_events": 0, "bad_sources": {}}
    sources = actions.get("cash_source", pd.Series("legacy_unknown", index=actions.index)).fillna("legacy_unknown")
    stock_plan = sources.eq("baostock_implementation_stock_plan")
    bad = ~sources.isin(good)
    gaps = pd.to_numeric(actions.get("cash_reconciliation_gap"), errors="coerce").abs()
    bad |= gaps.gt(0.011) & ~stock_plan
    counts = {str(k): int(v) for k, v in sources[bad].value_counts().items()}
    return {"passed": bool(not bad.any()), "events": int(len(actions)),
            "bad_source_events": int(bad.sum()), "imputed_events": int(sources.eq("exchange_reference_imputed").sum()),
            "stock_plan_events": int(stock_plan.sum()), "bad_sources": counts}


def require_execution_data_gates(instruments, fold_ids, root=ROOT, action_validator=None):
    """执行前强制重验数据；任何缺失、陈旧或冲突都 fail closed。"""
    import p27_cross_validate as cv
    meta = root / "meta"
    health = json.loads((meta / "data_health.json").read_text(encoding="utf-8"))
    limits = json.loads((meta / "limit_cross_validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((meta / "actions_manifest.json").read_text(encoding="utf-8"))
    coverage = {i: json.loads((root / "metrics" / f"coverage_fold{i}.json").read_text(encoding="utf-8"))
                for i in fold_ids}
    instruments = sorted({str(x).upper() for x in instruments})
    missing_actions = [x for x in instruments if not (root / "actions" / f"{x.lower()}.parquet").exists()]
    validator = action_validator or cv.validate_actions
    official_audit = {"status": "not_run_injected_validator"}
    if action_validator is None:
        official_audit = cv.apply_official_evidence(
            candidate_dir=root / "actions", evidence_path=meta / "action_official_evidence.json")
        (meta / "action_official_evidence_audit.json").write_text(
            json.dumps(official_audit, ensure_ascii=False, indent=2), encoding="utf-8")
        action_summary, action_detail = validator(candidate_dir=root / "actions", raw_dir=root / "raw_daily")
    else:
        action_summary, action_detail = validator(candidate_dir=root / "actions")
    (meta / "action_cross_validation.json").write_text(
        json.dumps(action_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    action_detail.to_parquet(meta / "action_cross_validation_detail.parquet", index=False)
    checks = {
        "raw_data_health": bool(health.get("passed")),
        "limit_cross_validation": bool(limits.get("mismatch_rows") == 0),
        "action_download_complete": bool(not manifest.get("failures") and not missing_actions),
        "fold_coverage": bool(all(x.get("passed") for x in coverage.values())),
        "action_cross_validation": bool(action_summary.get("passed")),
    }
    result = {"passed": bool(all(checks.values())), "checks": checks,
              "score_instruments": len(instruments), "missing_action_files": missing_actions,
              "action_download_failures": manifest.get("failures"), "official_evidence": official_audit,
              "actions": action_summary}
    (meta / "execution_data_gates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise AssertionError(result)
    return result


def run(folds, stress=1.0, capacity=False):
    capitals = CAPITALS if capacity else (1_000_000.0,)
    if capacity:
        out = ROOT / "capacity"
    else:
        out = ROOT / ("execution" if stress == 1 else f"execution_cost_x{stress:g}")
    out.mkdir(parents=True, exist_ok=True)
    scores_all = pd.concat([pd.read_parquet(pp.SCORES / f"csi500_alstm_fold{i}.parquet",
                                            columns=["instrument"]) for i in folds], ignore_index=True)
    require_execution_data_gates(scores_all["instrument"].unique(), folds)
    store = se.P27RawStore(scores_all["instrument"].str.upper().unique())
    originals = {k: er.PARAMS[k] for k in ("slip", "comm", "stamp", "transfer")}
    for k in originals:
        er.PARAMS[k] = originals[k] * stress
    fold_config = {f["id"]: f for f in wf.generate_folds(wf.load_config())}
    try:
        by_capital = {c: {"daily": [], "orders": [], "folds": []} for c in capitals}
        for fold_id in folds:
            fold = fold_config[fold_id]
            baseline = se.index_baseline("SH000905", ROOT / "csv", fold["test"][0],
                                         min(fold["test"][1], er.PARAMS["formal_end"]))
            raw_score = pd.read_parquet(pp.SCORES / f"csi500_alstm_fold{fold_id}.parquet")
            score, events = se.prepare_scores_qlib(raw_score, baseline, pp.PROVIDER)
            for capital in capitals:
                daily, orders, actions = se.run_account(score, baseline, store, capital=capital)
                orders, participation = attach_participation(orders, store)
                health = se.health(daily, orders, len(baseline))
                action_health = corporate_action_health(actions)
                health["corporate_action_health"] = action_health
                health["passed"] = bool(health["passed"] and action_health["passed"])
                if not health["passed"]:
                    raise AssertionError(health)
                daily["fold"] = fold_id
                target = out / f"capital_{int(capital)}"
                target.mkdir(parents=True, exist_ok=True)
                daily.to_parquet(target / f"daily_fold{fold_id}.parquet", index=False)
                orders.to_parquet(target / f"orders_fold{fold_id}.parquet", index=False)
                events.to_parquet(target / f"mask_events_fold{fold_id}.parquet", index=False)
                actions.to_parquet(target / f"actions_fold{fold_id}.parquet", index=False)
                metrics = se.metric(daily, capital=capital)
                order_stats = se.order_stats(orders)
                by_capital[capital]["folds"].append({
                    "fold": fold_id, "year": int(daily["datetime"].dt.year.iloc[0]),
                    **metrics, **order_stats, **participation,
                    "corporate_action_health": action_health,
                })
                by_capital[capital]["daily"].append(daily)
                by_capital[capital]["orders"].append(orders)
                print(f"P27 fold{fold_id} capital={capital:.0f} x{stress}: "
                      f"ann={metrics['annual_net_excess']*100:.2f}% IR={metrics['net_ir']:.3f} "
                      f"MDD={metrics['mdd']*100:.2f}%", flush=True)
        summaries = []
        for capital in capitals:
            bucket = by_capital[capital]
            daily = pd.concat(bucket["daily"], ignore_index=True).sort_values("datetime").reset_index(drop=True)
            orders = pd.concat(bucket["orders"], ignore_index=True)
            target = out / f"capital_{int(capital)}"
            daily.to_parquet(target / "daily_all.parquet", index=False)
            orders.to_parquet(target / "orders_all.parquet", index=False)
            _, participation = attach_participation(orders, store)
            summary = {"capital": capital, "overall": se.metric(daily, capital=capital),
                       "folds": bucket["folds"], "orders": se.order_stats(orders),
                       "participation": participation}
            (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            pd.DataFrame(bucket["folds"]).to_csv(target / "fold_summary.csv", index=False)
            summaries.append(summary)
        result = {"candidate": "p27_csi500_alstm158", "stress": stress,
                  "params": {k: er.PARAMS[k] for k in originals}}
        if capacity:
            result["capacities"] = summaries
            result["capacity_note"] = "成交额占比为事后压力诊断，未虚构成交量约束或市场冲击成交。"
        else:
            result.update(summaries[0])
            result["capital"] = capitals[0]
        (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        er.PARAMS.update(originals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="8-11")
    ap.add_argument("--stress", type=float, default=1.0, choices=[1.0, 2.0])
    ap.add_argument("--capacity", action="store_true")
    args = ap.parse_args()
    folds = pp.parse_folds(args.folds)
    run([f["id"] for f in folds], stress=args.stress, capacity=args.capacity)


if __name__ == "__main__":
    main()
