# -*- coding: utf-8 -*-
"""P30 候选的 P25 同口径执行与收益序列落盘。"""
import argparse
import json
from pathlib import Path

import pandas as pd

import execution_replay as er
import shadow_execution as se
import wf_pipeline as wf

WS = Path(__file__).resolve().parent
ROOT = WS / "data" / "p30_execution_aware"
P25 = WS / "data" / "backtest" / "execution_replay"


def require_p25_data_gates():
    """P30 共享 P25 原始价与行动缓存，继承其健康结论但不绕过强制入口。"""
    health = json.loads((P25 / "health.json").read_text(encoding="utf-8"))
    manifest = json.loads((P25 / "raw_manifest.json").read_text(encoding="utf-8"))
    checks = {"p25_health": bool(health.get("passed")),
              "p25_raw_complete": bool(not manifest.get("failures") and manifest.get("actions_refreshed"))}
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def require_candidate_data_gates(candidate, folds):
    """训练完成度、标签／配置／评分哈希与封存边界全部通过后才允许执行。"""
    import hashlib
    base = require_p25_data_gates()
    fold_audits = []
    for fold in folds:
        paths = {
            "health": ROOT / "metrics" / f"label_health_{candidate}_fold{fold}.json",
            "train": ROOT / "metrics" / f"train_{candidate}_fold{fold}.json",
            "eval": ROOT / "metrics" / f"eval_{candidate}_fold{fold}.json",
            "score": ROOT / "scores" / f"{candidate}_fold{fold}.parquet",
        }
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{candidate} fold{fold} 终审输入缺失：{missing}")
        health = json.loads(paths["health"].read_text(encoding="utf-8"))
        train = json.loads(paths["train"].read_text(encoding="utf-8"))
        evaluation = json.loads(paths["eval"].read_text(encoding="utf-8"))
        score = pd.read_parquet(paths["score"])
        se.validate_score_frame(score)
        epochs = [row["epoch"] for row in train.get("history", [])]
        checks = {
            "candidate_and_fold": bool(health.get("candidate") == evaluation.get("candidate") == candidate
                                       and health.get("fold") == evaluation.get("fold") == fold),
            "label_health": bool(health.get("passed") and all(health.get("checks", {}).values())),
            "completed_20_epochs": epochs == list(range(20)),
            "label_hash_match": bool(health.get("label_sha256") == evaluation.get("label_sha256")),
            "config_hash_match": bool(health.get("config_sha") and
                                      health.get("config_sha") == evaluation.get("config_sha")),
            "score_hash_match": bool(hashlib.sha256(paths["score"].read_bytes()).hexdigest() ==
                                     evaluation.get("score_sha256")),
            "score_rows_match": bool(len(score) == evaluation.get("rows")),
            "score_before_holdout": bool(pd.to_datetime(score["date"]).max() <= pd.Timestamp("2026-06-30")),
        }
        if not all(checks.values()):
            raise AssertionError({"candidate": candidate, "fold": fold, "checks": checks})
        fold_audits.append({"fold": fold, "checks": checks,
                            "label_sha256": health["label_sha256"], "config_sha": health["config_sha"],
                            "score_sha256": evaluation["score_sha256"]})
    result = {"passed": True, "candidate": candidate, "p25": base, "folds": fold_audits}
    (ROOT / "metrics" / f"execution_data_gates_{candidate}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run(candidate, folds, stress=1.0):
    data_gates = require_candidate_data_gates(candidate, folds)
    out = ROOT / ("execution" if stress == 1 else f"execution_cost_x{stress:g}") / candidate
    out.mkdir(parents=True, exist_ok=True)
    scores_all = pd.concat([pd.read_parquet(ROOT / "scores" / f"{candidate}_fold{i}.parquet", columns=["instrument"])
                            for i in folds], ignore_index=True)
    store = er.RawStore(scores_all["instrument"].str.upper().unique())
    originals = {k: er.PARAMS[k] for k in ("slip", "comm", "stamp", "transfer")}
    for k in originals:
        er.PARAMS[k] = originals[k] * stress
    try:
        daily_all, fold_rows = [], []
        for fold in folds:
            baseline = pd.read_parquet(P25 / f"daily_fold{fold}.parquet")
            baseline["return"] = baseline["baseline_return"]
            baseline["cost"] = baseline["baseline_cost"]
            baseline["excess"] = baseline["baseline_excess"]
            raw_score = pd.read_parquet(ROOT / "scores" / f"{candidate}_fold{fold}.parquet")
            score, events = se.prepare_scores_qlib(raw_score, baseline, P25 / "qlib_snapshot")
            daily, orders, actions = se.run_account(score, baseline, store)
            h = se.health(daily, orders, len(baseline))
            if not h["passed"]:
                raise AssertionError(h)
            daily["fold"] = fold
            daily.to_parquet(out / f"daily_fold{fold}.parquet", index=False)
            orders.to_parquet(out / f"orders_fold{fold}.parquet", index=False)
            events.to_parquet(out / f"mask_events_fold{fold}.parquet", index=False)
            actions.to_parquet(out / f"actions_fold{fold}.parquet", index=False)
            m = se.metric(daily)
            fold_rows.append({"fold": fold, "year": int(daily["datetime"].dt.year.iloc[0]),
                              **m, **se.order_stats(orders)})
            daily_all.append(daily)
            print(f"{candidate} fold{fold} x{stress}: ann={m['annual_net_excess']*100:.2f}% "
                  f"IR={m['net_ir']:.3f} MDD={m['mdd']*100:.2f}%", flush=True)
        daily = pd.concat(daily_all, ignore_index=True).sort_values("datetime").reset_index(drop=True)
        daily.to_parquet(out / "daily_all.parquet", index=False)
        overall = se.metric(daily)
        orders = pd.concat([pd.read_parquet(out / f"orders_fold{i}.parquet") for i in folds], ignore_index=True)
        summary = {"candidate": candidate, "stress": stress, "overall": overall,
                   "folds": fold_rows, "orders": se.order_stats(orders),
                   "data_gates": data_gates,
                   "params": {k: er.PARAMS[k] for k in originals}}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(fold_rows).to_csv(out / "fold_summary.csv", index=False)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        er.PARAMS.update(originals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=["p30_open_gross", "p30_open_net_masked"])
    ap.add_argument("--folds", default="8-11")
    ap.add_argument("--stress", type=float, default=1.0, choices=[1.0, 2.0])
    args = ap.parse_args()
    if "-" in args.folds:
        a, b = map(int, args.folds.split("-")); folds = list(range(a, b + 1))
    else:
        folds = [int(x) for x in args.folds.split(",")]
    run(args.candidate, folds, args.stress)


if __name__ == "__main__":
    main()
