# -*- coding: utf-8 -*-
"""P27 推断涨跌停与 P25 Tushare 真实涨跌停的跨源核验；不替换数据源。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent
P27 = WS / "data" / "p27_universe_expansion"
P25 = WS / "data" / "backtest" / "execution_replay" / "raw"


def validate_limits(raw_dir=P27 / "raw_daily", reference_dir=P25):
    frames = []
    for path in sorted(Path(raw_dir).glob("*.parquet")):
        reference = Path(reference_dir) / path.name
        if not reference.exists():
            continue
        left = pd.read_parquet(path, columns=["date", "up_limit", "down_limit"])
        right = pd.read_parquet(reference, columns=["date", "up_limit", "down_limit"])
        left["date"] = pd.to_datetime(left["date"])
        right["date"] = pd.to_datetime(right["date"])
        merged = left.merge(right, on="date", how="inner", suffixes=("_p27", "_tushare"),
                            validate="one_to_one")
        if not merged.empty:
            merged["instrument"] = path.stem.upper()
            frames.append(merged)
    if not frames:
        raise ValueError("P27/P25 没有可交叉核验的重叠涨跌停样本")
    detail = pd.concat(frames, ignore_index=True)
    valid = detail[["up_limit_p27", "up_limit_tushare", "down_limit_p27", "down_limit_tushare"]].notna().all(axis=1)
    detail = detail.loc[valid].copy()
    detail["up_abs_diff"] = (detail["up_limit_p27"] - detail["up_limit_tushare"]).abs()
    detail["down_abs_diff"] = (detail["down_limit_p27"] - detail["down_limit_tushare"]).abs()
    # 最小报价单位为 0.01 元；超过半分才是实际不一致。
    mismatch = detail[(detail["up_abs_diff"] > .005) | (detail["down_abs_diff"] > .005)].copy()
    summary = {
        "instruments": int(detail["instrument"].nunique()),
        "rows": int(len(detail)),
        "mismatch_rows": int(len(mismatch)),
        "mismatch_rate": float(len(mismatch) / len(detail)),
        "up_exact_rate": float((detail["up_abs_diff"] <= .005).mean()),
        "down_exact_rate": float((detail["down_abs_diff"] <= .005).mean()),
        "up_abs_diff_max": float(detail["up_abs_diff"].max()),
        "down_abs_diff_max": float(detail["down_abs_diff"].max()),
        "p27_source": "BaoStock preclose + historical board/ST + decimal half-up",
        "reference_source": "P25 Tushare stk_limit raw archive",
        "replacement_forbidden": True,
    }
    return summary, detail, mismatch


def apply_official_evidence(candidate_dir=P27 / "actions", reference_dir=P25, evidence_path=None):
    """只按可核验官方实施公告合并候选缺失计划；参考数据或候选金额不符即抛错。"""
    from p27_data import atomic_to_parquet
    evidence_path = Path(evidence_path) if evidence_path else P27 / "meta" / "action_official_evidence.json"
    result = {"passed": True, "events": 0, "records": [], "evidence_path": str(evidence_path)}
    if not evidence_path.exists():
        result["status"] = "no_evidence_file"
        return result
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    if document.get("schema") != "p27_official_action_evidence_v1":
        raise ValueError("未知官方公司行动证据 schema")
    for evidence in document.get("events", []):
        required = {"instrument", "ex_date", "pay_date", "cash_div", "stk_div", "candidate_components",
                    "official_components", "url", "verbatim", "announcement"}
        if not required.issubset(evidence):
            raise ValueError(f"官方证据字段不完整：{required - set(evidence)}")
        instrument = evidence["instrument"].upper()
        ex = pd.Timestamp(evidence["ex_date"])
        cash = float(evidence["cash_div"])
        stock = float(evidence["stk_div"])
        if not np.isclose(sum(float(x) for x in evidence["official_components"]), cash, atol=1e-10):
            raise AssertionError(f"官方组成金额不等于总额：{instrument} {ex.date()}")
        path = Path(candidate_dir) / f"{instrument.lower()}.parquet"
        reference_path = Path(reference_dir) / f"{instrument.lower()}_actions.parquet"
        if not path.exists() or not reference_path.exists():
            raise FileNotFoundError(f"官方证据对应缓存缺失：{instrument}")
        left = pd.read_parquet(path)
        right = pd.read_parquet(reference_path)
        left["ex_date"] = pd.to_datetime(left["ex_date"], errors="coerce")
        right["ex_date"] = pd.to_datetime(right["ex_date"], errors="coerce")
        li = left.index[left["ex_date"].eq(ex)]
        ri = right.index[right["ex_date"].eq(ex)]
        if len(li) != 1 or len(ri) != 1:
            raise AssertionError(f"官方证据对应事件不是唯一：{instrument} {ex.date()}")
        idx = li[0]
        ref = right.loc[ri[0]]
        if not (np.isclose(float(ref["cash_div"]), cash, atol=1e-10) and
                np.isclose(float(ref["stk_div"]), stock, atol=1e-10) and
                pd.Timestamp(ref["pay_date"]) == pd.Timestamp(evidence["pay_date"])):
            raise AssertionError(f"官方证据与参考实施明细不一致：{instrument} {ex.date()}")
        current = float(left.at[idx, "cash_div"])
        if not np.isclose(current, cash, atol=1e-10):
            allowed = any(np.isclose(current, float(x), atol=1e-10) for x in evidence["candidate_components"])
            if not allowed:
                raise AssertionError(f"候选金额不属于官方实施计划子集：{instrument} {ex.date()} cash={current}")
        if not np.isclose(float(left.at[idx, "stk_div"]), stock, atol=1e-10):
            raise AssertionError(f"候选送转与官方实施公告不一致：{instrument} {ex.date()}")
        changed = not np.isclose(current, cash, atol=1e-10) or left.at[idx, "cash_source"] != "official_implementation_combined"
        left.loc[idx, "cash_div"] = cash
        left.loc[idx, "cash_div_tax"] = cash
        left.loc[idx, "pay_date"] = pd.Timestamp(evidence["pay_date"])
        left.loc[idx, "cash_source"] = "official_implementation_combined"
        implied = pd.to_numeric(pd.Series([left.at[idx, "reference_implied_cash"]]), errors="coerce").iloc[0]
        if pd.notna(implied):
            left.loc[idx, "cash_reconciliation_gap"] = cash - float(implied)
        for key in ("url", "verbatim", "announcement"):
            left.loc[idx, f"evidence_{key}"] = evidence[key]
        left.loc[idx, "evidence_official_components"] = json.dumps(evidence["official_components"])
        if changed:
            atomic_to_parquet(left, path)
        result["records"].append({"instrument": instrument, "ex_date": str(ex.date()), "cash_div": cash,
                                  "status": "applied" if changed else "already_verified", "url": evidence["url"]})
    result["events"] = len(result["records"])
    result["status"] = "verified"
    return result


def adjudicate_one_sided(event, raw):
    """单边事件只有被原始除权参考价独立证实才可解除；不静默忽略。"""
    import p27_actions as pa
    side = str(event["_merge"])
    if side not in ("left_only", "right_only"):
        return {"status": "not_required"}
    suffix = "p27" if side == "left_only" else "tushare"
    cash = pd.to_numeric(pd.Series([event.get(f"cash_div_{suffix}")]), errors="coerce").iloc[0]
    stock = pd.to_numeric(pd.Series([event.get(f"stk_div_{suffix}")]), errors="coerce").fillna(0).iloc[0]
    reference = pa.exchange_reference(raw, event["ex_date"], stock)
    if reference is None:
        return {"status": "unresolved_missing_exchange_reference", **(reference or {})}
    implied = float(reference["reference_implied_cash"])
    result = {**reference, "cash_reconciliation_gap": float(cash - implied) if pd.notna(cash) else np.nan}
    if stock > 1e-8:
        return {"status": "unresolved_one_sided_stock_plan", **result}
    if implied < 0 or pd.isna(cash):
        return {"status": "unresolved_negative_or_missing_cash", **result}
    if abs(float(cash) - implied) > 0.011:
        return {"status": "unresolved_one_sided_cash_conflict", **result}
    if side == "left_only":
        return {"status": "reference_missing_exchange_verified", **result}
    # 候选缺失时采用精确到分的交易所参考价复原值，而非直接照抄另一数据商。
    return {"status": "exchange_reference_imputed", "cash_div": implied, **result}


def validate_actions(candidate_dir=P27 / "actions", reference_dir=P25, raw_dir=None):
    """公司行动外连接与独立裁决；可复原缺项原子补记，未决差异 fail closed。"""
    import p27_actions as pa
    from p27_data import atomic_to_parquet
    columns = ["instrument", "ex_date", "pay_date", "cash_div", "stk_div"]
    metadata = ["cash_source", "reference_implied_cash", "cash_reconciliation_gap"]
    frames = []
    for path in sorted(Path(candidate_dir).glob("*.parquet")):
        reference = Path(reference_dir) / f"{path.stem}_actions.parquet"
        if not reference.exists():
            continue
        left = pd.read_parquet(path)
        right = pd.read_parquet(reference)
        for frame, wanted in ((left, columns + metadata), (right, columns)):
            for col in wanted:
                if col not in frame:
                    frame[col] = pd.Series(dtype=float if col in ("cash_div", "stk_div") else object)
        # 保留候选全部实施明细字段，补记后原子写回不得丢审计列。
        left = left.copy()
        right = right[columns].copy()
        for frame in (left, right):
            frame["ex_date"] = pd.to_datetime(frame["ex_date"], errors="coerce")
            frame["pay_date"] = pd.to_datetime(frame["pay_date"], errors="coerce")
        # 只核验正式执行窗；补写必须使用完整缓存，不能删除更早年份供审计的事件。
        left_full = left.copy()
        left = left[left["ex_date"].between("2023-01-01", "2026-06-30")].copy()
        right = right[right["ex_date"].between("2023-01-01", "2026-06-30")].copy()
        left_for_merge = left.rename(columns={c: f"{c}_p27" for c in metadata})
        merged = left_for_merge.merge(right, on=["instrument", "ex_date"], how="outer",
                                      suffixes=("_p27", "_tushare"), indicator=True, validate="one_to_one")
        merged["original_merge"] = merged["_merge"].astype(str)
        merged["adjudication"] = "not_required"
        additions = []
        one_sided = merged[merged["_merge"].ne("both")]
        raw = None
        if raw_dir is not None and not one_sided.empty:
            raw_path = Path(raw_dir) / path.name
            if raw_path.exists():
                raw = pd.read_parquet(raw_path, columns=["date", "close", "preclose"])
        for idx, event in one_sided.iterrows():
            verdict = adjudicate_one_sided(event, raw) if raw is not None else {
                "status": "unresolved_no_raw_crosscheck"}
            merged.loc[idx, "adjudication"] = verdict["status"]
            if verdict["status"] == "reference_missing_exchange_verified":
                continue
            if verdict["status"] != "exchange_reference_imputed":
                continue
            cash = float(verdict["cash_div"])
            pay_date = event["pay_date_tushare"]
            merged.loc[idx, "_merge"] = "both"
            merged.loc[idx, "cash_div_p27"] = cash
            merged.loc[idx, "stk_div_p27"] = float(event["stk_div_tushare"])
            merged.loc[idx, "pay_date_p27"] = pay_date
            merged.loc[idx, "cash_source_p27"] = "exchange_reference_imputed"
            merged.loc[idx, "reference_implied_cash_p27"] = verdict["reference_implied_cash"]
            merged.loc[idx, "cash_reconciliation_gap_p27"] = 0.0
            addition = {c: np.nan for c in left.columns}
            addition.update({
                "instrument": event["instrument"], "ex_date": event["ex_date"], "pay_date": pay_date,
                "cash_div": cash, "cash_div_tax": cash, "stk_div": float(event["stk_div_tushare"]),
                "cash_source": "exchange_reference_imputed", "raw_cash_sum": 0.0,
                "n_raw_rows": 0, "n_plans": 0, "n_pay_dates": int(pd.notna(pay_date)),
                "reference_implied_cash": verdict["reference_implied_cash"],
                "cash_reconciliation_gap": 0.0,
            })
            additions.append(addition)
        if additions:
            left_full = pd.concat([left_full, pd.DataFrame(additions, columns=left_full.columns)], ignore_index=True)
            left_full = left_full.sort_values("ex_date").drop_duplicates(["instrument", "ex_date"], keep="last")
            atomic_to_parquet(left_full.reset_index(drop=True), path)
        frames.append(merged)
    detail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if detail.empty:
        return {"passed": False, "reason": "no_overlapping_action_events"}, detail
    both = detail["_merge"].eq("both")
    cash_diff = (detail["cash_div_p27"] - detail["cash_div_tushare"]).abs()
    stock_diff = (detail["stk_div_p27"] - detail["stk_div_tushare"]).abs()
    left_pay = detail["pay_date_p27"]
    right_pay = detail["pay_date_tushare"]
    cash_involved = detail[["cash_div_p27", "cash_div_tushare"]].abs().max(axis=1).fillna(0).gt(0)
    # 双方均 NaT 相等；纯送转单方无支付日不制造现金差异，有现金时仍严格失败。
    pay_mismatch = both & ((left_pay.notna() & right_pay.notna() & left_pay.ne(right_pay)) |
                           ((left_pay.isna() ^ right_pay.isna()) & cash_involved))
    detail["cash_abs_diff"] = cash_diff
    detail["stock_abs_diff"] = stock_diff
    original = detail["original_merge"]
    verified = detail["adjudication"].eq("reference_missing_exchange_verified")
    imputed = detail["adjudication"].eq("exchange_reference_imputed")
    missing_candidate = int(original.eq("right_only").sum() - imputed.sum())
    missing_reference = int(original.eq("left_only").sum() - verified.sum())
    cash_bad = int((both & cash_diff.gt(0.011)).sum())
    stock_bad = int((both & stock_diff.gt(1e-8)).sum())
    pay_bad = int(pay_mismatch.sum())
    adjudications = detail.loc[detail["adjudication"].ne("not_required"), "adjudication"].value_counts()
    sources = detail.get("cash_source_p27", pd.Series(index=detail.index, dtype=object))
    imputed_sources = sources.eq("exchange_reference_imputed")
    official_sources = sources.eq("official_implementation_combined")
    summary = {
        "instruments": int(detail["instrument"].nunique()), "events": int(len(detail)),
        "matched_events": int(both.sum()), "direct_matched_events": int(original.eq("both").sum()),
        "missing_in_candidate": missing_candidate, "missing_in_reference": missing_reference,
        "raw_missing_in_candidate": int(original.eq("right_only").sum()),
        "raw_missing_in_reference": int(original.eq("left_only").sum()),
        "reference_missing_exchange_verified": int(verified.sum()),
        "exchange_reference_imputed": int(imputed_sources.sum()),
        "official_implementation_combined": int(official_sources.sum()),
        "cash_mismatch_over_0_011": cash_bad, "stock_mismatch": stock_bad, "pay_date_mismatch": pay_bad,
        "adjudications": {str(k): int(v) for k, v in adjudications.items()},
        "passed": bool(missing_candidate == 0 and missing_reference == 0 and
                       cash_bad == 0 and stock_bad == 0 and pay_bad == 0),
        "scope": "outer join plus raw exchange-reference adjudication on instruments shared with P25; not a substitute for full-universe source completeness",
    }
    return summary, detail


def main():
    summary, detail, mismatch = validate_limits()
    out = P27 / "meta"
    detail.to_parquet(out / "limit_cross_validation_detail.parquet", index=False)
    mismatch.head(5000).to_csv(out / "limit_cross_validation_mismatches.csv", index=False)
    path = out / "limit_cross_validation.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
