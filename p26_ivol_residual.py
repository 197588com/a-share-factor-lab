# -*- coding: utf-8 -*-
"""P26：IVOL 对完整 Alpha158 的同日截面秩残差化及一次性闸门。

只读既有研究面板；正式标签终点最晚 2026-06-30。若第一层不过，只落拒绝判决，
不训练、不修改生产。预注册见 p26_ivol_residual.md。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import candidate_ledger as ledger
from holdout_guard import assert_window_allowed

WS = Path(__file__).resolve().parent
FD = WS / "data" / "factor_decay"
OUT = WS / "data" / "backtest" / "p26_ivol_residual"
CANDIDATE = "p26_ivol_alpha158_residual"
RESOURCE = "wf_folds0-11"
FORMAL_END = pd.Timestamp("2026-06-30")
START = pd.Timestamp("2015-01-01")
YEARS = list(range(2015, 2027))
PREREG_THRESHOLD = 0.0125
MIN_CROSS_SECTION = 200
MIN_DAYS_PER_YEAR = 30
MIN_ABS_RESIDUAL_IC = 0.015
MIN_NEGATIVE_YEARS = 9


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def last_allowed_signal(labels):
    """T 日标签终点为 T+2；只用日历位置确定最后可用信号日。"""
    dates = pd.DatetimeIndex(labels.index.get_level_values("datetime").unique()).sort_values()
    eligible = dates[dates <= FORMAL_END]
    if len(eligible) < 3:
        raise AssertionError("正式日历不足三日")
    signal_end = eligible[-3]
    label_end = eligible[-1]
    assert label_end == FORMAL_END, (label_end, FORMAL_END)
    assert_window_allowed(signal_end, label_end, purpose="candidate")
    return signal_end, label_end


def cross_section_residual(group, feature_cols):
    """返回原始/残差 RankIC 与数值诊断；不在本函数内筛窗口或调参。"""
    g = group.replace([np.inf, -np.inf], np.nan).dropna(subset=list(feature_cols) + ["IVOL", "y"])
    if len(g) < MIN_CROSS_SECTION:
        return None
    xr = g[feature_cols].rank(pct=True, method="average").to_numpy(dtype=np.float64)
    iv = g["IVOL"].rank(pct=True, method="average").to_numpy(dtype=np.float64)
    yr = g["y"].rank(pct=True, method="average").to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones(len(g)), xr])
    coef, _, matrix_rank, _ = np.linalg.lstsq(design, iv, rcond=None)
    fitted = design @ coef
    residual = iv - fitted
    raw_rank = pd.Series(iv).rank(method="average").to_numpy()
    residual_rank = pd.Series(residual).rank(method="average").to_numpy()
    if np.std(raw_rank) == 0 or np.std(residual_rank) == 0 or np.std(yr) == 0:
        raise AssertionError("常数截面无法计算 RankIC")
    centered = xr - xr.mean(axis=0)
    orth = np.max(np.abs(centered.T @ residual / len(g))) if len(g) else np.nan
    ss = float(np.sum((iv - iv.mean()) ** 2))
    return {
        "n": int(len(g)), "matrix_rank": int(matrix_rank),
        "projection_r2": float(1 - np.sum(residual ** 2) / ss) if ss > 0 else 0.0,
        "raw_ic": float(np.corrcoef(raw_rank, yr)[0, 1]),
        "residual_ic": float(np.corrcoef(residual_rank, yr)[0, 1]),
        "orthogonality_mean_abs": float(abs(residual.mean())),
        "orthogonality_cov_max_abs": float(orth),
    }


def regime_map():
    """与 wf_pipeline.regime_lookup 同优先级，但额外保留来源和缺失计数。"""
    import duckdb
    con = duckdb.connect(WS / "data" / "factor_lab.duckdb", read_only=True)
    frames = []
    for source in ("hmm_v2", "hmm_roll"):
        rows = con.execute("SELECT trade_date, regime FROM regime_labels WHERE source=?", [source]).fetchdf()
        rows["trade_date"] = pd.to_datetime(rows["trade_date"])
        rows = rows.set_index("trade_date")["regime"].rename(source)
        frames.append(rows)
    con.close()
    r = pd.concat(frames, axis=1)
    out = r["hmm_roll"].where(r["hmm_roll"].notna(), r["hmm_v2"])
    source = pd.Series(np.where(r["hmm_roll"].notna(), "hmm_roll",
                                np.where(r["hmm_v2"].notna(), "hmm_v2", "missing")), index=r.index)
    return out, source


def group_stats(daily, key):
    rows = []
    for value, g in daily.groupby(key, dropna=False):
        rows.append({
            key: value, "days": int(len(g)), "raw_ic": float(g["raw_ic"].mean()),
            "residual_ic": float(g["residual_ic"].mean()),
            "residual_minus_raw": float((g["residual_ic"] - g["raw_ic"]).mean()),
            "negative_residual_day_ratio": float((g["residual_ic"] < 0).mean()),
            "mean_n": float(g["n"].mean()), "mean_projection_r2": float(g["projection_r2"].mean()),
        })
    return pd.DataFrame(rows)


def annual_cluster_negative(yearly):
    values = yearly.set_index("year")["residual_ic"].reindex(YEARS)
    if values.isna().any():
        raise AssertionError("年度序列不完整，不能做预注册年聚类检验")
    x = values.to_numpy(dtype=float)
    stat = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
    return {"years": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "t": stat, "df": len(x) - 1, "p_one_sided_negative": float(student_t.cdf(stat, len(x) - 1)),
            "negative_years": int((x < 0).sum())}


def annual_cluster_if_coverage_valid(yearly):
    """覆盖不足是数据健康拒绝，不得伪造均值、t 或 p 值。"""
    rows = yearly.set_index("year").reindex(YEARS)
    missing = [int(y) for y in YEARS if pd.isna(rows.loc[y, "residual_ic"])]
    enough_days = bool((rows["days"].fillna(0) >= MIN_DAYS_PER_YEAR).all())
    if missing or not enough_days:
        return {"evaluable": False, "reason": "insufficient_complete_cross_section_coverage",
                "years": int(rows["residual_ic"].notna().sum()), "missing_years": missing,
                "each_year_at_least_30_days": enough_days, "mean": None, "std": None,
                "t": None, "df": None, "p_one_sided_negative": None, "negative_years": None}
    out = annual_cluster_negative(yearly)
    out["evaluable"] = True
    out["missing_years"] = []
    out["each_year_at_least_30_days"] = True
    return out


def legacy_model_increment(signal_end):
    """旧 P8 模型的描述性复核，不作为新残差候选的替代证据。"""
    pairs = []
    for fold in range(12):
        base = pd.read_csv(WS / "data" / "wf_ic" / f"alstm_fold{fold}.csv", parse_dates=["date"])
        cand = pd.read_csv(WS / "data" / "wf_ic" / f"alstm_min_fold{fold}.csv", parse_dates=["date"])
        j = base.merge(cand, on="date", how="inner", suffixes=("_base", "_min"), validate="one_to_one")
        j = j[(j["date"] >= START) & (j["date"] <= signal_end)].copy()
        j["fold"] = fold
        pairs.append(j)
    p = pd.concat(pairs, ignore_index=True)
    p["delta_ic"] = p["ic_min"] - p["ic_base"]
    p["year"] = p["date"].dt.year
    p["regime"] = p["regime_min"].fillna(p["regime_base"]).fillna("missing")
    yearly = p.groupby("year").agg(days=("delta_ic", "size"), baseline_ic=("ic_base", "mean"),
                                   min_ic=("ic_min", "mean"), delta_ic=("delta_ic", "mean")).reset_index()
    x = yearly.set_index("year")["delta_ic"].reindex(YEARS).to_numpy(dtype=float)
    stat = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
    cluster = {"mean_delta_ic": float(x.mean()), "t": stat, "df": len(x) - 1,
               "p_one_sided_positive": float(student_t.sf(stat, len(x) - 1)),
               "nonnegative_years": int((x >= 0).sum())}
    regime = p.groupby("regime").agg(days=("delta_ic", "size"), baseline_ic=("ic_base", "mean"),
                                     min_ic=("ic_min", "mean"), delta_ic=("delta_ic", "mean")).reset_index()
    return p, yearly, regime, cluster


def main():
    if not (WS / "p26_ivol_residual.md").exists():
        raise FileNotFoundError("P26 预注册缺失")
    OUT.mkdir(parents=True, exist_ok=True)
    existing = [e for e in ledger._entries() if e["candidate_id"] == CANDIDATE]
    if existing:
        threshold = float(existing[-1]["prespecified"]["bonferroni_threshold"])
    else:
        threshold, n_before = ledger.threshold(RESOURCE)
        if n_before != 4 or abs(threshold - PREREG_THRESHOLD) > 1e-12:
            raise AssertionError(f"账本已偏离预注册起点：n={n_before}, threshold={threshold}")

    labels_all = pd.read_parquet(FD / "labels.parquet")
    signal_end, label_end = last_allowed_signal(labels_all)
    labels = labels_all[(labels_all.index.get_level_values("datetime") >= START)
                        & (labels_all.index.get_level_values("datetime") <= signal_end)]
    ivol_all = pd.read_parquet(FD / "min_feat.parquet", columns=["IVOL"])
    ivol = ivol_all[(ivol_all.index.get_level_values("datetime") >= START)
                    & (ivol_all.index.get_level_values("datetime") <= signal_end)]
    del labels_all, ivol_all
    regimes, regime_sources = regime_map()
    daily_rows, skipped, input_files = [], [], []
    for year in YEARS:
        path = FD / f"feat_{year}.parquet"
        input_files.append({"path": str(path.relative_to(WS)), "sha256": sha256(path)})
        feat = pd.read_parquet(path)
        dates = feat.index.get_level_values("datetime")
        feat = feat[(dates >= START) & (dates <= signal_end)]
        feature_cols = list(feat.columns)
        if len(feature_cols) != 158:
            raise AssertionError(f"{year} Alpha158 列数={len(feature_cols)}")
        # 数据契约修复（首次运行在任何 IC 计算前发现）：Qlib 的 VWAP0 因无 VWAP 源
        # 在全部年份整列为空；生产 Alpha158 的 Fillna 本就把它置 0。只对该列复刻
        # 生产语义，其他 157 列仍要求逐股逐日完整，不放宽样本门槛或改动检验。
        if feat["VWAP0"].notna().any():
            raise AssertionError(f"{year} VWAP0 不再是结构全空列，需重新审计数据契约")
        feat["VWAP0"] = 0.0
        df = feat.join(ivol, how="inner").join(labels, how="inner")
        for date, group in df.groupby(level="datetime", sort=True):
            result = cross_section_residual(group, feature_cols)
            if result is None:
                skipped.append({"date": str(date.date()), "reason": "complete_cross_section_lt_200"})
                continue
            result["date"] = date
            daily_rows.append(result)
        print(f"{year}: rows={len(df)} valid_days={sum(pd.Timestamp(x['date']).year == year for x in daily_rows)}", flush=True)
    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    daily["delta_ic"] = daily["residual_ic"] - daily["raw_ic"]
    daily["year"] = daily["date"].dt.year
    daily["regime"] = daily["date"].map(regimes).fillna("missing")
    daily["regime_source"] = daily["date"].map(regime_sources).fillna("missing")
    observed_yearly = group_stats(daily, "year")
    yearly = observed_yearly.set_index("year").reindex(YEARS).reset_index().rename(columns={"index": "year"})
    yearly["days"] = yearly["days"].fillna(0).astype(int)
    yearly["formal_coverage_valid"] = yearly["residual_ic"].notna() & (yearly["days"] >= MIN_DAYS_PER_YEAR)
    regime = group_stats(daily, "regime")
    primary = annual_cluster_if_coverage_valid(yearly)
    health = {
        "signal_end": str(signal_end.date()), "label_end": str(label_end.date()),
        "no_sealed_label_outcome": bool(label_end <= FORMAL_END),
        "all_12_years_present": bool(not primary["missing_years"]),
        "each_year_at_least_30_days": bool(primary["each_year_at_least_30_days"]),
        "finite_daily_metrics": bool(np.isfinite(daily[["raw_ic", "residual_ic", "projection_r2"]].to_numpy()).all()),
        "residual_mean_orthogonal": bool((daily["orthogonality_mean_abs"] < 1e-10).all()),
        "residual_feature_cov_orthogonal": bool((daily["orthogonality_cov_max_abs"] < 1e-8).all()),
        "no_future_regime": bool(daily["date"].max() <= signal_end),
        "regime_missing_days": int(daily["regime"].eq("missing").sum()),
        "regime_fallback_hmm_v2_days": int(daily["regime_source"].eq("hmm_v2").sum()),
        "complete_cross_section_skipped_days": int(len(skipped)),
        "valid_years": int(primary["years"]), "missing_years": primary["missing_years"],
    }
    health["passed"] = bool(all(v for k, v in health.items() if isinstance(v, bool)))
    gates = {
        "mean_residual_ic_le_minus_0_015": bool(primary["evaluable"] and primary["mean"] <= -MIN_ABS_RESIDUAL_IC),
        "at_least_9_of_12_negative_years": bool(primary["evaluable"] and primary["negative_years"] >= MIN_NEGATIVE_YEARS),
        "bonferroni_year_cluster_p_lt_0_0125": bool(primary["evaluable"] and primary["p_one_sided_negative"] < threshold),
        "health_passed": bool(health["passed"]),
    }
    gate1_passed = bool(all(gates.values()))
    daily.to_parquet(OUT / "daily_ic.parquet", index=False)
    yearly.to_csv(OUT / "yearly_ic.csv", index=False)
    regime.to_csv(OUT / "regime_ic.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "skipped_days.csv", index=False)

    legacy_daily, legacy_yearly, legacy_regime, legacy_cluster = legacy_model_increment(signal_end)
    legacy_daily.to_parquet(OUT / "legacy_model_daily_pairs.parquet", index=False)
    legacy_yearly.to_csv(OUT / "legacy_model_yearly.csv", index=False)
    legacy_regime.to_csv(OUT / "legacy_model_regime.csv", index=False)

    prespecified = {
        "preregistration": "p26_ivol_residual.md", "bonferroni_threshold": threshold,
        "n_before": 4, "direction": "negative residual RankIC", "min_mean_abs_ic": MIN_ABS_RESIDUAL_IC,
        "min_negative_years": MIN_NEGATIVE_YEARS, "cross_section_min": MIN_CROSS_SECTION,
        "alpha158_rank_ols": True, "signal_end": str(signal_end.date()), "label_end": str(label_end.date()),
    }
    evidence = {
        "candidate": CANDIDATE, "resource": RESOURCE, "formal_window": [str(START.date()), str(label_end.date())],
        "primary": primary, "gates": gates, "gate1_passed": gate1_passed, "health": health,
        "threshold": threshold, "ledger_n_before": 4, "prespecified": prespecified,
        "partial_period_outputs_are_descriptive_only": True,
        "legacy_model_increment_descriptive": legacy_cluster,
        "inputs": input_files + [{"path": "data/factor_decay/min_feat.parquet", "sha256": sha256(FD / "min_feat.parquet")},
                                 {"path": "data/factor_decay/labels.parquet", "sha256": sha256(FD / "labels.parquet")}],
        "code_sha256": sha256(Path(__file__)), "preregistration_sha256": sha256(WS / "p26_ivol_residual.md"),
    }
    if not gate1_passed:
        if primary["evaluable"]:
            decision_detail = (f"年聚类 t={primary['t']:.6f}，均值={primary['mean']:.8f}，"
                               f"负年={primary['negative_years']}/12")
            decision_reason = "statistical_gate_failed"
        else:
            decision_detail = (f"数据健康闸失败：有效年份 {primary['years']}/12，缺失 {primary['missing_years']}；"
                               "完整截面不足，未执行年聚类检验，p=null；局部年份只作描述，不构成残差 alpha 证据")
            decision_reason = "data_health_failed"
        entry = ledger.record(CANDIDATE, "factor", RESOURCE, "rejected_gate1",
                              p_value=primary["p_one_sided_negative"], prespecified=prespecified,
                              notes=f"P26 一次定生死：{decision_detail}；第一层闸门 {gates}；不训练、不改生产。")
        evidence["decision"] = "rejected_gate1"
        evidence["decision_reason"] = decision_reason
        evidence["next_step"] = "close_ivol_route_no_retrain"
        evidence["ledger_entry"] = entry or existing[-1]
    else:
        evidence["decision"] = "gate1_passed_model_test_required"
        evidence["next_step"] = "train_single_preregistered_alstm159_then_model_and_executable_gates"
    evidence["ledger_chain_ok_after"] = ledger.verify_chain()
    (OUT / "p26_ivol_residual_results.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: evidence[k] for k in ["formal_window", "primary", "gates", "decision", "next_step",
                                                   "legacy_model_increment_descriptive", "ledger_chain_ok_after"]},
                     ensure_ascii=False, indent=2), flush=True)
    print("\n逐年 IC：\n" + yearly.round(6).to_string(index=False), flush=True)
    print("\n分 regime IC：\n" + regime.round(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
