# -*- coding: utf-8 -*-
"""P27 宇宙扩展：独立 Qlib provider + PIT 标签 + 固定 ALSTM walk-forward。

用法：python p27_pipeline.py dump|prep --fold 8|run --folds 8-11|audit-data
"""
import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent
ROOT = WS / "data" / "p27_universe_expansion"
PROVIDER = ROOT / "provider"
PREP = ROOT / "prep"
SCORES = ROOT / "scores"
CKPT = ROOT / "ckpt"
METRICS = ROOT / "metrics"
META = ROOT / "meta"
for p in (PREP, SCORES, CKPT, METRICS, META):
    p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WS))
import train_models as tm
import wf_pipeline as wf
from candidate_ledger import record
from holdout_guard import assert_window_allowed

UNIVERSE = "csi500"
SEED0 = 20261118


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump_provider():
    sys.path.insert(0, str(WS / "qlib-repo" / "scripts"))
    from dump_bin import DumpDataAll
    DumpDataAll(
        data_path=str(ROOT / "csv"), qlib_dir=str(PROVIDER), freq="day",
        max_workers=4, date_field_name="date", symbol_field_name="instrument",
        include_fields="open,high,low,close,volume,amount,factor",
    ).dump()
    (PROVIDER / "instruments").mkdir(parents=True, exist_ok=True)
    ranges = pd.read_csv(META / "csi500_fetch_ranges.txt", sep="\t", header=None,
                         names=["instrument", "start", "end"])
    ranges.to_csv(PROVIDER / "instruments" / "csi500.txt", sep="\t", header=False, index=False)
    manifest = {"provider": str(PROVIDER), "instruments": len(ranges),
                "calendar_days": len(pd.read_csv(PROVIDER / "calendars" / "day.txt", header=None))}
    (META / "provider_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def membership():
    m = pd.read_parquet(META / "csi500_membership_daily.parquet")
    m["date"] = pd.to_datetime(m["date"])
    m["instrument"] = m["instrument"].str.upper()
    return set(zip(m["date"], m["instrument"]))


def mask_incomplete_formal_labels(frame):
    """封存日前最后两个信号日没有完整 T+1/T+2 标签，特征保留、标签置空。"""
    out = frame.copy()
    dates = sorted(set(out.index.get_level_values("datetime")))
    label_cols = [c for c in out.columns if c[0] == "label"]
    for d in dates[-2:]:
        out.loc[out.index.get_level_values("datetime") == d, label_cols] = np.nan
    return out


def fetch_frames(fold, cfg):
    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.data.handler import Alpha158
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN)
    h = Alpha158(
        instruments=UNIVERSE, start_time=fold["train"][0], end_time=fold["test"][1],
        fit_start_time=fold["train"][0], fit_end_time=fold["train"][1],
        infer_processors=cfg["data"]["processors"]["infer_processors"],
        learn_processors=cfg["data"]["processors"]["learn_processors"],
    )
    pit = membership()
    raw, filtered = {}, {}
    for seg in ("train", "valid", "test"):
        s, e = fold[seg]
        if seg == "test":
            e = min(e, "2026-06-30")
        assert_window_allowed(s, e, purpose="candidate")
        t0 = time.time()
        frame = h.fetch(selector=slice(s, e), level="datetime", col_set=["feature", "label"])
        frame.index = frame.index.set_levels(pd.to_datetime(frame.index.levels[0]), level=0)
        frame = frame.sort_index()
        raw[seg] = frame
        idx = frame.index
        keep = np.fromiter(((d, i.upper()) in pit for d, i in idx), bool, len(idx))
        filtered[seg] = frame.loc[keep].copy()
        if seg == "test":
            filtered[seg] = mask_incomplete_formal_labels(filtered[seg])
        expected = sum(d >= pd.Timestamp(s) and d <= pd.Timestamp(e) for d, _ in pit)
        coverage = len(filtered[seg]) / max(expected, 1)
        print(f"fold{fold['id']} {seg}: raw={frame.shape} PIT={filtered[seg].shape} "
              f"coverage={coverage:.3f} elapsed={time.time()-t0:.0f}s", flush=True)
    return raw, filtered


def build_segment_panels(raw, filtered):
    """F 保留历史特征，L 只在当日真实成分与正式分段上赋值。

    相比原管线三个全日历大面板，这里每段只保存自身及前 7 个交易日的回看，
    内存约降为三分之一；WindowDS 的样本生成逻辑不变。
    """
    feat_cols = [c for c in raw["train"].columns if c[0] == "feature"]
    if len(feat_cols) != tm.D_FEAT:
        raise AssertionError(f"特征数 {len(feat_cols)} != {tm.D_FEAT}")
    insts = sorted(set().union(*[set(f.index.get_level_values("instrument").str.upper()) for f in raw.values()]))
    i_idx = {c: j for j, c in enumerate(insts)}
    panels, sample_indices, metas = {}, {}, {}
    previous = None
    for seg in ("train", "valid", "test"):
        current = raw[seg]
        if previous is not None:
            prev_dates = sorted(set(previous.index.get_level_values("datetime")))[-7:]
            history = previous.loc[previous.index.get_level_values("datetime").isin(prev_dates)]
            combined = pd.concat([history, current])
            actual_dates = sorted(set(current.index.get_level_values("datetime")))
        else:
            combined = current
            actual_dates = sorted(set(current.index.get_level_values("datetime")))
        dates = sorted(set(combined.index.get_level_values("datetime")))
        d_idx = {d: i for i, d in enumerate(dates)}
        F = np.full((len(dates), len(insts), tm.D_FEAT), np.nan, dtype=np.float32)
        L = np.full((len(dates), len(insts)), np.nan, dtype=np.float32)
        ci = combined.index
        di = ci.get_level_values("datetime").map(d_idx).to_numpy()
        ii = ci.get_level_values("instrument").str.upper().map(i_idx).to_numpy()
        F[di, ii, :] = combined[feat_cols].to_numpy(dtype=np.float32)
        selected = filtered[seg]
        li = selected.index
        ldi = li.get_level_values("datetime").map(d_idx).to_numpy()
        lii = li.get_level_values("instrument").str.upper().map(i_idx).to_numpy()
        L[ldi, lii] = selected["label"].iloc[:, 0].to_numpy(dtype=np.float32)
        panels[seg] = (F, L)
        sample_indices[seg] = np.array([d_idx[d] for d in actual_dates], dtype=int)
        metas[seg] = {"dates": [d.date() for d in dates]}
        previous = current
    return panels, sample_indices, {"insts": insts, "segments": metas}


def valid_sample_counts(F, L, seg_dates_idx):
    """与 WindowDS 完全一致：158 维特征连续 8 日完整且当日标签有效。"""
    finite_features = np.isfinite(F).all(axis=2)
    finite_labels = np.isfinite(L)
    counts = []
    for t in seg_dates_idx:
        if t < tm.LOOKBACK - 1:
            counts.append(0)
            continue
        window = finite_features[t - tm.LOOKBACK + 1:t + 1].all(axis=0)
        counts.append(int(np.count_nonzero(window & finite_labels[t])))
    return counts


def coverage_report(fold, filtered, panels, sample_indices, meta):
    m = pd.read_parquet(META / "csi500_membership_daily.parquet")
    m["date"] = pd.to_datetime(m["date"])
    out = {}
    for seg, frame in filtered.items():
        s, e = fold[seg]
        if seg == "test":
            e = min(e, "2026-06-30")
        expected = m[(m["date"] >= s) & (m["date"] <= e)].groupby("date").size()
        counts = valid_sample_counts(*panels[seg], sample_indices[seg])
        dates = [meta["segments"][seg]["dates"][i] for i in sample_indices[seg]]
        got = pd.Series(counts, index=pd.to_datetime(dates), dtype=float)
        ratio = got.reindex(expected.index, fill_value=0) / expected
        out[seg] = {"median": float(ratio.median()), "p05": float(ratio.quantile(.05)),
                    "min": float(ratio.min()), "days": int(len(ratio)),
                    "rows": int(sum(counts)), "raw_label_rows": int(len(frame)),
                    "expected_rows": int(expected.sum()),
                    "definition": "complete_158_features_8day_window_and_finite_label"}
    out["passed"] = bool(all(v["median"] >= 0.70 for k, v in out.items() if k != "passed"))
    return out


def prep_fold(fold, cfg):
    cache = PREP / f"csi500_fold{fold['id']}.pkl"
    report_path = METRICS / f"coverage_fold{fold['id']}.json"
    if cache.exists() and report_path.exists():
        print(f"cached {cache}", flush=True)
        return cache
    raw, filtered = fetch_frames(fold, cfg)
    panels, sample_indices, meta = build_segment_panels(raw, filtered)
    coverage = coverage_report(fold, filtered, panels, sample_indices, meta)
    report_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    if not coverage["passed"]:
        raise RuntimeError(f"fold{fold['id']} 数据覆盖闸失败: {coverage}")
    blob = {"panels": panels, "sample_indices": sample_indices, "meta": meta,
            "universe": UNIVERSE, "fold": fold, "config_sha": wf.config_sha(cfg),
            "membership_sha": file_sha(META / "csi500_membership_daily.parquet")}
    tmp = cache.with_suffix(".tmp.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache)
    print(f"saved {cache} size={cache.stat().st_size/1e9:.2f}GB sha={file_sha(cache)}", flush=True)
    del raw, filtered, panels, blob
    gc.collect()
    return cache


def load_blob(fold_id):
    with open(PREP / f"csi500_fold{fold_id}.pkl", "rb") as f:
        return pickle.load(f)


def train_fold(fold, cfg):
    import torch
    from torch.utils.data import DataLoader
    seed = SEED0 + int(fold["id"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    blob = load_blob(fold["id"])
    panels, idx, meta = blob["panels"], blob["sample_indices"], blob["meta"]
    G = np.zeros((panels["train"][0].shape[0], tm.D_GATE), dtype=np.float32)
    tr = tm.WindowDS(*panels["train"], G, idx["train"], False)
    Gv = np.zeros((panels["valid"][0].shape[0], tm.D_GATE), dtype=np.float32)
    va = tm.DSWithDate(*panels["valid"], Gv, idx["valid"], False)
    model = tm.ALSTM(hidden=64, num_layers=2).to(tm.DEVICE)
    ck = CKPT / f"csi500_alstm_fold{fold['id']}.pt"
    start_ep, best, hist = 0, -1e9, []
    if ck.exists():
        state = torch.load(ck, map_location=tm.DEVICE, weights_only=False)
        if state.get("seed") != seed or state.get("config_sha") != wf.config_sha(cfg):
            raise AssertionError("检查点种子或配置不一致，禁止混用")
        model.load_state_dict(state["model"])
        start_ep, best, hist = state["epoch"] + 1, state["best"], state.get("history", [])
        print(f"resume fold{fold['id']} epoch={start_ep}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    lossf = torch.nn.MSELoss()
    print(f"fold{fold['id']} train={len(tr)} valid={len(va)} seed={seed}", flush=True)
    for ep in range(start_ep, 20):
        model.train()
        loader = DataLoader(tr, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
        total, batches, t0 = 0.0, 0, time.time()
        for x, y in loader:
            x, y = x.to(tm.DEVICE), y.to(tm.DEVICE)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        ps, ys, ds = tm.predict(model, DataLoader(va, batch_size=1024, shuffle=False))
        dates = [meta["segments"]["valid"]["dates"][i] for i in ds]
        metric = tm.daily_ic(ps, ys, dates)
        row = {"epoch": ep, "loss": total / max(batches, 1), **metric,
               "seconds": time.time() - t0}
        hist.append(row)
        state = {"model": model.state_dict(), "epoch": ep, "best": best, "history": hist,
                 "seed": seed, "config_sha": wf.config_sha(cfg), "universe": UNIVERSE}
        torch.save(state, ck)
        if metric["IC"] > best:
            best = metric["IC"]
            state["best"] = best
            torch.save(state, Path(str(ck) + ".best"))
        print(f"fold{fold['id']} ep{ep} loss={row['loss']:.6f} validIC={metric['IC']:.5f} "
              f"RankIC={metric['RankIC']:.5f} {row['seconds']:.0f}s", flush=True)
        (METRICS / f"train_fold{fold['id']}.json").write_text(
            json.dumps({"seed": seed, "history": hist, "best_valid_ic": best}, indent=2), encoding="utf-8")
    del blob, panels, tr, va, model
    gc.collect()


def eval_fold(fold):
    import torch
    from torch.utils.data import DataLoader
    blob = load_blob(fold["id"])
    panels, idx, meta = blob["panels"], blob["sample_indices"], blob["meta"]
    G = np.zeros((panels["test"][0].shape[0], tm.D_GATE), dtype=np.float32)
    te = tm.DSWithDate(*panels["test"], G, idx["test"], False)
    model = tm.ALSTM(hidden=64, num_layers=2).to(tm.DEVICE)
    ck = CKPT / f"csi500_alstm_fold{fold['id']}.pt"
    best_path = Path(str(ck) + ".best")
    state = torch.load(best_path if best_path.exists() else ck, map_location=tm.DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    ps, ys, ds = tm.predict(model, DataLoader(te, batch_size=2048, shuffle=False))
    dates = [meta["segments"]["test"]["dates"][i] for i in ds]
    instruments = [meta["insts"][te.samples[k][1]] for k in range(len(te.samples))]
    score = pd.DataFrame({"date": pd.to_datetime(dates), "instrument": instruments,
                          "score": ps, "label": ys, "fold": fold["id"]})
    out = SCORES / f"csi500_alstm_fold{fold['id']}.parquet"
    score.to_parquet(out, index=False)
    metric = tm.daily_ic(ps, ys, dates)
    daily = score.groupby("date").apply(
        lambda g: pd.Series({"ic": g["score"].corr(g["label"]),
                             "rank_ic": g["score"].corr(g["label"], method="spearman"),
                             "n": len(g)}) if len(g) >= 30 else pd.Series({"ic": np.nan, "rank_ic": np.nan, "n": len(g)}),
        include_groups=False).reset_index()
    daily.to_csv(METRICS / f"ic_fold{fold['id']}.csv", index=False)
    result = {"fold": fold["id"], "metrics": metric, "best_epoch": state["epoch"],
              "best_valid_ic": state["best"], "scores": str(out), "rows": len(score),
              "score_sha256": file_sha(out)}
    (METRICS / f"eval_fold{fold['id']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    del blob, panels, te, model
    gc.collect()
    return result


def audit_data():
    # 中证1000：当前源只覆盖当期，禁止回填历史；记录为数据闸失败而非统计拒绝。
    evidence = {
        "candidate_id": "p27_csi1000_alstm158",
        "decision": "rejected_gate0_data_health",
        "reason": "缺少覆盖2016-2026正式训练／检验期的可验证历史point-in-time成分",
        "observed": {
            "baostock_query_zz1000_stocks": "not_available_in_0.9.4",
            "akshare_index_stock_cons_csindex": "current_snapshot_only",
            "tushare_index_member_all": "empty_response_on_2026-11-18",
            "comein_searchIndexConstituents": "OAuth_expired_on_2026-11-18",
        },
        "historical_backfill_forbidden": True, "trained": False, "returns_seen": False,
    }
    (META / "csi1000_data_gate.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    record("p27_csi1000_alstm158", "structure", "wf_folds8-11", "rejected_gate0_data_health",
           None, {"preregistration": "p27_universe_expansion.md", "gate": "PIT history required"},
           "无全历史PIT成分；未训练、未看收益，不允许当前名单倒填")
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)


def parse_folds(spec):
    folds = {f["id"]: f for f in wf.generate_folds(wf.load_config())}
    ids = []
    for part in spec.split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(part))
    return [folds[i] for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "prep", "train", "eval", "run", "audit-data"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--folds", default="8-11")
    args = ap.parse_args()
    if not (WS / "p27_universe_expansion.md").exists():
        raise FileNotFoundError("P27 预注册缺失")
    if args.cmd == "audit-data":
        audit_data()
        return
    if args.cmd == "dump":
        dump_provider()
        return
    folds = parse_folds(str(args.fold)) if args.fold is not None else parse_folds(args.folds)
    cfg = wf.load_config()
    for fold in folds:
        if args.cmd in ("prep", "run"):
            prep_fold(fold, cfg)
        if args.cmd in ("train", "run"):
            train_fold(fold, cfg)
        if args.cmd in ("eval", "run"):
            eval_fold(fold)


if __name__ == "__main__":
    main()
