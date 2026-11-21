# -*- coding: utf-8 -*-
"""P30 open-to-open 与可执行净成本标签 shadow；独立权重、评分与缓存。"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent
ROOT = WS / "data" / "p30_execution_aware"
for sub in ("scores", "ckpt", "metrics"):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WS))
import train_models as tm
import wf_pipeline as wf
from holdout_guard import assert_window_allowed

CANDIDATES = ("p30_open_gross", "p30_open_net_masked")
SEEDS = {"p30_open_gross": 20261130, "p30_open_net_masked": 20261230}
FORMAL_END = pd.Timestamp("2026-06-30")
REFERENCE_ORDER_VALUE = 19000.0


def open_close_panel(meta):
    dates = pd.to_datetime(meta["dates"])
    insts = [x.upper() for x in meta["insts"]]
    d_idx = {d: i for i, d in enumerate(dates)}
    O = np.full((len(dates), len(insts)), np.nan, dtype=np.float64)
    C = np.full_like(O, np.nan)
    for j, instrument in enumerate(insts):
        p = WS / "data" / "csv" / f"{instrument.lower()}.csv"
        if not p.exists() or p.stat().st_size == 0:
            continue
        d = pd.read_csv(p, usecols=["date", "open", "close"])
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"].isin(d_idx) & d["date"].le(FORMAL_END)].drop_duplicates("date", keep="last")
        ii = d["date"].map(d_idx).to_numpy(dtype=int)
        O[ii, j] = d["open"].to_numpy(dtype=float)
        C[ii, j] = d["close"].to_numpy(dtype=float)
    return O, C


def make_labels(old_label, O, C, dates, insts, kind, allowed_end=FORMAL_END):
    """仅旧标签非空且 T+2 不越过本段终点与封存日的样本可成为新标签。"""
    dates = pd.to_datetime(dates)
    label_end = min(pd.Timestamp(allowed_end), FORMAL_END)
    nD, nI = old_label.shape
    assert O.shape == C.shape == old_label.shape
    gross = np.full_like(old_label, np.nan, dtype=np.float64)
    for t in range(nD - 2):
        if dates[t + 2] > label_end:
            continue
        entry, exit_ = O[t + 1], O[t + 2]
        valid = (np.isfinite(old_label[t]) & np.isfinite(entry) & np.isfinite(exit_)
                 & (entry > 0) & (exit_ > 0))
        gross[t, valid] = exit_[valid] / entry[valid] - 1
    if kind == "p30_open_gross":
        return gross.astype(np.float32)
    if kind != "p30_open_net_masked":
        raise ValueError(kind)
    out = np.full_like(gross, np.nan)
    commission = max(5.0, REFERENCE_ORDER_VALUE * 0.00025) / REFERENCE_ORDER_VALUE
    code = np.array([x[2:] for x in insts])
    transfer = np.where(np.array([x.startswith("SH") for x in insts]), 0.00001, 0.0)
    buy_cost = 0.001 + commission + transfer
    sell_cost = 0.001 + commission + 0.0005 + transfer
    for t in range(nD - 2):
        if dates[t + 2] > label_end:
            continue
        threshold = np.where(np.char.startswith(code, "68"), 0.198, 0.098)
        gem20 = np.char.startswith(code, "30") & (dates[t] >= pd.Timestamp("2020-08-24"))
        threshold = np.where(gem20, 0.198, threshold)
        close_t = C[t]
        gap = O[t + 1] / close_t - 1
        gap_block = gap > 0.015
        limit_threshold = threshold - 0.005 / np.clip(close_t, 1e-9, None)
        limit_block = gap >= limit_threshold
        valid = (np.isfinite(gross[t]) & np.isfinite(close_t) & (close_t > 0)
                 & np.isfinite(gap) & ~gap_block & ~limit_block)
        out[t, valid] = ((1 - sell_cost[valid]) / (1 + buy_cost[valid])) * (1 + gross[t, valid]) - 1
    return out.astype(np.float32)


def labels_for_fold(blob, kind, fold=None):
    O, C = open_close_panel(blob["meta"])
    labels = {}
    for seg in ("train", "valid", "test"):
        allowed_end = fold[seg][1] if fold is not None else FORMAL_END
        labels[seg] = make_labels(blob["panels"][seg][1], O, C,
                                  blob["meta"]["dates"], blob["meta"]["insts"], kind,
                                  allowed_end=allowed_end)
    return labels


def train_fold(fold, cfg, kind, blob, labels):
    import torch
    from torch.utils.data import DataLoader
    seed = SEEDS[kind] + int(fold["id"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    panels = {s: (blob["panels"][s][0], labels[s]) for s in ("train", "valid", "test")}
    G = np.zeros((panels["train"][0].shape[0], tm.D_GATE), dtype=np.float32)
    tr = tm.WindowDS(*panels["train"], G, wf.seg_indices({"panels": panels, "meta": blob["meta"]}, "train"), False)
    va = tm.DSWithDate(*panels["valid"], G, wf.seg_indices({"panels": panels, "meta": blob["meta"]}, "valid"), False)
    model = tm.ALSTM(hidden=64, num_layers=2).to(tm.DEVICE)
    ck = ROOT / "ckpt" / f"{kind}_fold{fold['id']}.pt"
    start_ep, best, hist = 0, -1e9, []
    if ck.exists():
        state = torch.load(ck, map_location=tm.DEVICE, weights_only=False)
        if (state.get("seed") != seed or state.get("candidate") != kind or
                state.get("fold") != fold["id"] or state.get("config_sha") != wf.config_sha(cfg) or
                state.get("label_sha") != label_sha(labels)):
            raise AssertionError("P30 检查点候选、fold、种子、配置或标签哈希不一致")
        model.load_state_dict(state["model"])
        start_ep, best, hist = state["epoch"] + 1, state["best"], state.get("history", [])
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    lossf = torch.nn.MSELoss()
    print(f"{kind} fold{fold['id']} train={len(tr)} valid={len(va)} seed={seed}", flush=True)
    for ep in range(start_ep, 20):
        model.train()
        loader = DataLoader(tr, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
        total, nb, t0 = 0.0, 0, time.time()
        for x, y in loader:
            x, y = x.to(tm.DEVICE), y.to(tm.DEVICE)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        ps, ys, ds = tm.predict(model, DataLoader(va, batch_size=1024, shuffle=False))
        metric = tm.daily_ic(ps, ys, [blob["meta"]["dates"][i] for i in ds])
        hist.append({"epoch": ep, "loss": total / max(nb, 1), **metric, "seconds": time.time() - t0})
        state = {"model": model.state_dict(), "epoch": ep, "best": best, "history": hist,
                 "seed": seed, "candidate": kind, "fold": fold["id"],
                 "config_sha": wf.config_sha(cfg), "label_sha": label_sha(labels)}
        torch.save(state, ck)
        if metric["IC"] > best:
            best = metric["IC"]
            state["best"] = best
            torch.save(state, Path(str(ck) + ".best"))
        print(f"{kind} fold{fold['id']} ep{ep} loss={hist[-1]['loss']:.6f} "
              f"validIC={metric['IC']:.5f} RankIC={metric['RankIC']:.5f} {hist[-1]['seconds']:.0f}s", flush=True)
        (ROOT / "metrics" / f"train_{kind}_fold{fold['id']}.json").write_text(
            json.dumps({"history": hist, "best": best}, indent=2), encoding="utf-8")
    del panels, tr, va, model
    gc.collect()


def label_sha(labels):
    h = hashlib.sha256()
    for seg in ("train", "valid", "test"):
        h.update(seg.encode())
        h.update(np.ascontiguousarray(labels[seg]).tobytes())
    return h.hexdigest()


def score_indices(blob, test_labels):
    """没有完整可评估开盘标签或越过封存日的日期不进入评分与交易信号。"""
    dates = pd.to_datetime(blob["meta"]["dates"])
    return np.array([i for i, d in enumerate(dates)
                     if d <= FORMAL_END and np.isfinite(test_labels[i]).any()], dtype=int)


def eval_fold(fold, kind, blob, labels):
    import torch
    from torch.utils.data import DataLoader
    # 测试集评分不依赖未来标签：用 0 占位让 WindowDS 只按特征完整性产生样本。
    # 真正标签单独保存，IC 计算自动 dropna；2026-06-30 后日期完全不进入评分集。
    test_label = np.zeros_like(labels["test"], dtype=np.float32)
    sample_idx = score_indices(blob, labels["test"])
    G = np.zeros((test_label.shape[0], tm.D_GATE), dtype=np.float32)
    te = tm.DSWithDate(blob["panels"]["test"][0], test_label, G, sample_idx, False)
    model = tm.ALSTM(hidden=64, num_layers=2).to(tm.DEVICE)
    ck = ROOT / "ckpt" / f"{kind}_fold{fold['id']}.pt"
    best_path = Path(str(ck) + ".best")
    state = torch.load(best_path if best_path.exists() else ck, map_location=tm.DEVICE, weights_only=False)
    if (state.get("candidate") != kind or state.get("fold") != fold["id"] or
            state.get("label_sha") != label_sha(labels)):
        raise AssertionError("P30 评分检查点与候选、fold 或标签哈希不一致")
    model.load_state_dict(state["model"])
    ps, _, ds = tm.predict(model, DataLoader(te, batch_size=2048, shuffle=False))
    sample_lookup = {(t, i): k for k, (t, i) in enumerate(te.samples)}
    # predict 与 te.samples 顺序一致；取真实标签，不用占位 0 计算指标。
    true_y = np.array([labels["test"][t, i] for t, i in te.samples], dtype=float)
    score_dates = [blob["meta"]["dates"][t] for t, _ in te.samples]
    instruments = [blob["meta"]["insts"][i].upper() for _, i in te.samples]
    score = pd.DataFrame({"date": pd.to_datetime(score_dates), "instrument": instruments,
                          "score": ps, "label": true_y, "fold": fold["id"]})
    out = ROOT / "scores" / f"{kind}_fold{fold['id']}.parquet"
    score.to_parquet(out, index=False)
    metric = tm.daily_ic(ps, true_y, score_dates)
    daily = score.dropna(subset=["label"]).groupby("date").apply(
        lambda g: pd.Series({"ic": g["score"].corr(g["label"]),
                             "rank_ic": g["score"].corr(g["label"], method="spearman"), "n": len(g)})
        if len(g) >= 30 else pd.Series({"ic": np.nan, "rank_ic": np.nan, "n": len(g)}),
        include_groups=False).reset_index()
    daily.to_csv(ROOT / "metrics" / f"ic_{kind}_fold{fold['id']}.csv", index=False)
    result = {"candidate": kind, "fold": fold["id"], "metrics": metric,
              "best_epoch": state["epoch"], "best_valid_ic": state["best"],
              "rows": len(score), "label_sha256": label_sha(labels),
              "config_sha": state.get("config_sha"),
              "score_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (ROOT / "metrics" / f"eval_{kind}_fold{fold['id']}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    del te, model
    gc.collect()


def health(blob, labels, fold=None):
    dates = pd.to_datetime(blob["meta"]["dates"])
    checks = {"no_sealed_label": True, "no_cross_segment_label": True,
              "label_shape": True, "finite_labels_safe": True}
    for seg, label in labels.items():
        checks["label_shape"] &= label.shape == blob["panels"][seg][1].shape
        checks["finite_labels_safe"] &= bool(np.isnan(label).all() or np.isfinite(label[~np.isnan(label)]).all())
        segment_end = min(pd.Timestamp(fold[seg][1]), FORMAL_END) if fold is not None else FORMAL_END
        for t in range(label.shape[0]):
            if not np.isfinite(label[t]).any():
                continue
            if t + 2 >= len(dates) or dates[t + 2] > FORMAL_END:
                checks["no_sealed_label"] = False
            elif dates[t + 2] > segment_end:
                checks["no_cross_segment_label"] = False
    checks["finite_labels_safe"] = bool(checks["finite_labels_safe"])
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "run"])
    ap.add_argument("--candidate", choices=CANDIDATES, required=True)
    ap.add_argument("--folds", default="8-11")
    args = ap.parse_args()
    if not (WS / "p30_execution_aware.md").exists():
        raise FileNotFoundError("P30 预注册缺失")
    cfg = wf.load_config()
    folds = {f["id"]: f for f in wf.generate_folds(cfg)}
    ids = []
    for part in args.folds.split(","):
        if "-" in part:
            a, b = map(int, part.split("-")); ids.extend(range(a, b + 1))
        else:
            ids.append(int(part))
    for fold_id in ids:
        fold = folds[fold_id]
        assert_window_allowed(fold["train"][0], min(fold["test"][1], str(FORMAL_END.date())), purpose="candidate")
        blob = wf.load_fold(fold_id)
        labels = labels_for_fold(blob, args.candidate, fold)
        checks = health(blob, labels, fold)
        label_health = {"candidate": args.candidate, "fold": fold_id, "passed": bool(all(checks.values())),
                        "checks": checks, "label_sha256": label_sha(labels), "config_sha": wf.config_sha(cfg),
                        "segment_ends": {seg: str(min(pd.Timestamp(fold[seg][1]), FORMAL_END).date())
                                         for seg in ("train", "valid", "test")},
                        "formal_label_end": str(FORMAL_END.date())}
        (ROOT / "metrics" / f"label_health_{args.candidate}_fold{fold_id}.json").write_text(
            json.dumps(label_health, ensure_ascii=False, indent=2), encoding="utf-8")
        print("label health", checks, flush=True)
        if args.cmd in ("train", "run"):
            train_fold(fold, cfg, args.candidate, blob, labels)
        if args.cmd in ("eval", "run"):
            assert_window_allowed(fold["test"][0], min(fold["test"][1], str(FORMAL_END.date())), purpose="candidate")
            eval_fold(fold, args.candidate, blob, labels)
        del blob, labels
        gc.collect()


if __name__ == "__main__":
    main()
