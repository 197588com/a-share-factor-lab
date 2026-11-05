# -*- coding: utf-8 -*-
"""wf_dragon.py — dragon 因子（LBCNT/LIMUC）全量 walk-forward 终审（P6 准入闸门）。

increment_test.py 只在 fold11 上验证了 160 维组合的增量；本脚本把同一口径推广到
全部 12 个 fold，逐 fold：
  1. 在各自训练窗 fit RobustZScoreNorm（防泄漏，逻辑同 increment_test.build_panel）
  2. ALSTMN(d_feat=160) 训练 20ep（断点续训，检查点 ckpt/wf/alstm_dragon/fold_N.pt，
     不触碰线上 alstm/158 维检查点）
  3. 测试段 RankIC/IC + 日度 IC 序列落盘 data/wf_ic/alstm_dragon_foldN.csv（DSR 输入）
最后汇总 data/live/wf_dragon_summary.json，与基线（alstm158 逐年 RankIC）对照。

用法：
  python wf_dragon.py build --folds 0-11     # 构建各 fold 的 dragon 面板缓存
  python wf_dragon.py train --folds 0-11     # 训练（断点续训）
  python wf_dragon.py eval  --folds 0-11     # 评估 + 汇总
  python wf_dragon.py run   --folds 0-11     # 一键
"""
import os, sys, json, time, argparse, pickle
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import train_models as tm
import wf_pipeline as wf
from increment_test import ALSTMN, GROUPS
GROUP = os.environ.get("WF_GROUP", "dragon")
COLS = GROUPS[GROUP][1]
D_FEAT = 158 + len(COLS)
MODEL = f"alstm_{GROUP}"
FEAT_PQ = os.path.join(WS, "data", "factor_decay", GROUPS[GROUP][0])
SUMMARY = os.path.join(WS, "data", "live", f"wf_{GROUP}_summary.json")
# 基线（alstm158 全量 walk-forward 已实测，wf_ic/alstm_fold{8..11} 同源）
BASELINE_RANKIC = {2023: 0.0111, 2024: 0.0315, 2025: 0.0013, 2026: -0.0010}


def cfg():
    return wf.load_config()


def ckdir():
    d = os.path.join(WS, "ckpt", "wf", MODEL)
    os.makedirs(d, exist_ok=True)
    return d


def panel_cache(fid):
    return os.path.join(WS, "data", "prep_wf", f"fold_{fid}_{GROUP}.pkl")


# ---------------- 面板构建 ----------------
def build(fold):
    c = panel_cache(fold["id"])
    if os.path.exists(c):
        print(f"fold{fold['id']} {GROUP} panel cached")
        return
    blob = wf.load_fold(fold["id"])
    panels, G, meta = blob["panels"], blob["G"], blob["meta"]
    dates, insts = meta["dates"], meta["insts"]
    nD, nI = len(dates), len(insts)

    m = pd.read_parquet(FEAT_PQ)[COLS]
    di = {d: i for i, d in enumerate(dates)}
    ii = {x: j for j, x in enumerate(insts)}
    M = np.zeros((nD, nI, len(COLS)), dtype=np.float32)
    dtv = m.index.get_level_values("datetime").date
    iv = m.index.get_level_values("instrument")
    sdt = pd.Series(dtv).map(di)
    siv = pd.Series(iv).map(ii)
    ok = sdt.notna().to_numpy() & siv.notna().to_numpy()
    M[sdt[ok].astype(int).to_numpy(), siv[ok].astype(int).to_numpy(), :] = \
        m.to_numpy(dtype=np.float32)[ok]
    print(f"fold{fold['id']} {GROUP} aligned: coverage "
          f"{np.isfinite(M).any(axis=2).mean():.1%}")

    # RobustZScoreNorm：本 fold 训练窗 per-instrument fit（与 qlib 处理器同口径）
    tr_end = pd.Timestamp(fold["train"][1]).date()
    tr = M[:sum(1 for d in dates if d <= tr_end)]
    if GROUPS[GROUP][2].get("nonzero_stats"):
        tr = np.where(tr == 0, np.nan, tr)   # 稀疏激活因子：零=无数据，不参与统计
    med = np.nanmedian(tr, axis=0)
    mad = np.nanmedian(np.abs(tr - med), axis=0)
    scale = 1.4826 * np.where(mad < 1e-12, np.nan, mad)
    Z = np.clip((M - med) / scale, -3, 3)
    Z = np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)

    panels2 = {s: (np.concatenate([F, Z], axis=2), L) for s, (F, L) in panels.items()}
    with open(c, "wb") as f:
        pickle.dump({"panels": panels2, "G": G, "meta": meta}, f)
    print(f"saved {c} (train fit <= {tr_end})")


# ---------------- 训练 ----------------
def train(fold, epochs=None):
    import torch
    from torch.utils.data import DataLoader
    cf = cfg()
    blob = pickle.load(open(panel_cache(fold["id"]), "rb"))
    panels, G, meta = blob["panels"], blob["G"], blob["meta"]
    mc = cf["models"]["alstm"]
    ep_t = epochs or mc["epochs"]
    tr = tm.WindowDS(*panels["train"], G, wf.seg_indices(blob, "train"), False)
    va = tm.DSWithDate(*panels["valid"], G, wf.seg_indices(blob, "valid"), False)
    print(f"fold{fold['id']} {MODEL}: train={len(tr)} valid={len(va)} target_epochs={ep_t}")

    model = ALSTMN(D_FEAT, hidden=mc.get("hidden", 64), num_layers=mc.get("num_layers", 2)).to(tm.DEVICE)
    ck = os.path.join(ckdir(), f"fold_{fold['id']}.pt")
    start_ep, best = 0, -1e9
    if os.path.exists(ck):
        st = torch.load(ck, map_location=tm.DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); start_ep = st["epoch"] + 1; best = st["best"]
        print(f"  resumed epoch {start_ep}, best valid IC {best:.4f}")
    opt = torch.optim.Adam(model.parameters(), lr=mc["lr"])
    lossf = torch.nn.MSELoss()
    for ep in range(start_ep, ep_t):
        model.train()
        tl = DataLoader(tr, batch_size=mc["batch_size"], shuffle=True, num_workers=0, drop_last=True)
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(tm.DEVICE), y.to(tm.DEVICE)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        vl = DataLoader(va, batch_size=1024, shuffle=False)
        ps, ys, ds = tm.predict(model, vl)
        m = tm.daily_ic(ps, ys, [meta["dates"][i] for i in ds])
        print(f"  ep{ep}: loss={tot/nb:.5f} valid IC={m['IC']:.4f} RankIC={m['RankIC']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep, "best": best,
                    "fold": fold["id"], "model_name": MODEL}, ck)
        if m["IC"] > best:
            best = m["IC"]
            torch.save({"model": model.state_dict(), "epoch": ep, "best": best,
                        "fold": fold["id"], "model_name": MODEL}, ck + ".best")
    print(f"fold{fold['id']} {MODEL} TRAIN DONE best_valid_IC={best:.4f}")


# ---------------- 评估 ----------------
def evaluate(fold):
    import torch
    from torch.utils.data import DataLoader
    blob = pickle.load(open(panel_cache(fold["id"]), "rb"))
    panels, G, meta = blob["panels"], blob["G"], blob["meta"]
    mc = cfg()["models"]["alstm"]
    model = ALSTMN(D_FEAT, hidden=mc.get("hidden", 64), num_layers=mc.get("num_layers", 2)).to(tm.DEVICE)
    ck = os.path.join(ckdir(), f"fold_{fold['id']}.pt")
    load_from = ck + ".best" if os.path.exists(ck + ".best") else ck
    st = torch.load(load_from, map_location=tm.DEVICE, weights_only=False)
    model.load_state_dict(st["model"])

    out = {}
    for seg in ("valid", "test"):
        ds_set = tm.DSWithDate(*panels[seg], G, wf.seg_indices(blob, seg), False)
        ps, ys, dd = tm.predict(model, DataLoader(ds_set, batch_size=2048, shuffle=False))
        dates = [meta["dates"][i] for i in dd]
        m = tm.daily_ic(ps, ys, dates)
        out[seg] = m
        if seg == "test":
            # 日度 IC 序列落盘（DSR 审计输入，与 wf_pipeline.eval_fold 同格式）
            lut = wf.regime_lookup(dates)
            df = pd.DataFrame({"d": dates, "p": ps, "y": ys})
            daily = [{"date": str(d), "ic": float(np.corrcoef(g["p"], g["y"])[0, 1]),
                      "regime": lut.get(d, "")}
                     for d, g in df.groupby("d") if len(g) >= 30]
            os.makedirs(os.path.join(WS, "data", "wf_ic"), exist_ok=True)
            pd.DataFrame(daily).to_csv(
                os.path.join(WS, "data", "wf_ic", f"{MODEL}_fold{fold['id']}.csv"), index=False)
        print(f"fold{fold['id']} {seg}: IC={m['IC']:.4f} ICIR={m['ICIR']:.3f} "
              f"RankIC={m['RankIC']:.4f} RankICIR={m['RankICIR']:.3f} ({m['days']}d)")
    return out


def summarize(folds):
    rows = []
    for fold in folds:
        rpath = os.path.join(WS, "data", "live", f"wf_{GROUP}_fold{fold['id']}.json")
        if not os.path.exists(rpath):
            continue
        r = json.load(open(rpath))
        year = int(fold["test"][0][:4])
        rows.append({"fold": fold["id"], "test_year": year,
                     "test_rankic": r["test"]["RankIC"], "test_ic": r["test"]["IC"],
                     "test_icir": r["test"]["ICIR"], "test_rankicir": r["test"]["RankICIR"],
                     "test_days": r["test"]["days"],
                     "valid_rankic": r["valid"]["RankIC"],
                     "baseline_rankic": BASELINE_RANKIC.get(year)})
    summ = {"model": f"{MODEL}{D_FEAT}", "group": GROUP, "n_folds": len(rows),
            "baseline": "alstm158", "rows": rows,
            "improved_years": [r["test_year"] for r in rows
                               if r["baseline_rankic"] is not None
                               and r["test_rankic"] > r["baseline_rankic"]],
            "all_test_mean_rankic": float(np.mean([r["test_rankic"] for r in rows]))}
    json.dump(summ, open(SUMMARY, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print(f"saved {SUMMARY}")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train", "eval", "run"])
    ap.add_argument("--folds", default="0-11")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    folds = wf.parse_folds(a.folds, wf.generate_folds(cfg()))
    for fold in folds:
        if a.cmd in ("build", "run"):
            build(fold)
        if a.cmd in ("train", "run"):
            train(fold, epochs=a.epochs)
        if a.cmd in ("eval", "run"):
            out = evaluate(fold)
            json.dump({s: {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                           for k, v in m.items()}
                       for s, m in out.items()},
                      open(os.path.join(WS, "data", "live", f"wf_{GROUP}_fold{fold['id']}.json"), "w"),
                      ensure_ascii=False, indent=2)
    if a.cmd in ("eval", "run"):
        summarize(folds)


if __name__ == "__main__":
    main()
