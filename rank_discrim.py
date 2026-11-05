# -*- coding: utf-8 -*-
"""rank_discrim.py — 模型分数的名次区分度验证（top1 参考口径的证据）。

问题：模型的日度分数在不同名次之间有没有单调区分度？rank1 是否真的优于 rank2-10？

方法（纯推理，不重训；与 wf_dragon.evaluate 同口径）：
  12 fold 的测试段逐日横截面：按分数排名分桶
  [1] [2-3] [4-5] [6-10] [11-20] [21-50] [51-150] [151-300]
  每桶每日超额 = 桶内 label 均值 - 当日全截面 label 均值；
  汇总：总均值/t 值 + 按测试年分层（重点 2023-2026 样本外）。
  附：rank1 日度超额分布（均值/标准差/正比率）与 rank1 vs top2-10 等权的逐日胜率。

用法：
  python3 rank_discrim.py infer --folds 0-3    # 推理并缓存（每 fold 一次）
  python3 rank_discrim.py infer --folds 4-7
  python3 rank_discrim.py infer --folds 8-11
  python3 rank_discrim.py report               # 汇总报告
"""
import os, sys, json, argparse, pickle
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import train_models as tm
import wf_pipeline as wf
from increment_test import ALSTMN

CACHE = os.path.join(WS, "data", "rank_discrim")
BUCKETS = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 150), (151, 300)]
OOS_YEARS = (2023, 2024, 2025, 2026)


def parse_folds(spec):
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))


def infer_one(fid):
    import torch
    from torch.utils.data import DataLoader
    outp = os.path.join(CACHE, f"fold_{fid}.pkl")
    if os.path.exists(outp):
        print(f"fold{fid} cached")
        return
    blob = pickle.load(open(os.path.join(WS, "data", "prep_wf", f"fold_{fid}.pkl"), "rb"))
    panels, G, meta = blob["panels"], blob["G"], blob["meta"]
    mc = wf.load_config()["models"]["alstm"]
    model = ALSTMN(158, hidden=mc.get("hidden", 64), num_layers=mc.get("num_layers", 2)).to(tm.DEVICE)
    ck = os.path.join(WS, "ckpt", "wf", "alstm", f"fold_{fid}.pt")
    load_from = ck + ".best" if os.path.exists(ck + ".best") else ck
    st = torch.load(load_from, map_location=tm.DEVICE, weights_only=False)
    model.load_state_dict(st["model"])
    ds_set = tm.DSWithDate(*panels["test"], G, wf.seg_indices(blob, "test"), False)
    ps, ys, dd = tm.predict(model, DataLoader(ds_set, batch_size=2048, shuffle=False))
    # dd 与 predict 输出逐样本对齐；仪器索引直接取 dataset.samples
    ii = [ds_set.samples[k][1] for k in range(len(ds_set))]
    dates = [meta["dates"][t] for t in dd]
    insts = meta["insts"]
    rec = pd.DataFrame({"d": dates, "ii": ii, "p": ps, "y": ys})
    rec["code"] = [insts[i] for i in rec["ii"]]
    os.makedirs(CACHE, exist_ok=True)
    rec.to_pickle(outp)
    print(f"fold{fid} test preds: {len(rec)} rows, {rec.d.nunique()} days -> {outp}")


def load_all():
    frames = []
    for f in sorted(os.listdir(CACHE)):
        if f.startswith("fold_") and f.endswith(".pkl"):
            frames.append(pd.read_pickle(os.path.join(CACHE, f)))
    if not frames:
        raise SystemExit("无缓存，先跑 infer")
    df = pd.concat(frames, ignore_index=True)
    df["d"] = pd.to_datetime(df["d"])
    df["year"] = df.d.dt.year
    return df


def bucketize(df):
    """逐日排名分桶，返回 桶label × 日 的超额序列长表。"""
    rows = []
    for d, g in df.groupby("d"):
        g = g[np.isfinite(g.p) & np.isfinite(g.y)]
        if len(g) < 60:
            continue
        g = g.sort_values("p", ascending=False).reset_index(drop=True)
        mu = g.y.mean()
        r = np.arange(1, len(g) + 1)
        for lo, hi in BUCKETS:
            m = (r >= lo) & (r <= hi)
            if m.sum() == 0:
                continue
            rows.append({"d": d, "year": g.d.iloc[0].year if hasattr(g.d, "dt") else pd.Timestamp(d).year,
                         "bucket": f"{lo}" if lo == hi else f"{lo}-{hi}",
                         "excess": float(g.y[m].mean() - mu), "n": int(m.sum())})
    return pd.DataFrame(rows)


def report():
    df = load_all()
    b = bucketize(df)
    print(f"== 名次区分度（{df.d.nunique()} 个测试日，12 fold 测试段池化）==\n")
    print(f"{'名次桶':>8} | {'日均超额':>9} | {'t 值':>6} | {'样本外23-26':>11} | {'23':>7} {'24':>7} {'25':>7} {'26':>7}")
    for lo, hi in BUCKETS:
        lab = f"{lo}" if lo == hi else f"{lo}-{hi}"
        s = b[b.bucket == lab].excess
        t = s.mean() / (s.std() / np.sqrt(len(s))) if s.std() > 0 else np.nan
        o = b[(b.bucket == lab) & (b.year.isin(OOS_YEARS))].excess
        yy = {y: b[(b.bucket == lab) & (b.year == y)].excess.mean() for y in OOS_YEARS}
        print(f"{lab:>8} | {s.mean()*100:+8.4f}% | {t:6.2f} | {o.mean()*100:+10.4f}% | "
              + " ".join(f"{yy[y]*100:+6.3f}" for y in OOS_YEARS))

    # rank1 细看
    g = df[np.isfinite(df.p) & np.isfinite(df.y)].groupby("d").apply(
        lambda x: (lambda s: pd.Series({
            "top1": s.y.iloc[0] - s.y.mean(),
            "top1_vs_rest": s.y.iloc[0] - s.y.iloc[1:10].mean(),
            "top10_vs_rest": s.y.iloc[:10].mean() - s.y.iloc[10:].mean()}))(
            x.sort_values("p", ascending=False))).reset_index()
    g["year"] = pd.to_datetime(g.d).dt.year
    print(f"\n== rank1 细看 ==")
    print(f"rank1 日度超额: mean={g.top1.mean()*100:+.4f}%  std={g.top1.std()*100:.4f}%  "
          f"正比率={(g.top1 > 0).mean():.1%}  t={g.top1.mean()/(g.top1.std()/np.sqrt(len(g))):.2f}")
    print(f"rank1 vs top2-10 等权: 日均={g.top1_vs_rest.mean()*100:+.4f}%  "
          f"胜率={(g.top1_vs_rest > 0).mean():.1%}  t={g.top1_vs_rest.mean()/(g.top1_vs_rest.std()/np.sqrt(len(g))):.2f}")
    for y in OOS_YEARS:
        gy = g[g.year == y]
        if len(gy):
            print(f"  {y}: rank1超额 {gy.top1.mean()*100:+.4f}%  胜率 {(gy.top1 > 0).mean():.0%}  "
                  f"vs2-10 {gy.top1_vs_rest.mean()*100:+.4f}%")
    out = os.path.join(CACHE, "rank_buckets.csv")
    b.to_csv(out, index=False)
    print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["infer", "report"])
    ap.add_argument("--folds", default="0-11")
    a = ap.parse_args()
    if a.cmd == "infer":
        for fid in parse_folds(a.folds):
            infer_one(fid)
    else:
        report()


if __name__ == "__main__":
    main()
