# -*- coding: utf-8 -*-
"""rank_decomp.py — 非平稳下模型排名反转的三件套分解（arXiv 2611.04116 的操作化）。

对 fold 8-11 的测试段，逐日截面分解每个模型的 OOS 表现：
  R2_total : 池化截面去均后 1 - MSE/Var(y)
  Align    : 对齐上限 rho^2（完美幅度校准下可达的 R2）
  AggrLoss : 误校准激进度损失 = Align - R2_total（幅度缩放错误浪费的部分）
  Instab   : 预测不稳定性 = 日 IC 截面标准差 / |均值|（ICIR 倒数维度）+ 负 IC 日占比

由此可检验：fold 间模型排名反转，到底由对齐波动驱动（真信号差异）
还是由激进度/不稳定性驱动（校准与噪声问题）。

  python rank_decomp.py <fold_id>     # 跑单个 fold 的全部模型，断点续跑
  python rank_decomp.py summary       # 跨 fold 排名反转汇总
"""
import os, sys, json
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import train_models as tm
import wf_pipeline as wf

OUT_DIR = os.path.join(WS, "data", "live", "rank_decomp")
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS = {  # 与 increment_test.GROUPS 对齐（排除已淘汰的 margin）
    "dragon": ["LBCNT", "LIMUC"],
    "min": ["IVOL", "HEADR"],
    "evolve1": ["RPOS", "RPOS_D5"],
    "evolve2": ["RVOL", "RGAP"],
    "corrx": ["RC20"],
}


class ALSTMN(tm.ALSTM):
    def __init__(self, d_feat, hidden=64, num_layers=2):
        super().__init__(d_feat=d_feat, hidden=hidden, num_layers=num_layers)

    def forward(self, x):
        import torch
        out, _ = self.rnn(x)
        w = torch.softmax(self.att(out), dim=1)
        return self.fc((out * w).sum(dim=1)).squeeze(-1)


def model_specs():
    specs = [("base", None)]
    specs += [(g, cols) for g, cols in GROUPS.items()]
    return specs


def decompose(ps, ys, dates_idx, meta_dates):
    """返回逐日 r_t 与池化三件套。"""
    df = pd.DataFrame({"d": [meta_dates[i] for i in dates_idx], "p": ps, "y": ys}).dropna()
    r_days, keep = [], []
    for d, g in df.groupby("d"):
        if len(g) < 30:
            continue
        p, y = g["p"].to_numpy(), g["y"].to_numpy()
        p = p - p.mean(); y = y - y.mean()
        sp, sy = p.std(), y.std()
        if sp < 1e-12 or sy < 1e-12:
            continue
        r_days.append(float(np.corrcoef(p, y)[0, 1]))
        keep.append((p, y))
    r = np.array(r_days)
    P = np.concatenate([k[0] for k in keep])
    Y = np.concatenate([k[1] for k in keep])
    vy = float(np.var(Y))
    r2_total = 1.0 - float(np.mean((Y - P) ** 2)) / vy
    rho = float(np.corrcoef(P, Y)[0, 1])
    align = rho ** 2
    bstar = float(np.cov(P, Y)[0, 1] / np.var(P))
    aggr_loss = align - r2_total
    return {
        "days": int(len(r)),
        "R2_total": round(r2_total, 6),
        "Align_rho2": round(align, 6),
        "rho": round(rho, 6),
        "AggrLoss": round(aggr_loss, 6),
        "b_star": round(bstar, 4),
        "IC_mean": round(float(r.mean()), 6),
        "IC_std": round(float(r.std()), 6),
        "Instab_CV": round(float(r.std() / (abs(r.mean()) + 1e-12)), 4),
        "NegDayShare": round(float((r < 0).mean()), 4),
    }


def run_fold(fold_id):
    import torch
    from torch.utils.data import DataLoader
    cfg = wf.load_config()
    mc = cfg["models"]["alstm"]
    for name, cols in model_specs():
        out = os.path.join(OUT_DIR, f"fold{fold_id}_{name}.json")
        if os.path.exists(out):
            print(f"skip fold{fold_id} {name} (done)")
            continue
        ck = os.path.join(WS, "ckpt", "wf",
                          "alstm" if cols is None else f"alstm_{name}",
                          f"fold_{fold_id}.pt.best")
        if not os.path.exists(ck):
            print(f"MISSING ckpt fold{fold_id} {name}")
            continue
        if cols is None:
            blob = wf.load_fold(fold_id)
            model = tm.ALSTM(hidden=mc.get("hidden", 64), num_layers=mc.get("num_layers", 2)).to(tm.DEVICE)
        else:
            with open(os.path.join(wf.PREP_WF, f"fold_{fold_id}_{name}.pkl"), "rb") as f:
                blob = pickle.load(f)
            model = ALSTMN(158 + len(cols), hidden=mc.get("hidden", 64),
                           num_layers=mc.get("num_layers", 2)).to(tm.DEVICE)
        st = torch.load(ck, map_location=tm.DEVICE, weights_only=False)
        model.load_state_dict(st["model"])
        panels, G, meta = blob["panels"], blob["G"], blob["meta"]
        ds_set = tm.DSWithDate(*panels["test"], G, wf.seg_indices(blob, "test"), False)
        ps, ys, dd = tm.predict(model, DataLoader(ds_set, batch_size=2048, shuffle=False))
        res = decompose(ps, ys, dd, meta["dates"])
        res.update({"fold": fold_id, "model": name})
        json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
        print(f"fold{fold_id} {name}: {res}", flush=True)


def summary():
    import glob as gb
    rows = []
    for f in sorted(gb.glob(os.path.join(OUT_DIR, "fold*_*.json"))):
        rows.append(json.load(open(f)))
    df = pd.DataFrame(rows)
    folds = sorted(df["fold"].unique())
    models = sorted(df["model"].unique())
    print(f"{'model':<9}" + "".join(f"{f'>fold{f}':>9}" for f in folds) + f"{'>均值':>9}{'>反转源':>28}")
    rank_store = {}
    for m in models:
        sub = df[df["model"] == m].set_index("fold")
        r2 = [sub.loc[f, "R2_total"] for f in folds]
        align = [sub.loc[f, "Align_rho2"] for f in folds]
        cv = [sub.loc[f, "Instab_CV"] for f in folds]
        # 反转来源：R2 的 fold 间标准差中，Align 波动贡献 vs 激进度波动贡献
        sa, sg = np.std(align), np.std([a - r for a, r in zip(align, r2)])
        src = f"align波动{sa:.4f}/激进度{sg:.4f}"
        print(f"{m:<9}" + "".join(f"{v:>9.4f}" for v in r2) + f"{np.mean(r2):>9.4f}{src:>28}")
        rank_store[m] = dict(zip(folds, r2))
    # 逐 fold 排名 + Kendall 一致性
    from itertools import combinations
    print("\n逐 fold 排名（按 R2_total）：")
    ranks = {}
    for f in folds:
        order = df[df["fold"] == f].sort_values("R2_total", ascending=False)["model"].tolist()
        ranks[f] = order
        print(f"  fold{f}: {' > '.join(order)}")
    concord = []
    for a, b in combinations(folds, 2):
        ra = {m: i for i, m in enumerate(ranks[a])}
        rb = {m: i for i, m in enumerate(ranks[b])}
        n_c = sum(1 for x, y in combinations(models, 2)
                  if (ra[x] - ra[y]) * (rb[x] - rb[y]) > 0)
        n_d = sum(1 for x, y in combinations(models, 2)
                  if (ra[x] - ra[y]) * (rb[x] - rb[y]) < 0)
        concord.append(n_c / (n_c + n_d + 1e-12))
    print(f"\nKendall 跨 fold 排名一致性: mean={np.mean(concord):.3f} "
          f"（1=完全稳定，0.5=随机）")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        import pickle  # noqa: F401  (run_fold 内延迟使用)
        run_fold(int(sys.argv[1]))
