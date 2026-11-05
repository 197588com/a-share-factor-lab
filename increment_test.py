# -*- coding: utf-8 -*-
"""increment_test.py — 新因子组合增量测试（fold11 口径）。

把候选特征组追加到 Alpha158 之后，处理器与训练同口径：RobustZScoreNorm 按个股在
fold11 训练窗(2019-2024) fit，clip ±3、NaN 填 0。ALSTM 单独实例化 d_feat=N
（train_models.ALSTM.forward 硬切 D_FEAT=158，不能直接复用）。训练/评估逻辑镜像
wf_pipeline.train_fold/eval_fold，检查点写 ckpt/wf/alstm_<组名>/，不触碰线上 fold11。

  python increment_test.py build [margin|dragon]   # fold_11_<组名>.pkl 面板
  python increment_test.py train [组名]            # 断点续训
  python increment_test.py eval  [组名]            # valid/test RankIC + 对比报告

特征组：
  margin  MBR/MBAL/MSRT        -> 161 维（已淘汰，2026-11-04）
  dragon  LBCNT/LIMUC          -> 160 维（待测，2026-11-04）
"""
import os, sys, time, json, pickle
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import train_models as tm
import wf_pipeline as wf

GROUPS = {
    "margin": ("margin_feat.parquet", ["MBR", "MBAL", "MSRT"], {}),
    "dragon": ("dragon_feat.parquet", ["LBCNT", "LIMUC"], {}),
    "corrx": ("corrx_feat.parquet", ["RC20"], {}),
    # 分钟族：2023 年起才有数据，2019-2022 训练窗全零——零值视为缺失做归一化统计
    "min": ("min_feat.parquet", ["IVOL", "HEADR"], {"nonzero_stats": True}),
    # L2 进化第一批：range_pos 族（2015-2026 全覆盖，无零窗问题）
    "evolve1": ("evolve_feat.parquet", ["RPOS", "RPOS_D5"], {}),
    # L2 第二批：rpos_vol（量交互）+ rpos_gap（跳空方向交互），族内簇代表
    "evolve2": ("evolve2_feat.parquet", ["RVOL", "RGAP"], {}),
}
REPORT = os.path.join(WS, "data", "live", "increment_margin.json")


def group_paths(g):
    ckdir = os.path.join(WS, "ckpt", "wf", f"alstm_{g}")
    os.makedirs(ckdir, exist_ok=True)
    return (os.path.join(WS, "data", "prep_wf", f"fold_11_{g}.pkl"),
            ckdir, 158 + len(GROUPS[g][1]),
            os.path.join(WS, "data", "live", f"increment_{g}.json"))


def load_cfg():
    import yaml
    return yaml.safe_load(open(os.path.join(WS, "walkforward.yaml")))


# ---------------- 面板构建 ----------------
def build_panel(g):
    feat_file, cols, opts = GROUPS[g]
    cache, _, _, _ = group_paths(g)
    blob = wf.load_fold(11)
    panels, G, meta = blob["panels"], blob["G"], blob["meta"]
    dates, insts = meta["dates"], meta["insts"]
    nD, nI = len(dates), len(insts)
    d_feat = 158 + len(cols)
    print(f"fold11 base panel: nD={nD} nI={nI} feat={panels['train'][0].shape[2]} -> +{cols} = {d_feat}d")

    m = pd.read_parquet(os.path.join(WS, "data", "factor_decay", feat_file))[cols]
    di = {d: i for i, d in enumerate(dates)}
    ii = {c: j for j, c in enumerate(insts)}
    M = np.zeros((nD, nI, len(cols)), dtype=np.float32)
    dtv = m.index.get_level_values("datetime").date
    iv = m.index.get_level_values("instrument")
    ok = pd.Series(dtv).map(di).notna().to_numpy() & pd.Series(iv).map(ii).notna().to_numpy()
    M[pd.Series(dtv).map(di)[ok].astype(int).to_numpy(),
      pd.Series(iv).map(ii)[ok].astype(int).to_numpy(), :] = \
        m.to_numpy(dtype=np.float32)[ok]
    cover = np.isfinite(M).any(axis=2).mean()
    print(f"{g} aligned: coverage {cover:.1%}")

    # RobustZScoreNorm：训练窗 per-instrument fit（与 qlib 处理器同口径）
    tr0, tr1 = 0, len([d for d in dates if d <= pd.Timestamp("2024-12-31").date()])
    tr = M[tr0:tr1]
    if opts.get("nonzero_stats"):
        # 稀疏激活因子（零=无数据）：零值不参与统计，避免 2023 前全零段压垮 MAD
        tr = np.where(tr == 0, np.nan, tr)
    med = np.nanmedian(tr, axis=0)
    mad = np.nanmedian(np.abs(tr - med), axis=0)
    scale = 1.4826 * np.where(mad < 1e-12, np.nan, mad)
    Z = (M - med) / scale
    Z = np.clip(Z, -3, 3)
    Z = np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)

    panels2 = {}
    for seg in ("train", "valid", "test"):
        F, L = panels[seg]
        panels2[seg] = (np.concatenate([F, Z], axis=2), L)
        print(f"  {seg}: F {F.shape} -> {panels2[seg][0].shape}")
    with open(cache, "wb") as f:
        pickle.dump({"panels": panels2, "G": G, "meta": meta}, f)
    print(f"saved {cache}")


# ---------------- 训练 ----------------
class ALSTMN(tm.ALSTM):
    def __init__(self, d_feat, hidden=64, num_layers=2):
        super().__init__(d_feat=d_feat, hidden=hidden, num_layers=num_layers)

    def forward(self, x):
        import torch
        out, _ = self.rnn(x)
        w = torch.softmax(self.att(out), dim=1)
        return self.fc((out * w).sum(dim=1)).squeeze(-1)


def train(g):
    import torch
    from torch.utils.data import DataLoader
    cache, ckdir, d_feat, _ = group_paths(g)
    cfg = load_cfg()
    blob = pickle.load(open(cache, "rb"))
    panels, G, meta = blob["panels"], blob["G"], blob["meta"]
    mc = cfg["models"]["alstm"]
    tr = tm.WindowDS(*panels["train"], G, wf.seg_indices(blob, "train"), False)
    va = tm.DSWithDate(*panels["valid"], G, wf.seg_indices(blob, "valid"), False)
    print(f"[{g}] train={len(tr)} valid={len(va)} epochs={mc['epochs']} d_feat={d_feat}")

    model = ALSTMN(d_feat, hidden=mc.get("hidden", 64), num_layers=mc.get("num_layers", 2)).to(tm.DEVICE)
    ck = os.path.join(ckdir, "fold_11.pt")
    start_ep, best = 0, -1e9
    if os.path.exists(ck):
        st = torch.load(ck, map_location=tm.DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); start_ep = st["epoch"] + 1; best = st["best"]
        print(f"resume epoch {start_ep}, best {best:.4f}")
    opt = torch.optim.Adam(model.parameters(), lr=mc["lr"])
    lossf = torch.nn.MSELoss()
    for ep in range(start_ep, mc["epochs"]):
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
        print(f"ep{ep}: loss={tot/nb:.5f} valid RankIC={m['RankIC']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep, "best": best,
                    "fold": 11, "model_name": f"alstm_{g}"}, ck)
        if m["IC"] > best:
            best = m["IC"]
            torch.save({"model": model.state_dict(), "epoch": ep, "best": best,
                        "fold": 11, "model_name": f"alstm_{g}"}, ck + ".best")
    print(f"TRAIN DONE best_valid_IC={best:.4f}")


def evaluate(g):
    import torch
    from torch.utils.data import DataLoader
    cfg = load_cfg()
    cache, ckdir, d_feat, report = group_paths(g)
    blob = pickle.load(open(cache, "rb"))
    panels, G, meta = blob["panels"], blob["G"], blob["meta"]
    mc = cfg["models"]["alstm"]
    model = ALSTMN(d_feat, hidden=mc.get("hidden", 64), num_layers=mc.get("num_layers", 2)).to(tm.DEVICE)
    st = torch.load(os.path.join(ckdir, "fold_11.pt.best"), map_location=tm.DEVICE, weights_only=False)
    model.load_state_dict(st["model"])
    out = {}
    for seg, name in (("valid", "valid2025"), ("test", "test2026")):
        ds_set = tm.DSWithDate(*panels[seg], G, wf.seg_indices(blob, seg), False)
        ps, ys, dd = tm.predict(model, DataLoader(ds_set, batch_size=2048, shuffle=False))
        m = tm.daily_ic(ps, ys, [meta["dates"][i] for i in dd])
        out[name] = {k: round(float(m[k]), 4) if isinstance(m[k], (int, float)) else m[k]
                     for k in ("IC", "ICIR", "RankIC", "RankICIR", "days")}
        print(f"{name}: {out[name]}")
    out["baseline_alstm158"] = {"valid2025": {"RankIC": 0.0013}, "test2026": {"RankIC": -0.0010}}
    out["model"] = f"alstm_{g}{d_feat}"
    out["fold"] = 11
    json.dump(out, open(report, "w"), ensure_ascii=False, indent=2)
    print(f"saved {report}")


def main():
    cmd = sys.argv[1]
    g = sys.argv[2] if len(sys.argv) > 2 else "margin"
    if cmd == "build":
        build_panel(g)
    elif cmd == "train":
        train(g)
    elif cmd == "eval":
        evaluate(g)


if __name__ == "__main__":
    main()
