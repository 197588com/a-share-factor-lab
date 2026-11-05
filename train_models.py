# -*- coding: utf-8 -*-
"""ALSTM / MASTER 统一训练与评估（Alpha158, csi300, 8 日窗口）。

用法:
  python train_models.py prep          # 构建面板缓存
  python train_models.py train alstm   # 训练 ALSTM（自动断点续训）
  python train_models.py train master  # 训练 MASTER（自动断点续训）
  python train_models.py eval  <name>  # 评估测试段 IC/RankIC
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WS = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(WS, "data", "prep")
CKPT = os.path.join(WS, "ckpt")
os.makedirs(CKPT, exist_ok=True)
MARKET_CSV = os.path.join(WS, "master-repo", "data", "csi_market_information.csv")

LOOKBACK = 8
D_FEAT = 158
D_GATE = 63
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEGS = ["train", "valid", "test"]
PANEL_CACHE = os.path.join(PREP, "panels.pkl")


# ---------------- 面板构建 ----------------
def build_panels():
    frames = {s: pd.read_pickle(os.path.join(PREP, f"{s}.pkl")) for s in SEGS}
    dates = sorted(set().union(*[set(f.index.get_level_values("datetime").date) for f in frames.values()]))
    insts = sorted(set().union(*[set(f.index.get_level_values("instrument")) for f in frames.values()]))
    d_idx = {d: i for i, d in enumerate(dates)}
    i_idx = {c: j for j, c in enumerate(insts)}
    nD, nI = len(dates), len(insts)
    feat_cols = [c for c in frames["train"].columns if c[0] == "feature"]

    panels = {}
    for s in SEGS:
        F = np.full((nD, nI, D_FEAT), np.nan, dtype=np.float32)
        L = np.full((nD, nI), np.nan, dtype=np.float32)
        df = frames[s]
        dt = df.index.get_level_values("datetime").date
        ii = df.index.get_level_values("instrument").map(i_idx)
        di = pd.Series(dt).map(d_idx).to_numpy()
        F[di, ii, :] = df[feat_cols].to_numpy(dtype=np.float32)
        L[di, ii] = df["label"].iloc[:, 0].to_numpy(dtype=np.float32)
        panels[s] = (F, L)
        print(f"{s}: filled {np.sum(~np.isnan(L))} labels")

    # 市场门控特征（官方发布，已标准化，2008-2022）
    m = pd.read_csv(MARKET_CSV, index_col=0)
    m = m.iloc[1:].apply(pd.to_numeric, errors="coerce")
    m = m.dropna(how="all")
    m.index = pd.to_datetime(m.index, errors="coerce")
    m = m[m.index.notna()]
    m = m[~m.index.duplicated(keep="last")]
    m.index = m.index.date
    G = np.zeros((nD, D_GATE), dtype=np.float32)
    miss = 0
    for i, d in enumerate(dates):
        if d in m.index:
            G[i] = m.loc[d].to_numpy(dtype=np.float32)
        else:
            miss += 1
    print(f"market features: {miss}/{nD} dates missing")

    meta = {"dates": dates, "insts": insts}
    with open(PANEL_CACHE, "wb") as f:
        pickle.dump({"panels": panels, "G": G, "meta": meta}, f)
    print("panels saved:", PANEL_CACHE)


def load_panels():
    with open(PANEL_CACHE, "rb") as f:
        return pickle.load(f)


# ---------------- 数据集 ----------------
class WindowDS(Dataset):
    def __init__(self, F, L, G, seg_dates_idx, with_gate):
        self.F, self.L, self.G = F, L, G
        self.with_gate = with_gate
        self.samples = []
        Fm = np.isfinite(F).all(axis=2)  # nD x nI 特征完整
        for t in seg_dates_idx:
            if t < LOOKBACK - 1:
                continue
            win = Fm[t - LOOKBACK + 1: t + 1].all(axis=0)
            lab = np.isfinite(L[t])
            ok = win & lab
            if with_gate and not np.isfinite(G[t]).all():
                ok &= False
            for i in np.where(ok)[0]:
                self.samples.append((t, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        t, i = self.samples[k]
        x = self.F[t - LOOKBACK + 1: t + 1, i, :]
        if self.with_gate:
            g = np.tile(self.G[t], (LOOKBACK, 1))
            x = np.concatenate([x, g], axis=1)
        return torch.from_numpy(x), torch.tensor(self.L[t, i], dtype=torch.float32)


# ---------------- 模型 ----------------
class ALSTM(nn.Module):
    """qlib pytorch_alstm 等价结构：2层LSTM + 时间注意力"""
    def __init__(self, d_feat=158, hidden=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(d_feat, hidden, num_layers, batch_first=True, dropout=dropout)
        self.att = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x[:, :, :D_FEAT])
        w = torch.softmax(self.att(out), dim=1)
        z = (out * w).sum(dim=1)
        return self.fc(z).squeeze(-1)


class MASTER(nn.Module):
    """官方实现包装（master-repo/master.py）"""
    def __init__(self, d_model=128, t_nhead=4, s_nhead=2, dropout=0.5, beta=5.0):
        super().__init__()
        sys.path.insert(0, os.path.join(WS, "master-repo"))
        from master import MASTER as _M
        self.net = _M(D_FEAT, d_model, t_nhead, s_nhead, dropout, dropout,
                      gate_input_start_index=D_FEAT, gate_input_end_index=D_FEAT + D_GATE, beta=beta)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------- 评估 ----------------
def daily_ic(preds, labels, dates_list):
    df = pd.DataFrame({"d": dates_list, "p": preds, "y": labels}).dropna()
    ic, ric = [], []
    for _, g in df.groupby("d"):
        if len(g) < 30:
            continue
        ic.append(np.corrcoef(g["p"], g["y"])[0, 1])
        ric.append(g["p"].corr(g["y"], method="spearman"))
    ic, ric = np.array(ic), np.array(ric)
    return {"IC": ic.mean(), "ICIR": ic.mean() / (ic.std() + 1e-12),
            "RankIC": ric.mean(), "RankICIR": ric.mean() / (ric.std() + 1e-12), "days": len(ic)}


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ps, ys, ds = [], [], []
    for x, y, d in loader:
        ps.append(model(x.to(DEVICE)).cpu().numpy())
        ys.append(y.numpy())
        ds.append(d.numpy())
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(ds)


class DSWithDate(WindowDS):
    def __getitem__(self, k):
        t, i = self.samples[k]
        x = self.F[t - LOOKBACK + 1: t + 1, i, :]
        if self.with_gate:
            g = np.tile(self.G[t], (LOOKBACK, 1))
            x = np.concatenate([x, g], axis=1)
        return torch.from_numpy(x), torch.tensor(self.L[t, i], dtype=torch.float32), t


# ---------------- 训练 ----------------
def train(name):
    data = load_panels()
    panels, G, meta = data["panels"], data["G"], data["meta"]
    d_idx = {d: i for i, d in enumerate(meta["dates"])}
    seg_idx = {s: np.array([d_idx[d] for d in meta["dates"]
                            if panels[s][1][d_idx[d]].shape and not np.isnan(panels[s][1][d_idx[d]]).all()])
               for s in SEGS}
    with_gate = name == "master"
    tr = WindowDS(*panels["train"], G, seg_idx["train"], with_gate)
    va = DSWithDate(*panels["valid"], G, seg_idx["valid"], with_gate)
    print(f"train samples: {len(tr)}, valid samples: {len(va)}")

    model = ALSTM().to(DEVICE) if name == "alstm" else MASTER().to(DEVICE)
    ck = os.path.join(CKPT, f"{name}.pt")
    start_ep, best = 0, -1e9
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); start_ep = st["epoch"] + 1; best = st["best"]
        print(f"resumed from epoch {start_ep}, best valid IC {best:.4f}")

    EPOCHS = 10
    opt = torch.optim.Adam(model.parameters(), lr=1e-3 if name == "alstm" else 3e-5)
    lossf = nn.MSELoss()
    for ep in range(start_ep, EPOCHS):
        model.train()
        tl = DataLoader(tr, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        vl = DataLoader(va, batch_size=1024, shuffle=False)
        ps, ys, ds = predict(model, vl)
        m = daily_ic(ps, ys, [meta["dates"][i] for i in ds])
        print(f"ep{ep}: loss={tot/nb:.5f} valid IC={m['IC']:.4f} RankIC={m['RankIC']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if m["IC"] > best:
            best = m["IC"]
            torch.save({"model": model.state_dict(), "epoch": ep, "best": best}, ck)
    print(f"TRAIN DONE best_valid_IC={best:.4f}")


def ev(name):
    data = load_panels()
    panels, G, meta = data["panels"], data["G"], data["meta"]
    d_idx = {d: i for i, d in enumerate(meta["dates"])}
    seg_idx = {s: np.array([d_idx[d] for d in meta["dates"]
                            if not np.isnan(panels[s][1][d_idx[d]]).all()]) for s in SEGS}
    with_gate = name == "master"
    te = DSWithDate(*panels["test"], G, seg_idx["test"], with_gate)
    model = ALSTM().to(DEVICE) if name == "alstm" else MASTER().to(DEVICE)
    st = torch.load(os.path.join(CKPT, f"{name}.pt"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(st["model"])
    loader = DataLoader(te, batch_size=2048, shuffle=False)
    ps, ys, ds = predict(model, loader)
    m = daily_ic(ps, ys, [meta["dates"][i] for i in ds])
    print(f"[{name}] TEST  IC={m['IC']:.4f} ICIR={m['ICIR']:.3f} RankIC={m['RankIC']:.4f} RankICIR={m['RankICIR']:.3f} ({m['days']} days)")
    return m


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        build_panels()
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "eval":
        ev(sys.argv[2])
