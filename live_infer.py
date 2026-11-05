# -*- coding: utf-8 -*-
"""P5 实时信号链路：最新 fold 的 ALSTM 检查点对最新交易日出 top50 信号。

两阶段设计：
  bootstrap  首次全量构建——按最新 fold 训练窗 fit Alpha158 处理器（与 walk-forward
             训练同口径，无泄漏），拉取截至 asof 的特征，保存拟合处理器 + 特征列序
             + 出信号。Alpha158 全跨度计算，较慢（走托管任务）。
  daily      日常快速路径——加载已存处理器，只计算近 90 自然日特征，分钟级完成，
             供每日巡检任务调用。

用法：
  python live_infer.py bootstrap [--asof 2026-11-03]
  python live_infer.py daily [--asof 2026-11-03]
"""
import os, sys, json, time, pickle, argparse, datetime
import numpy as np
import pandas as pd
import torch
import yaml

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
LIVE = os.path.join(WS, "data", "live")
os.makedirs(LIVE, exist_ok=True)
PROC_PKL = os.path.join(LIVE, "processors_alstm.pkl")
ORDER_JSON = os.path.join(LIVE, "feature_order.json")
LOOKBACK = 8


def load_cfg():
    with open(os.path.join(WS, "walkforward.yaml")) as f:
        return yaml.safe_load(f)


def latest_fold(cfg):
    import wf_pipeline as wf
    folds = wf.generate_folds(cfg)
    return folds[-1]


def default_asof():
    cal = os.path.expanduser("~/.qlib/qlib_data/cn_data/calendars/day.txt")
    return open(cal).read().strip().split("\n")[-1]


def raw_live_frame(cfg, asof, procs):
    """近窗口原始特征 + 应用拟合处理器（bootstrap/daily 共用，约 1 分钟）。"""
    from qlib.data.dataset.loader import QlibDataLoader
    from qlib.contrib.data.handler import Alpha158
    a158 = Alpha158.__new__(Alpha158)          # 仅借字段配置，不触发全量 setup
    conf = {"feature": a158.get_feature_config()}
    loader = QlibDataLoader(config=conf)
    live_start = (pd.Timestamp(asof) - pd.Timedelta(days=120)).date()
    raw = loader.load(instruments=cfg["data"]["universe"],
                      start_time=str(live_start), end_time=asof)
    for p in procs:
        raw = p(raw)
    feat = raw["feature"]
    print(f"live frame {feat.shape}, window {live_start}..{asof}", flush=True)
    return feat


def fetch_live_frame_bootstrap(cfg, fold, asof):
    """全量构建：handler 仅在 fold 训练窗上 setup+fit（约 12 分钟），
    处理器存盘后 live 窗口走 raw_live_frame（避免 learn 链 DropnaLabel
    把最后 2 个交易日标签空行整行删掉的问题）。"""
    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.data.handler import Alpha158
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    t0 = time.time()
    h = Alpha158(
        instruments=cfg["data"]["universe"],
        start_time=fold["train"][0], end_time=fold["train"][1],
        fit_start_time=fold["train"][0], fit_end_time=fold["train"][1],
        infer_processors=cfg["data"]["processors"]["infer_processors"],
        learn_processors=cfg["data"]["processors"]["learn_processors"],
    )
    print(f"handler fit on {fold['train']} in {time.time()-t0:.0f}s", flush=True)
    pickle.dump(h.infer_processors, open(PROC_PKL, "wb"))
    feat = raw_live_frame(cfg, asof, h.infer_processors)
    json.dump(list(feat.columns), open(ORDER_JSON, "w"))
    return feat


def fetch_live_frame_daily(cfg, asof):
    """日常路径：QLibDataLoader 原始特征 + 复用已存拟合处理器。"""
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    procs = pickle.load(open(PROC_PKL, "rb"))
    feat = raw_live_frame(cfg, asof, procs)
    order = json.load(open(ORDER_JSON))
    return feat.reindex(columns=order)


def build_panel(feat):
    dt = feat.index.get_level_values("datetime")
    ii = feat.index.get_level_values("instrument")
    dates = sorted(dt.unique())
    insts = sorted(ii.unique())
    d_idx = {d: i for i, d in enumerate(dates)}
    i_idx = {c: j for j, c in enumerate(insts)}
    F = np.full((len(dates), len(insts), feat.shape[1]), np.nan, dtype=np.float32)
    di = pd.Series(dt).map(d_idx).to_numpy()
    ij = pd.Series(ii).map(i_idx).to_numpy()
    F[di, ij, :] = feat.to_numpy(dtype=np.float32)
    return F, dates, insts


@torch.no_grad()
def predict_scores(F, dates, insts, asof, cfg, fold):
    t = max(i for i, d in enumerate(dates) if str(d)[:10] <= asof)
    if t < LOOKBACK - 1:
        raise SystemExit(f"可用历史不足 {LOOKBACK} 天")
    win = np.isfinite(F[t - LOOKBACK + 1: t + 1]).all(axis=(0, 2))
    ok = np.where(win)[0]
    import train_models as tm
    mc = cfg["models"]["alstm"]
    model = tm.ALSTM(hidden=mc.get("hidden", 64), num_layers=mc.get("num_layers", 2)).to(tm.DEVICE)
    ck = os.path.join(WS, "ckpt", "wf", "alstm", f"fold_{fold['id']}.pt.best")
    st = torch.load(ck, map_location=tm.DEVICE, weights_only=False)
    model.load_state_dict(st["model"])
    model.eval()
    X = torch.from_numpy(F[t - LOOKBACK + 1: t + 1][:, ok, :].transpose(1, 0, 2))
    scores = []
    for k in range(0, len(ok), 1024):
        scores.append(model(X[k:k + 1024].to(tm.DEVICE)).cpu().numpy())
    s = pd.DataFrame({"code": [insts[i].upper().replace("SH", "").replace("SZ", "") for i in ok],
                      "instrument": [insts[i] for i in ok],
                      "score": np.concatenate(scores)}, ).sort_values("score", ascending=False)
    return s, str(dates[t])[:10], ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bootstrap", "daily"])
    ap.add_argument("--asof", default=None)
    a = ap.parse_args()
    asof = a.asof or default_asof()
    cfg = load_cfg()
    fold = latest_fold(cfg)
    print(f"asof={asof}  fold={fold['id']} train={fold['train']} valid={fold['valid']} test={fold['test']}", flush=True)
    t0 = time.time()
    if a.cmd == "bootstrap":
        feat = fetch_live_frame_bootstrap(cfg, fold, asof)
    else:
        if not (os.path.exists(PROC_PKL) and os.path.exists(ORDER_JSON)):
            raise SystemExit("先跑 bootstrap")
        feat = fetch_live_frame_daily(cfg, asof)
    F, dates, insts = build_panel(feat)
    s, d, ck = predict_scores(F, dates, insts, asof, cfg, fold)
    out = os.path.join(LIVE, f"signal_{d}.csv")
    s.to_csv(out, index=False)
    print(f"signal date={d} universe={len(s)} ckpt={os.path.basename(ck)} elapsed={time.time()-t0:.0f}s")
    print("TOP10:")
    print(s.head(10).to_string(index=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
