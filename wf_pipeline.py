# -*- coding: utf-8 -*-
"""walk-forward 深度模型自动再训练管线（framework_design.md 1.3 + 1.4c/1.4d）。

设计要点：
- 滚动窗口完全由 walkforward.yaml 生成，禁止手写切分（1.3 硬约束）
- 每个 fold 的 Alpha158 处理器只在 fold 训练窗上 fit（防止归一化泄漏）
- 每 fold 每模型 MLflow 记录：config 哈希、数据快照 ID、超参、逐 epoch 指标、测试段指标、模型权重
- 测试段指标按 HMM regime 分组（1.4b：所有表现按市场状态记录）
- 测试段日度 IC 序列落盘后供 DSR 审计（1.4c 上线前强制闸门）
- 全部断点续训：任何时刻中断，重跑同一命令继续

用法：
  python wf_pipeline.py folds                        # 打印 fold 切分表
  python wf_pipeline.py prep --fold N [--legacy-cache]   # 构建 fold 面板缓存
  python wf_pipeline.py train --model alstm --fold N [--epochs K]
  python wf_pipeline.py eval  --model alstm --fold N
  python wf_pipeline.py run   --model alstm --folds 0-7 [--epochs K]  # prep+train+eval 一键
  python wf_pipeline.py audit --model alstm [--trials N]              # DSR 过拟合审计
"""
import os, sys, json, time, hashlib, argparse, pickle
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import train_models as tm            # 复用 ALSTM/MASTER/数据集/IC 评估（P0 基线同源）
import yaml

CFG_PATH = os.path.join(WS, "walkforward.yaml")
PREP_WF = os.path.join(WS, "data", "prep_wf")
MLRUNS = os.path.join(WS, "mlruns")
DB = os.path.join(WS, "data", "factor_lab.duckdb")
LEGACY_PREP = os.path.join(WS, "data", "prep")   # P0 缓存（fit 2015-2019），仅供 smoke


def load_config():
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_sha(cfg):
    return hashlib.sha1(yaml.dump(cfg, sort_keys=True).encode()).hexdigest()[:12]


def generate_folds(cfg):
    """由配置推导滚动窗口（年粒度），返回 fold 列表。"""
    wf = cfg["walkforward"]
    start = pd.Timestamp(cfg["data"]["start"])
    end = pd.Timestamp(cfg["data"]["end"])
    ty, vy, wy = wf["train_years"], wf["valid_years"], wf["test_years"]
    folds = []
    f0 = start
    i = 0
    while True:
        tr_s, tr_e = f0, f0 + pd.DateOffset(years=ty) - pd.Timedelta(days=1)
        va_s, va_e = tr_e + pd.Timedelta(days=1), tr_e + pd.DateOffset(years=vy)
        te_s, te_e = va_e + pd.Timedelta(days=1), va_e + pd.DateOffset(years=wy)
        if te_e > end:
            break
        folds.append({"id": i,
                      "train": (str(tr_s.date()), str(tr_e.date())),
                      "valid": (str(va_s.date()), str(va_e.date())),
                      "test": (str(te_s.date()), str(te_e.date()))})
        f0 = f0 + pd.DateOffset(months=wf["step_months"])
        i += 1
    return folds


# ---------------- 数据：每 fold 独立 fit 处理器 ----------------
def fetch_fold_frames(fold, cfg):
    """Alpha158 在 fold 训练窗 fit，分别取 train/valid/test 三段。"""
    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.data.handler import Alpha158
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    h = Alpha158(
        instruments=cfg["data"]["universe"],
        start_time=fold["train"][0], end_time=fold["test"][1],
        fit_start_time=fold["train"][0], fit_end_time=fold["train"][1],
        infer_processors=cfg["data"]["processors"]["infer_processors"],
        learn_processors=cfg["data"]["processors"]["learn_processors"],
    )
    out = {}
    for seg in ("train", "valid", "test"):
        s, e = fold[seg]
        t0 = time.time()
        df = h.fetch(selector=slice(s, e), level="datetime", col_set=["feature", "label"])
        out[seg] = df
        print(f"  fold{fold['id']} {seg} {s}..{e}: {df.shape} ({time.time()-t0:.0f}s)", flush=True)
    return out


def build_fold_panels(frames):
    """三段 DataFrame -> (panels, G, meta)，复用 P0 口径（含市场门控 63 维）。"""
    segs = ["train", "valid", "test"]
    dates = sorted(set().union(*[set(f.index.get_level_values("datetime").date) for f in frames.values()]))
    insts = sorted(set().union(*[set(f.index.get_level_values("instrument")) for f in frames.values()]))
    d_idx = {d: i for i, d in enumerate(dates)}
    i_idx = {c: j for j, c in enumerate(insts)}
    nD, nI = len(dates), len(insts)
    feat_cols = [c for c in frames["train"].columns if c[0] == "feature"]
    panels = {}
    for s in segs:
        F = np.full((nD, nI, tm.D_FEAT), np.nan, dtype=np.float32)
        L = np.full((nD, nI), np.nan, dtype=np.float32)
        df = frames[s]
        dt = df.index.get_level_values("datetime").date
        ii = df.index.get_level_values("instrument").map(i_idx)
        di = pd.Series(dt).map(d_idx).to_numpy()
        F[di, ii, :] = df[feat_cols].to_numpy(dtype=np.float32)
        L[di, ii] = df["label"].iloc[:, 0].to_numpy(dtype=np.float32)
        panels[s] = (F, L)
    m = pd.read_csv(tm.MARKET_CSV, index_col=0)
    m = m.iloc[1:].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    m.index = pd.to_datetime(m.index, errors="coerce")
    m = m[m.index.notna()]
    m = m[~m.index.duplicated(keep="last")]
    m.index = m.index.date
    G = np.zeros((nD, tm.D_GATE), dtype=np.float32)
    miss = sum(1 for d in dates if d not in m.index)
    for i, d in enumerate(dates):
        if d in m.index:
            G[i] = m.loc[d].to_numpy(dtype=np.float32)
    print(f"  market gate missing: {miss}/{nD} dates")
    return panels, G, {"dates": dates, "insts": insts}


def prep_fold(fold, cfg, legacy_cache=False):
    os.makedirs(PREP_WF, exist_ok=True)
    cache = os.path.join(PREP_WF, f"fold_{fold['id']}.pkl")
    if os.path.exists(cache):
        print(f"fold{fold['id']} panel cached: {cache}")
        return cache
    if legacy_cache:
        frames = {s: pd.read_pickle(os.path.join(LEGACY_PREP, f"{s}.pkl")) for s in ("train", "valid", "test")}
        print("  using legacy P0 cache (fit 2015-2019) — smoke only")
    else:
        frames = fetch_fold_frames(fold, cfg)
    panels, G, meta = build_fold_panels(frames)
    blob = {"panels": panels, "G": G, "meta": meta}
    with open(cache, "wb") as f:
        pickle.dump(blob, f)
    sha = hashlib.sha1(open(cache, "rb").read()).hexdigest()[:12]
    print(f"fold{fold['id']} panel saved: {cache} (data_sha={sha})")
    return cache


def data_sha(cache):
    return hashlib.sha1(open(cache, "rb").read()).hexdigest()[:12]


def load_fold(fold_id):
    with open(os.path.join(PREP_WF, f"fold_{fold_id}.pkl"), "rb") as f:
        return pickle.load(f)


def seg_indices(blob, seg):
    """某段有标签的日期索引（标签非全 NaN 即该段交易日）。"""
    panels, meta = blob["panels"], blob["meta"]
    idx = [i for i, d in enumerate(meta["dates"])
           if not np.isnan(panels[seg][1][i]).all()]
    return np.array(idx)


# ---------------- 训练（断点续训） ----------------
def ckpt_path(cfg, model, fold_id):
    d = os.path.join(WS, cfg["tracking"]["ckpt_dir"], model)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"fold_{fold_id}.pt")


def train_fold(cfg, model_name, fold, max_epochs=None):
    import torch
    from torch.utils.data import DataLoader
    blob = load_fold(fold["id"])
    panels, G, meta = blob["panels"], blob["G"], blob["meta"]
    mc = cfg["models"][model_name]
    epochs_target = max_epochs or mc["epochs"]
    with_gate = model_name == "master"
    tr = tm.WindowDS(*panels["train"], G, seg_indices(blob, "train"), with_gate)
    va = tm.DSWithDate(*panels["valid"], G, seg_indices(blob, "valid"), with_gate)
    print(f"fold{fold['id']} {model_name}: train={len(tr)} valid={len(va)} target_epochs={epochs_target}")

    model = tm.ALSTM(hidden=mc.get("hidden", 64), num_layers=mc.get("num_layers", 2)).to(tm.DEVICE) \
        if model_name == "alstm" else tm.MASTER(d_model=mc.get("d_model", 128)).to(tm.DEVICE)
    ck = ckpt_path(cfg, model_name, fold["id"])
    start_ep, best = 0, -1e9
    if os.path.exists(ck):
        st = torch.load(ck, map_location=tm.DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); start_ep = st["epoch"] + 1; best = st["best"]
        print(f"  resumed epoch {start_ep}, best valid IC {best:.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=mc["lr"])
    lossf = torch.nn.MSELoss()
    hist = []
    for ep in range(start_ep, epochs_target):
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
        print(f"  ep{ep}: loss={tot/nb:.5f} valid IC={m['IC']:.4f} RankIC={m['RankIC']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        hist.append({"epoch": ep, "loss": tot / nb, "valid_ic": m["IC"], "valid_ric": m["RankIC"]})
        # 双轨存盘：ck 始终存最新 epoch（保证 epochs_done 单调、续训不重跑），
        # ck+".best" 存最优权重（评估用）；二者分开避免最优 epoch 卡住进度判断
        torch.save({"model": model.state_dict(), "epoch": ep, "best": best,
                    "fold": fold["id"], "model_name": model_name}, ck)
        if m["IC"] > best:
            best = m["IC"]
            torch.save({"model": model.state_dict(), "epoch": ep, "best": best,
                        "fold": fold["id"], "model_name": model_name}, ck + ".best")
    print(f"fold{fold['id']} {model_name} TRAIN DONE best_valid_IC={best:.4f}")
    return {"epochs_done": epochs_target, "best_valid_ic": best, "history": hist}


# ---------------- 评估（含 regime 分组） ----------------
def regime_lookup(dates):
    """日期 -> regime 标签；hmm_roll（walk-forward 口径）优先，缺失回退 hmm_v2。"""
    import duckdb
    out = {}
    con = duckdb.connect(DB, read_only=True)
    for src in ("hmm_roll", "hmm_v2"):   # 先写 v2 再写 roll，roll 覆盖 v2
        for d, s in con.execute("SELECT trade_date, regime FROM regime_labels WHERE source=?", [src]).fetchall():
            out[d] = s
    con.close()
    return out


def eval_fold(cfg, model_name, fold):
    import torch
    from torch.utils.data import DataLoader
    blob = load_fold(fold["id"])
    panels, G, meta = blob["panels"], blob["G"], blob["meta"]
    with_gate = model_name == "master"
    te = tm.DSWithDate(*panels["test"], G, seg_indices(blob, "test"), with_gate)
    mc = cfg["models"][model_name]
    model = tm.ALSTM(hidden=mc.get("hidden", 64), num_layers=mc.get("num_layers", 2)).to(tm.DEVICE) \
        if model_name == "alstm" else tm.MASTER(d_model=mc.get("d_model", 128)).to(tm.DEVICE)
    ck = ckpt_path(cfg, model_name, fold["id"])
    load_from = ck + ".best" if os.path.exists(ck + ".best") else ck   # 评估用最优权重
    st = torch.load(load_from, map_location=tm.DEVICE, weights_only=False)
    model.load_state_dict(st["model"])
    if load_from.endswith(".best"):
        print(f"  eval using best-epoch weights (epoch {st['epoch']})")
    ps, ys, ds = tm.predict(model, DataLoader(te, batch_size=2048, shuffle=False))
    dates = [meta["dates"][i] for i in ds]
    m = tm.daily_ic(ps, ys, dates)

    # 按 regime 分组日度 IC
    lut = regime_lookup(dates)
    df = pd.DataFrame({"d": dates, "p": ps, "y": ys})
    by_regime = {}
    for d, g in df.groupby("d"):
        if len(g) < 30 or d not in lut:
            continue
        ic = np.corrcoef(g["p"], g["y"])[0, 1]
        by_regime.setdefault(lut[d], []).append(ic)
    regime_ic = {r: {"IC": float(np.mean(v)), "days": len(v)} for r, v in by_regime.items()}
    for r, v in sorted(regime_ic.items()):
        print(f"  regime[{r}]: IC={v['IC']:.4f} ({v['days']}d)")

    # 日度 IC 序列落盘（DSR 审计输入）
    daily = []
    for d, g in df.groupby("d"):
        if len(g) < 30:
            continue
        daily.append({"date": str(d), "ic": float(np.corrcoef(g["p"], g["y"])[0, 1]),
                      "regime": lut.get(d, "")})
    os.makedirs(os.path.join(WS, "data", "wf_ic"), exist_ok=True)
    ic_csv = os.path.join(WS, "data", "wf_ic", f"{model_name}_fold{fold['id']}.csv")
    pd.DataFrame(daily).to_csv(ic_csv, index=False)
    print(f"fold{fold['id']} {model_name} TEST IC={m['IC']:.4f} ICIR={m['ICIR']:.3f} "
          f"RankIC={m['RankIC']:.4f} RankICIR={m['RankICIR']:.3f} ({m['days']}d)")
    return {"metrics": m, "regime_ic": regime_ic, "ic_csv": ic_csv,
            "best_valid_ic": st["best"], "final_epoch": st["epoch"]}


# ---------------- MLflow ----------------
def mlflow_log(cfg, model_name, fold, cache, train_res=None, eval_res=None):
    import mlflow
    os.environ["MLFLOW_ARTIFACT_URI"] = "file://" + MLRUNS   # 权重/IC 序列等大文件走文件仓库
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(WS, "mlflow.db"))
    exp = f"{cfg['tracking']['experiment_prefix']}-{model_name}"
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name=f"fold{fold['id']}_{fold['test'][0]}_{fold['test'][1]}"):
        mlflow.log_params({
            "fold_id": fold["id"], "train": f"{fold['train'][0]}..{fold['train'][1]}",
            "valid": f"{fold['valid'][0]}..{fold['valid'][1]}",
            "test": f"{fold['test'][0]}..{fold['test'][1]}",
            "config_sha": config_sha(cfg), "data_sha": data_sha(cache),
            "model": model_name, **{k: v for k, v in cfg["models"][model_name].items()},
        })
        if train_res:
            mlflow.log_metrics({"best_valid_ic": train_res["best_valid_ic"],
                                "epochs": train_res["epochs_done"]})
        if eval_res:
            m = eval_res["metrics"]
            mlflow.log_metrics({"test_ic": m["IC"], "test_icir": m["ICIR"],
                                "test_rankic": m["RankIC"], "test_rankicir": m["RankICIR"],
                                "test_days": m["days"]})
            for r, v in eval_res["regime_ic"].items():
                mlflow.log_metrics({f"test_ic_regime_{r}": v["IC"]})
            mlflow.log_artifact(eval_res["ic_csv"])
        with open(CFG_PATH, "rb") as f:
            mlflow.log_artifact(CFG_PATH)
        ck = ckpt_path(cfg, model_name, fold["id"])
        if os.path.exists(ck):
            mlflow.log_artifact(ck)
        print(f"  mlflow logged: {exp} fold{fold['id']}")


# ---------------- DSR 审计（1.4c 上线闸门） ----------------
def audit(cfg, model_name, trials=None, threshold=0.95):
    sys.path.insert(0, os.path.join(WS, "qs-skills", "skill-backtest-overfit", "scripts"))
    from deflated_sharpe import deflated_sharpe_ratio
    icdir = os.path.join(WS, "data", "wf_ic")
    if not os.path.isdir(icdir):
        print("no fold IC files; run eval first")
        return None
    files = sorted(f for f in os.listdir(icdir) if f.startswith(f"{model_name}_fold") and f.endswith(".csv"))
    if not files:
        print("no fold IC files; run eval first")
        return None
    series = pd.concat([pd.read_csv(os.path.join(icdir, f)) for f in files])
    n_trials = trials or (len(files) * len(cfg["models"]))
    res = deflated_sharpe_ratio(series["ic"].to_numpy(dtype=float), n_trials=n_trials)
    out = {"model": model_name, "folds": len(files), "n_trials": n_trials,
           "observed_ic_mean": float(series["ic"].mean()),
           "observed_ic_ir": float(series["ic"].mean() / (series["ic"].std() + 1e-12)),
           **res.to_dict(), "threshold": threshold, "passed": bool(res.deflated_sharpe_ratio >= threshold)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    os.makedirs(os.path.join(WS, "data", "wf_ic"), exist_ok=True)
    with open(os.path.join(icdir, f"{model_name}_dsr.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


# ---------------- CLI ----------------
def parse_folds(spec, folds):
    if spec == "all":
        return folds
    ids = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return [f for f in folds if f["id"] in ids]


def resolve_fold(a, folds):
    """--legacy-cache 且无 --fold 时 = smoke fold：P0 基准切分（fit 2015-2019），
    只用于验证管线正确性，不参与正式 walk-forward 统计。"""
    if a.legacy_cache and a.fold is None:
        return {"id": "smoke",
                "train": ("2015-01-01", "2019-12-31"),
                "valid": ("2020-01-01", "2020-12-31"),
                "test": ("2021-01-01", "2022-12-31")}
    if a.fold is None:
        raise SystemExit("需要 --fold N（或 smoke 用 --legacy-cache 不带 --fold）")
    return folds[a.fold]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["folds", "prep", "train", "eval", "run", "audit"])
    ap.add_argument("--model", default="alstm", choices=["alstm", "master"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖目标 epochs（smoke 用）")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--legacy-cache", action="store_true", help="smoke：复用 P0 缓存代替重新 fetch")
    a = ap.parse_args()
    cfg = load_config()
    folds = generate_folds(cfg)

    if a.cmd == "folds":
        for f in folds:
            print(f"fold{f['id']}: train {f['train'][0]}..{f['train'][1]} | "
                  f"valid {f['valid'][0]}..{f['valid'][1]} | test {f['test'][0]}..{f['test'][1]}")
        print(f"total {len(folds)} folds, config_sha={config_sha(cfg)}")
    elif a.cmd == "prep":
        fold = resolve_fold(a, folds)
        prep_fold(fold, cfg, legacy_cache=a.legacy_cache)
    elif a.cmd == "train":
        fold = resolve_fold(a, folds)
        prep_fold(fold, cfg, legacy_cache=a.legacy_cache)
        cache = os.path.join(PREP_WF, f"fold_{fold['id']}.pkl")
        tr = train_fold(cfg, a.model, fold, max_epochs=a.epochs)
        mlflow_log(cfg, a.model, fold, cache, train_res=tr)
    elif a.cmd == "eval":
        fold = resolve_fold(a, folds)
        cache = os.path.join(PREP_WF, f"fold_{fold['id']}.pkl")
        ev = eval_fold(cfg, a.model, fold)
        mlflow_log(cfg, a.model, fold, cache, eval_res=ev)
    elif a.cmd == "run":
        for fold in parse_folds(a.folds, folds):
            prep_fold(fold, cfg, legacy_cache=a.legacy_cache)
            cache = os.path.join(PREP_WF, f"fold_{fold['id']}.pkl")
            tr = train_fold(cfg, a.model, fold, max_epochs=a.epochs)
            ev = eval_fold(cfg, a.model, fold)
            mlflow_log(cfg, a.model, fold, cache, train_res=tr, eval_res=ev)
    elif a.cmd == "audit":
        audit(cfg, a.model, trials=a.trials)


if __name__ == "__main__":
    main()
