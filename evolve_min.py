# -*- coding: utf-8 -*-
"""evolve_min.py — L2 分钟因子生成式进化引擎（多批次）。

设计：
  候选因子 = 纯函数(单日 5min bar 数组)，无未来函数、无跨日状态（跨日变换在面板层做）。
  计算层：每只股票一个进程任务，输出 data/evolve/panel/<inst>_b<N>.parquet（断点续跑）。
  面板层：concat + 批次专属衍生 → data/evolve/panel_b<N>.parquet
  IC 层：与 labels.parquet（T+1→T+2 收益）join，逐年 RankIC 2015–2026。

预注册判定标准（写于实验前，批次间不变，不许后改）：
  单因子入门：12 年 RankIC 均值 |IC| ≥ 0.015 且 ≥ 9/12 年同号
  进模型层加试：与基线 158 维模型 IC 序列 |corr| < 0.8（正交性）

批次档案：
  batch 1（2026-11-05）：IVOL 邻域 + 量价/路径族 12 候选 → range_pos 过单因子闸，
      模型层 p=0.249 未过；vol_skew/pv_corr 稳定弱信号入池。
  batch 2（2026-11-05）：池反馈驱动的强度增强变异——
      母本 range_pos → 时段分解/趋势加权/跳空交互/放量交互；
      母本 vol_skew → 峰值复合/尾盘偏度/CV/峰度（稳健与尖锐化两路）；
      母本 pv_corr → 时段分解/差分/金额版/滞后版（多尺度四路）。

用法：
  python3 evolve_min.py run --batch 2 [--limit N] [--workers 8]
  python3 evolve_min.py assemble --batch 2
  python3 evolve_min.py ic --batch 2
"""
import os, sys, glob, argparse
from functools import partial
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
MIN_DIR = os.path.join(WS, "data", "min")
OUT_DIR = os.path.join(WS, "data", "evolve")
PANEL_DIR = os.path.join(OUT_DIR, "panel")
LABELS = os.path.join(WS, "data", "factor_decay", "labels.parquet")
EPS = 1e-12


# ---------------- 共用工具 ----------------

def _day_arrays(g):
    """g: 单股单日 5min bars（升序）→ 预计算数组字典。"""
    o, h, l, c = g["open"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy()
    amt, vol = g["amount"].to_numpy(), g["vol"].to_numpy()
    t = g["dt"].dt.hour * 100 + g["dt"].dt.minute
    ret = np.diff(np.log(np.maximum(c, EPS)))
    rng = np.maximum(h - l, EPS)
    return dict(o=o, h=h, l=l, c=c, amt=amt, vol=vol, ret=ret, rng=rng, t=t.to_numpy(),
                n=len(c), amt_sum=amt.sum() + EPS, vol_sum=vol.sum() + EPS)


def _skew(x):
    if len(x) < 5: return np.nan
    sd = x.std()
    return float(((x - x.mean()) ** 3).mean() / (sd ** 3 + EPS)) if sd > EPS else np.nan


def _kurt(x):
    if len(x) < 5: return np.nan
    sd = x.std()
    return float(((x - x.mean()) ** 4).mean() / (sd ** 4 + EPS) - 3) if sd > EPS else np.nan


def _pvc(ret, dv):
    if len(ret) < 10 or dv.std() < EPS or ret.std() < EPS: return np.nan
    return float(np.corrcoef(ret, dv)[0, 1])


def _dvol(vol):
    return np.diff(np.log(np.maximum(vol, 1.0)))


# ---------------- batch 1：IVOL 邻域 + 量价/路径族 ----------------

def ivol_am(a):
    if a["n"] < 20: return np.nan
    return float(np.sqrt(np.sum(a["ret"][: 12] ** 2)))

def ivol_volov(a):
    ar = np.abs(a["ret"])
    m = ar.mean()
    return float(ar.std() / (m + EPS)) if a["n"] >= 20 and m > 0 else np.nan

def ivol_tailshare(a):
    if a["n"] < 20: return np.nan
    r2 = a["ret"] ** 2
    tot = r2.sum()
    return float(r2[-6:].sum() / (tot + EPS)) if tot > 0 else np.nan

def gap_vol_ratio(a, prev_close):
    if a["n"] < 10 or prev_close is None or prev_close <= 0: return np.nan
    gap = abs(a["o"][0] / prev_close - 1)
    iv = np.sqrt(np.sum(a["ret"] ** 2))
    return float(gap / (iv + 0.001))

def vol_skew(a):
    if a["n"] < 20 or a["amt_sum"] <= 0: return np.nan
    return _skew(a["amt"] / a["amt_sum"])

def vol_peak_pos(a):
    if a["n"] < 20: return np.nan
    return float(np.argmax(a["amt"]) / a["n"])

def pv_corr(a):
    return _pvc(a["ret"], _dvol(a["vol"]))

def path_eff(a):
    L = np.abs(a["ret"]).sum()
    if a["n"] < 10 or L < EPS: return np.nan
    return float(abs(np.log(a["c"][-1] / (a["o"][0] + EPS))) / L)

def path_rev(a):
    r = a["ret"][np.abs(a["ret"]) > EPS]
    if len(r) < 10: return np.nan
    return float(np.mean(np.sign(r[1:]) * np.sign(r[:-1]) < 0))

def range_pos(a):
    span = a["h"].max() - a["l"].min()
    if a["n"] < 10 or span < EPS: return np.nan
    return float((a["c"][-1] - a["l"].min()) / span)

def wick_up(a):
    if a["n"] < 10: return np.nan
    w = (a["h"] - np.maximum(a["o"], a["c"])) / a["rng"]
    return float(np.clip(w, 0, 1).mean())

BATCH1_FACTORS = {
    "ivol_am": ivol_am, "ivol_volov": ivol_volov, "ivol_tailshare": ivol_tailshare,
    "gap_vol_ratio": gap_vol_ratio, "vol_skew": vol_skew, "vol_peak_pos": vol_peak_pos,
    "pv_corr": pv_corr, "path_eff": path_eff, "path_rev": path_rev,
    "range_pos": range_pos, "wick_up": wick_up,
}
BATCH1_TS = ["ivol_tailshare", "vol_peak_pos", "pv_corr", "range_pos"]


# ---------------- batch 2：池反馈强度增强变异 ----------------

def gap_ret(a, prev_close):            # 母本交互原料：带符号隔夜跳空
    if a["n"] < 5 or prev_close is None or prev_close <= 0: return np.nan
    return float(a["o"][0] / prev_close - 1)

def vol_sum(a):                        # 母本交互原料：当日总成交量
    return float(a["vol_sum"])

def rpos_am(a):                        # range_pos 时段分解：上午版
    m = a["t"] <= 1130
    if m.sum() < 6: return np.nan
    lo, hi = a["l"][m].min(), a["h"][m].max()
    span = hi - lo
    return float((a["c"][m][-1] - lo) / span) if span >= EPS else np.nan

def rpos_patheff(a):                   # range_pos × 趋势纯度（趋势日收盘位置更可信）
    rp, pe = range_pos(a), path_eff(a)
    return rp * pe if np.isfinite(rp) and np.isfinite(pe) else np.nan

def skew_x_peak(a):                    # vol_skew × (1-量峰值位置)：偏度且峰值靠后
    s, p = vol_skew(a), vol_peak_pos(a)
    return s * (1 - p) if np.isfinite(s) and np.isfinite(p) else np.nan

def tail_skew(a):                      # 尾盘 30 分钟成交额偏度
    m = a["t"] >= 1430
    return _skew(a["amt"][m]) if m.sum() >= 4 else np.nan

def amt_cv(a):                         # 成交额分布变异系数（偏度稳健版）
    if a["n"] < 20: return np.nan
    sh = a["amt"] / a["amt_sum"]
    return float(sh.std() / (sh.mean() + EPS))

def amt_kurt(a):                       # 成交额分布峰度（极端 bar 集中度）
    if a["n"] < 20: return np.nan
    return _kurt(a["amt"] / a["amt_sum"])

def pv_corr_am(a):                     # 量价相关：上午时段
    m = a["t"][1:] <= 1130
    return _pvc(a["ret"][m], _dvol(a["vol"])[m]) if m.sum() >= 10 else np.nan

def pv_corr_diff(a):                   # 量价相关：上午 - 下午（背离差分）
    mam, mpm = a["t"][1:] <= 1130, a["t"][1:] >= 1300
    if mam.sum() < 10 or mpm.sum() < 10: return np.nan
    am, pm = _pvc(a["ret"][mam], _dvol(a["vol"])[mam]), _pvc(a["ret"][mpm], _dvol(a["vol"])[mpm])
    return am - pm if np.isfinite(am) and np.isfinite(pm) else np.nan

def pv_corr_amt(a):                    # 量价相关：金额版
    return _pvc(a["ret"], np.diff(np.log(np.maximum(a["amt"], 1.0))))

def pv_lag1(a):                        # 量价滞后相关：ret_t 对 t-1 量变化（资金流方向）
    dv = _dvol(a["vol"])
    return _pvc(a["ret"][1:], dv[:-1]) if len(dv) >= 12 else np.nan

BATCH2_FACTORS = {
    "rpos_raw": range_pos, "gap_ret": gap_ret, "vol_sum": vol_sum,
    "rpos_am": rpos_am, "rpos_patheff": rpos_patheff, "skew_x_peak": skew_x_peak,
    "tail_skew": tail_skew, "amt_cv": amt_cv, "amt_kurt": amt_kurt,
    "pv_corr_am": pv_corr_am, "pv_corr_diff": pv_corr_diff,
    "pv_corr_amt": pv_corr_amt, "pv_lag1": pv_lag1,
}
BATCHES = {1: (BATCH1_FACTORS, BATCH1_TS), 2: (BATCH2_FACTORS, [])}


# ---------------- 计算层：逐股（断点续跑） ----------------

def process_stock(inst, batch=1):
    factors, _ = BATCHES[batch]
    out = os.path.join(PANEL_DIR, f"{inst}_b{batch}.parquet")
    if os.path.exists(out):
        return inst, "cached"
    files = sorted(glob.glob(os.path.join(MIN_DIR, inst, "*.csv")))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            continue  # 空块：中断下载残留，跳过
    if not frames:
        return inst, "all_empty"
    df = pd.concat(frames, ignore_index=True)
    df["dt"] = pd.to_datetime(df["trade_time"])
    df = df.sort_values("dt")
    df["date"] = df["dt"].dt.date.astype(str)
    rows = []
    prev_close = None
    for d, g in df.groupby("date"):
        a = _day_arrays(g)
        rec = {"date": d}
        for name, fn in factors.items():
            rec[name] = fn(a, prev_close) if name in ("gap_vol_ratio", "gap_ret") else fn(a)
        rows.append(rec)
        last_c = a["c"][-1]
        if np.isfinite(last_c) and last_c > 0:
            prev_close = float(last_c)
    pd.DataFrame(rows).to_parquet(out)
    return inst, f"ok({len(rows)}d)"


def cmd_run(a):
    os.makedirs(PANEL_DIR, exist_ok=True)
    suffix = f"_b{a.batch}.parquet"
    insts = sorted(os.listdir(MIN_DIR))
    if a.only:
        insts = [i for i in insts if i in a.only.split(",")]
    todo = [i for i in insts if not os.path.exists(os.path.join(PANEL_DIR, i + suffix))]
    if a.limit:
        todo = todo[: a.limit]
    print(f"batch{a.batch} todo={len(todo)} cached={len(insts) - len(todo)}")
    from multiprocessing import Pool
    done = 0
    with Pool(a.workers) as p:
        for inst, st in p.imap_unordered(partial(process_stock, batch=a.batch), todo):
            done += 1
            if done % 40 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} {inst} {st}", flush=True)
    print("run done")


# ---------------- 面板层 ----------------

def cmd_assemble(a):
    suffix = f"_b{a.batch}.parquet"
    files = sorted(glob.glob(os.path.join(PANEL_DIR, f"*{suffix}")))
    df = pd.concat([pd.read_parquet(f).assign(instrument=os.path.basename(f)[: -len(suffix)])
                    for f in files], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    if a.batch == 1:
        for name in BATCH1_TS:
            m = df.groupby("instrument")[name].transform(lambda s: s.rolling(5, min_periods=3).mean())
            df[name + "_d5"] = df[name] - m
    if a.batch == 2:
        # TS 交互衍生：放量交互 / 跳空方向交互（无未来函数）
        g = df.groupby("instrument")["vol_sum"]
        vol20 = g.transform(lambda s: s.rolling(20, min_periods=10).mean())
        df["rpos_vol"] = df["rpos_raw"] * df["vol_sum"] / (vol20 + EPS)
        df["rpos_gap"] = df["rpos_raw"] * np.sign(df["gap_ret"])
    out = os.path.join(OUT_DIR, f"panel_b{a.batch}.parquet")
    df.to_parquet(out)
    print(f"panel_b{a.batch} {df.shape} -> {out}  span {df['date'].min()}..{df['date'].max()}")


# ---------------- IC 层（逐年 RankIC，同 P8 口径） ----------------

def cmd_ic(a):
    panel = pd.read_parquet(os.path.join(OUT_DIR, f"panel_b{a.batch}.parquet"))
    lab = pd.read_parquet(LABELS).reset_index()[["datetime", "instrument", "y"]]
    df = panel.merge(lab, on=["datetime", "instrument"], how="inner")
    factors = [c for c in panel.columns if c not in ("date", "datetime", "instrument")]
    recs = []
    for f in factors:
        d = df[["date", f, "y"]].dropna()
        for y, g in d.groupby(d["date"].str[:4]):
            g = g.copy()
            g["r"] = g[f].rank()
            recs.append({"factor": f, "year": int(y),
                         "ic": g["r"].corr(g["y"], method="spearman"), "n": len(g)})
    ic = pd.DataFrame(recs)
    ic.to_csv(os.path.join(OUT_DIR, f"ic_batch{a.batch}.csv"), index=False)
    s = ic.groupby("factor").agg(
        mean_ic=("ic", "mean"), pct_neg=("ic", lambda x: (x < 0).mean()),
        n_years=("year", "nunique")).reset_index()
    s["abs_ic"] = s["mean_ic"].abs()
    s["pass_gate"] = (s["abs_ic"] >= 0.015) & (s["pct_neg"].ge(0.75) | s["pct_neg"].le(0.25))
    s = s.sort_values("abs_ic", ascending=False)
    s.to_csv(os.path.join(OUT_DIR, f"ic_batch{a.batch}_summary.csv"), index=False)
    print(s.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "assemble", "ic"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    {"run": cmd_run, "assemble": cmd_assemble, "ic": cmd_ic}[a.cmd](a)


if __name__ == "__main__":
    main()
