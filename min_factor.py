# -*- coding: utf-8 -*-
"""min_factor.py — 日内微观结构因子准入评估（factor_decay.py 同口径，第五单/分钟级）。

 stk_mins 5min 线 → 日度聚合特征。T 日盘后可知，对 T+1→T+2 标签无泄漏。
 与 Alpha158 日频价量天然正交（日内分布信息在日频聚合中不可恢复）。

因子族 min（5 个候选）：
  TAILR  尾盘成交占比   last30min_vol / day_vol       —— 机构尾盘行为
  HEADR  开盘成交占比   first30min_vol / day_vol      —— 隔夜信息消化/抢筹
  IVOL   日内已实现波动 std(5min 收益)×√48            —— 日内与隔夜波动分解
  PMD    午后-上午收益 afternoon_ret - morning_ret     —— 日内动量形态
  TCONT  趋势连续性    |day_ret| / (sum|5min ret|+ε)   —— 路径效率（1=直线）

口径限制（诚实清单）：分钟数据 2023 年起拉取（体量约束），IC 评估仅 2023-2026 四年；
   样本外年份与 walk-forward 终审口径对齐（fold8-11 测试窗正是 2023-2026）。

  python min_factor.py fetch [--n 40]   # 按股断点续传，每次调用最多新拉 n 只（默认 40）
  python min_factor.py build            # 日度聚合 -> data/factor_decay/min_feat.parquet
  python min_factor.py report           # 逐年 RankIC -> data/factor_decay/min_ic.csv，
                                        # 并入 family_ic_matrix.csv（族名 min）
"""
import os, sys, time
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(WS, "data", "min")
FD = os.path.join(WS, "data", "factor_decay")
os.makedirs(MD, exist_ok=True)

import sys as _sys, os as _os
for _root in {_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}:
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
from token_loader import tushare_token as _tushare_token
TOKEN = _tushare_token()
FACTORS = ["TAILR", "HEADR", "IVOL", "PMD", "TCONT"]
START, END = "2023-01-01", "2026-12-31"
HIST_START, HIST_END = "2015-01-01", "2022-12-31"
CHUNK_DAYS = 65          # 2023+ 块（沿用既有粒度）
HIST_CHUNK = 130         # 历史回填块（~4200 行/块，低于接口行数上限）


def live_codes():
    out = []
    for ln in open(os.path.join(WS, "data", "csi300.txt")):
        p = ln.split("\t")
        if len(p) >= 3 and p[2].strip() >= "2026-01-01":
            out.append(p[0].strip())
    return out


def chunks(s, e, days):
    ds = pd.date_range(s, e, freq="D")
    return [(str(ds[i].date()), str(ds[min(i + days - 1, len(ds) - 1)].date()))
            for i in range(0, len(ds), days)]


def all_chunks():
    return chunks(HIST_START, HIST_END, HIST_CHUNK) + chunks(START, END, CHUNK_DAYS)


def fetch(n=40):
    import tushare as ts
    pro = ts.pro_api(TOKEN)
    cks = all_chunks()
    codes = live_codes()
    done = [c for c in codes if all(
        os.path.exists(os.path.join(MD, c, f"{s}_{e}.csv")) for s, e in cks)]
    todo = [c for c in codes if c not in done][:n]
    print(f"codes total={len(codes)} fully_done={len(done)} this_run={len(todo)} "
          f"chunks/stock={len(cks)}", flush=True)
    for c in todo:
        ts_code = c[2:] + (".SH" if c.startswith("sh") else ".SZ")
        od = os.path.join(MD, c)
        os.makedirs(od, exist_ok=True)
        for s, e in cks:
            fp = os.path.join(od, f"{s}_{e}.csv")
            if os.path.exists(fp):
                continue
            try:
                d = pro.stk_mins(ts_code=ts_code, freq="5min",
                                 start_date=f"{s} 09:00:00", end_date=f"{e} 15:30:00")
                if d is not None and len(d):
                    d.to_csv(fp, index=False)
                else:
                    open(fp, "w").close()      # 空窗口也落盘，避免重试
            except Exception as ex:
                print(f"FAIL {c} {s}: {ex}", flush=True)
            time.sleep(0.13)
    done2 = len([c for c in codes if all(
        os.path.exists(os.path.join(MD, c, f"{s}_{e}.csv")) for s, e in cks)])
    print(f"FETCH BATCH DONE fully_done={done2}/{len(codes)}", flush=True)


def build():
    frames = []
    for c in sorted(os.listdir(MD)):
        od = os.path.join(MD, c)
        if not os.path.isdir(od):
            continue
        parts = [pd.read_csv(os.path.join(od, f), dtype={"trade_time": str})
                 for f in sorted(os.listdir(od)) if f.endswith(".csv") and os.path.getsize(os.path.join(od, f))]
        if not parts:
            continue
        m = pd.concat(parts, ignore_index=True).drop_duplicates("trade_time")
        m["dt"] = pd.to_datetime(m["trade_time"])
        m = m.sort_values("dt")
        m["date"] = m["dt"].dt.strftime("%Y-%m-%d")
        m["hm"] = m["dt"].dt.strftime("%H:%M")
        g = m.groupby("date")
        day_vol = g["vol"].sum()
        last30 = m[m["hm"] >= "14:30"].groupby("date")["vol"].sum()
        first30 = m[m["hm"] <= "10:00"].groupby("date")["vol"].sum()
        tailr = (last30 / day_vol).reindex(day_vol.index)
        headr = (first30 / day_vol).reindex(day_vol.index)
        r5 = m.groupby("date")["close"].diff() / m.groupby("date")["close"].shift()
        ivol = m.assign(r=r5).groupby("date")["r"].std()
        o = g["open"].first()
        cl = g["close"].last()
        day_ret = cl / o - 1
        am = m[m["hm"] <= "11:30"].groupby("date")["close"].last() / o - 1
        pm = (cl / m[m["hm"] >= "13:00"].groupby("date")["open"].first() - 1)
        pmd = (pm - am).reindex(day_vol.index)
        path = m.assign(absr=r5.abs()).groupby("date")["absr"].sum()
        tcont = (day_ret.abs() / (path + 1e-12)).reindex(day_vol.index)
        f = pd.DataFrame({"TAILR": tailr, "HEADR": headr, "IVOL": ivol,
                          "PMD": pmd, "TCONT": tcont})
        f["instrument"] = c
        frames.append(f.reset_index().set_index(["date", "instrument"]))
    feat = pd.concat(frames).sort_index()
    feat.index.set_names(["datetime", "instrument"], inplace=True)
    feat = feat.reset_index()
    feat["datetime"] = pd.to_datetime(feat["datetime"])
    feat = feat.set_index(["datetime", "instrument"])
    feat.to_parquet(os.path.join(FD, "min_feat.parquet"))
    print(f"min panel {feat.shape} -> min_feat.parquet", flush=True)


def rank_ic_by_year(year):
    feat = pd.read_parquet(os.path.join(FD, "min_feat.parquet"))
    lab = pd.read_parquet(os.path.join(FD, "labels.parquet"))
    d0, d1 = f"{year}-01-01", f"{year}-12-31"
    feat = feat[(feat.index.get_level_values("datetime") >= d0)
                & (feat.index.get_level_values("datetime") <= d1)]
    df = feat.join(lab, how="inner")
    dd = df.index.get_level_values("datetime")
    rows = []
    for col in FACTORS:
        sub = pd.DataFrame({"d": dd, "f": df[col], "y": df["y"]}).dropna()
        if len(sub) < 1000:
            continue
        r = sub.groupby("d")[["f", "y"]].rank()
        sd = sub["d"]
        g = r["f"].groupby(sd)
        fx = r["f"] - g.transform("mean")
        fy = r["y"] - r["y"].groupby(sd).transform("mean")
        cov = (fx * fy).groupby(sd).sum()
        ic = cov / np.sqrt((fx ** 2).groupby(sd).sum() * (fy ** 2).groupby(sd).sum())
        ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
        ic = ic[sd.groupby(sd).size().reindex(ic.index) >= 30]
        if len(ic) < 30:
            continue
        rows.append({"year": year, "feature": col, "family": "min",
                     "IC": float(ic.mean()), "ICIR": float(ic.mean() / (ic.std() + 1e-12))})
    return pd.DataFrame(rows)


def report():
    years = list(range(2015, 2027))
    ic = pd.concat([rank_ic_by_year(y) for y in years if len(rank_ic_by_year(y))])
    ic.to_csv(os.path.join(FD, "min_ic.csv"), index=False)
    print(ic.pivot(index="feature", columns="year", values="IC").round(4).to_string())
    fp = os.path.join(FD, "family_ic_matrix.csv")
    fam = pd.read_csv(fp, index_col=0)
    fam = fam.drop(index="min", errors="ignore")
    mrow = ic.groupby("year")["IC"].mean()
    all_years = [int(c) for c in fam.columns]
    fam.loc["min"] = [round(float(mrow.get(y, np.nan)), 6) for y in all_years]
    fam.to_csv(fp)
    print("\nfamily_ic_matrix.csv 已并入 min 行", flush=True)


def main():
    cmd = sys.argv[1]
    if cmd == "fetch":
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40
        fetch(n)
    elif cmd == "build":
        build()
    elif cmd == "report":
        report()


if __name__ == "__main__":
    main()
