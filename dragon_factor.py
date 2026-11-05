# -*- coding: utf-8 -*-
"""dragon_factor.py — 龙虎榜 + 涨跌停微观结构因子准入评估（margin_factor.py 同口径第二单）。

因子族 dragon：
  LBNET   龙虎榜净买入额 / 当日成交额（上榜日，非上榜 0）
  LBCNT   20 日内龙虎榜上榜次数
  LIMUC   20 日内涨停次数
  LIMDC   20 日内跌停次数

  python dragon_factor.py fetch    # 按日缓存（断点续传，data/dragon/top_*.csv / limit_*.csv）
  python dragon_factor.py build    # 合成面板 -> data/factor_decay/dragon_feat.parquet
  python dragon_factor.py report   # 逐年 RankIC -> dragon_ic.csv，并入 family_ic_matrix.csv
"""
import os, sys, time
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(WS, "data", "dragon")
FD = os.path.join(WS, "data", "factor_decay")
os.makedirs(DD, exist_ok=True)

import sys as _sys, os as _os
for _root in {_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}:
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
from token_loader import tushare_token as _tushare_token
TOKEN = _tushare_token()
FACTORS = ["LBNET", "LBCNT", "LIMUC", "LIMDC"]


def trade_days():
    import tushare as ts
    pro = ts.pro_api(TOKEN)
    cal = pro.trade_cal(exchange="SSE", start_date="20150101", end_date="20261104")
    return sorted(cal[cal["is_open"] == 1]["cal_date"].tolist())


def fetch():
    import tushare as ts
    pro = ts.pro_api(TOKEN)
    days = trade_days()
    todo = [d for d in days
            if not os.path.exists(os.path.join(DD, f"top_{d}.csv"))]
    print(f"total={len(days)} todo={len(todo)}", flush=True)
    for i, d in enumerate(todo):
        for kind, fn in (("top", pro.top_list), ("limit", pro.limit_list_d)):
            p = os.path.join(DD, f"{kind}_{d}.csv")
            if os.path.exists(p):
                continue
            try:
                r = fn(trade_date=d)
                (r if r is not None else pd.DataFrame()).to_csv(p, index=False)
            except Exception as e:
                print(f"FAIL {kind} {d}: {e}", flush=True)
            time.sleep(0.21)
        if i % 100 == 99:
            print(f"{i+1}/{len(todo)}", flush=True)
    print("DRAGON FETCH DONE", flush=True)


def build():
    frames = []
    for fn in sorted(os.listdir(DD)):
        if not fn.startswith("top_"):
            continue
        d = fn[4:12]
        day = pd.to_datetime(d, format="%Y%m%d").strftime("%Y-%m-%d")
        try:
            top = pd.read_csv(os.path.join(DD, fn), dtype={"ts_code": str})
        except Exception:
            continue
        if "ts_code" not in top.columns:
            continue
        lim_p = os.path.join(DD, f"limit_{d}.csv")
        try:
            lim = pd.read_csv(lim_p, dtype={"ts_code": str}) if os.path.exists(lim_p) else pd.DataFrame()
        except Exception:
            lim = pd.DataFrame()
        rec = {}
        if len(top):
            t = top.dropna(subset=["ts_code"]).copy()
            t["net_ratio"] = t["net_amount"] / t["amount"].replace(0, np.nan)
            g = t.groupby("ts_code")
            rec["LBNET"] = g["net_ratio"].sum()
            rec["LBCNT_day"] = g.size().astype(float)
        up, dn = pd.Series(dtype=float), pd.Series(dtype=float)
        if len(lim):
            l = lim.dropna(subset=["ts_code"])
            up = l[l["limit"] == "U"].groupby("ts_code").size().astype(float)
            dn = l[l["limit"] == "D"].groupby("ts_code").size().astype(float)
        idx = sorted(set(rec.get("LBNET", pd.Series()).index)
                     | set(up.index) | set(dn.index))
        df = pd.DataFrame(index=pd.Index(idx, name="ts_code"))
        df["LBNET"] = rec.get("LBNET", pd.Series(dtype=float)).reindex(df.index).fillna(0.0)
        df["LBCNT_day"] = rec.get("LBCNT_day", pd.Series(dtype=float)).reindex(df.index).fillna(0.0)
        df["LIMUC_day"] = up.reindex(df.index).fillna(0.0)
        df["LIMDC_day"] = dn.reindex(df.index).fillna(0.0)
        frames.append(df.assign(datetime=day))
    if not frames:
        print("no data")
        return
    raw = pd.concat(frames)
    raw["instrument"] = ["sh" + c[:6] if c.endswith(".SH") else "sz" + c[:6]
                         for c in raw.index]
    raw = raw.reset_index(drop=True)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.set_index(["datetime", "instrument"]).sort_index()
    uni = {ln.split("\t")[0].strip() for ln in open(os.path.join(WS, "data", "csi300.txt"))}
    raw = raw[raw.index.get_level_values("instrument").isin(uni)]
    print(f"universe filter: {len(raw)} rows")
    # 事件稀疏因子必须零填充到全 universe × 交易日网格，再做滚动窗口
    lab = pd.read_parquet(os.path.join(FD, "labels.parquet"))
    dates = lab.index.get_level_values("datetime").unique().sort_values()
    insts = sorted(set(lab.index.get_level_values("instrument")) & uni)
    grid = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    raw = raw.reindex(grid).fillna(0.0)
    for c in ("LBCNT", "LIMUC", "LIMDC"):
        raw[c] = raw[c + "_day"].groupby("instrument").transform(
            lambda s: s.rolling(20, min_periods=5).sum())
    feat = raw[FACTORS].astype(np.float32)
    dt_level = pd.to_datetime(feat.index.get_level_values("datetime"))
    feat.index = pd.MultiIndex.from_arrays(
        [dt_level, feat.index.get_level_values("instrument")],
        names=["datetime", "instrument"])
    feat.to_parquet(os.path.join(FD, "dragon_feat.parquet"))
    print(f"dragon panel {feat.shape} -> dragon_feat.parquet", flush=True)


def rank_ic_by_year(year):
    feat = pd.read_parquet(os.path.join(FD, "dragon_feat.parquet"))
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
        # 事件因子：多数日期截面全零，只对"活跃日"（≥10 只非零）计 IC
        nz = sub[sub["f"] != 0]
        active = nz.groupby("d").size()
        active = active[active >= 10].index
        sub = sub[sub["d"].isin(active)]
        if sub["d"].nunique() < 20:
            continue
        r = sub.groupby("d")[["f", "y"]].rank()
        sd = sub["d"]
        g = r["f"].groupby(sd)
        fx = r["f"] - g.transform("mean")
        fy = r["y"] - r["y"].groupby(sd).transform("mean")
        cov = (fx * fy).groupby(sd).sum()
        ic = cov / np.sqrt((fx ** 2).groupby(sd).sum() * (fy ** 2).groupby(sd).sum())
        ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
        # dragon 为事件稀疏因子：沪深300 内每日上榜/涨跌停仅 10-30 只，门槛降到 10
        ic = ic[sd.groupby(sd).size().reindex(ic.index) >= 10]
        if len(ic) < 20:
            continue
        rows.append({"year": year, "feature": col, "family": "dragon",
                     "IC": float(ic.mean()), "ICIR": float(ic.mean() / (ic.std() + 1e-12))})
    return pd.DataFrame(rows)


def report():
    years = list(range(2015, 2027))
    per = [rank_ic_by_year(y) for y in years]
    per = [r for r in per if len(r)]
    ic = pd.concat(per) if per else pd.DataFrame(
        columns=["year", "feature", "family", "IC", "ICIR"])
    ic.to_csv(os.path.join(FD, "dragon_ic.csv"), index=False)
    if len(ic):
        print(ic.pivot(index="feature", columns="year", values="IC").round(4).to_string())
    fp = os.path.join(FD, "family_ic_matrix.csv")
    fam = pd.read_csv(fp, index_col=0)
    fam = fam.drop(index="dragon", errors="ignore")
    if len(ic):
        mrow = ic.groupby("year")["IC"].mean()
        fam.loc["dragon"] = [round(float(mrow.get(y, np.nan)), 6) for y in years]
    fam = fam.reindex(columns=[str(y) for y in years])
    fam.to_csv(fp)
    print("family_ic_matrix.csv 已并入 dragon 行", flush=True)


def main():
    cmd = sys.argv[1]
    {"fetch": fetch, "build": build, "report": report}[cmd]()


if __name__ == "__main__":
    main()
