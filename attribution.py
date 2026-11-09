# -*- coding: utf-8 -*-
"""attribution.py — 超额归因（回应终审审计：验证"收益来自预测"而非风格 beta）。

方法（时序回归版 Brinson 替代）：
  策略日超额 excess_t 对 CSI300 池内四个风格因子日收益做 OLS + Newey-West：
    SMB   小市值减大市值（ln total_mv 中位分组）
    MOM   20-1 动量十分位多空
    VOL   20 日收益波动（低减高）
    LIQ   20 日平均成交额（高减低）
  特征 t-1 可知、行业中性化（按 tushare 行业去均值后排序，防行业项混入风格项）；
  截距 = 年化 alpha；R² = 被风格解释的超额份额。

  python attribution.py           # 全流程：拉市值 → 建因子 → 回归 → 报告
  python attribution.py dsr       # 只对已有日度序列重算 DSR（n_trials=账本 9）
"""
import os, sys, json, time
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
BT_DAILY = os.path.join(WS, "data", "backtest", "daily")
CSI300 = os.path.join(WS, "data", "csi300.txt")
OUT = os.path.join(WS, "data", "backtest", "attribution.json")
FOLDS = [8, 9, 10, 11]
SEAL = "2026-07-01"          # holdout 封存线：归因是运维诊断，但也只用到封存线，保持纪律一致
START, END = "2022-01-01", SEAL


def load_excess():
    """四 fold 日度 excess 拼接（fold11 截到封存线）。"""
    import glob as gb
    parts = []
    for f in sorted(gb.glob(os.path.join(BT_DAILY, "alstm_fold*_oc0.0005_cc0.0015.parquet"))):
        d = pd.read_parquet(f)
        d["date"] = pd.to_datetime(d.iloc[:, 0]) if not np.issubdtype(d.iloc[:, 0].dtype, np.datetime64) else pd.to_datetime(d.iloc[:, 0])
        parts.append(d[["date", "excess"]])
    df = pd.concat(parts).drop_duplicates("date").set_index("date").sort_index()
    df = df[df.index < SEAL]
    return df["excess"]


def load_universe():
    raw = [ln.split("\t")[0].strip() for ln in open(CSI300)]
    # csi300.txt 存的是 sh600000 格式；归一化为裸 6 位码
    out = []
    for c in raw:
        if not c:
            continue
        if c[:2].lower() in ("sh", "sz"):
            c = c[2:]
        if len(c) == 6 and c.isdigit():
            out.append(c)
    return out


def load_qlib_frames(codes):
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    insts = [("SH" + c if c.startswith("6") else "SZ" + c) for c in codes]
    px = D.features(insts, ["$close", "$volume", "$amount"],
                    start_time=START, end_time=END)
    # index (instrument, datetime) -> (datetime, instrument)
    close = px["$close"].unstack("instrument")
    amt = px["$amount"].unstack("instrument")
    vol = px["$volume"].unstack("instrument")
    return close, amt, vol


def _pro():
    from token_loader import tushare_token
    import tushare as ts
    return ts.pro_api(tushare_token())


def load_mktcap(codes):
    """tushare daily_basic total_mv，(date × code) 矩阵；6 线程并发拉取。"""
    from concurrent.futures import ThreadPoolExecutor
    def one(c):
        for _ in range(3):
            try:
                d = _pro().daily_basic(ts_code=f"{c}.SH" if c.startswith("6") else f"{c}.SZ",
                                       start_date="20220101", end_date=END.replace("-", ""),
                                       fields="trade_date,total_mv")
                if d is not None and not d.empty:
                    return c, pd.Series(d["total_mv"].to_numpy(dtype=float),
                                        index=pd.to_datetime(d["trade_date"]))
                return c, None
            except Exception:
                time.sleep(1.0)
        return c, None
    out = {}
    with ThreadPoolExecutor(6) as ex:
        for i, (c, s) in enumerate(ex.map(one, codes)):
            if s is not None:
                out[c] = s
            if i % 40 == 0:
                print(f"mv {i}/{len(codes)}", flush=True)
    mv = pd.DataFrame(out)
    mv.index = pd.to_datetime(mv.index)
    return mv.sort_index()


def load_industry(codes):
    """单调用全量 stock_basic，再按代码过滤。"""
    d = _pro().stock_basic(exchange="", list_status="L", fields="ts_code,industry")
    m = {str(r["ts_code"]).split(".")[0]: str(r["industry"]) for _, r in d.iterrows()}
    return {c: m.get(c, "未知") for c in codes}


def xs_spread(z, ret, q=0.3):
    """按 z 头尾 30% 分组，做多高组做空低组（SIZE 反向：小减大在调用侧处理）。"""
    hi = z.rank(axis=1, pct=True) >= 1 - q
    lo = z.rank(axis=1, pct=True) <= q
    r_hi = ret.where(hi).mean(axis=1)
    r_lo = ret.where(lo).mean(axis=1)
    return r_hi - r_lo


def nw_ols(y, X, lags=10):
    """OLS + Newey-West t（协方差 HAC）。"""
    import statsmodels.api as sm
    X = sm.add_constant(X)
    m = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m


def main():
    codes = load_universe()
    print(f"universe {len(codes)}", flush=True)
    MV_PQ = os.path.join(WS, "data", "backtest", "mv_panel.parquet")
    F_PQ = os.path.join(WS, "data", "backtest", "factor_returns.parquet")
    if os.path.exists(F_PQ):
        F = pd.read_parquet(F_PQ)
    else:
        close, amt, vol = load_qlib_frames(codes)
        print(f"qlib px {close.shape}", flush=True)
        mv = pd.read_parquet(MV_PQ) if os.path.exists(MV_PQ) else load_mktcap(codes)
        mv.to_parquet(MV_PQ)
        # mv 列是裸 6 位码，qlib 列是 SH600000 格式 —— 对齐后再 reindex
        mv = mv.rename(columns={c: ("SH" + c if c.startswith("6") else "SZ" + c) for c in mv.columns})
        industry = load_industry(codes)
        # 因子
        ret = close.pct_change()
        fac = {}
        for name, z_raw, reverse in (
            ("SMB", np.log(mv.reindex(close.index).reindex(columns=close.columns)), True),
            ("MOM", close.shift(1) / close.shift(21) - 1, False),
            ("VOL", ret.rolling(20).std(), True),       # 低波动减高波动 → 反向
            ("LIQ", np.log(amt.reindex(close.index).reindex(columns=close.columns).rolling(20).mean()), False),
        ):
            z = z_raw.copy()
            inds = pd.Series(industry).reindex(close.columns)
            for _, g_cols in inds.groupby(inds).groups.items():
                z[g_cols] = z[g_cols].sub(z[g_cols].mean(axis=1), axis=0)
            z = z.shift(1)
            s = xs_spread(z, ret)
            fac[name] = -s if reverse else s
        F = pd.DataFrame(fac)
        F.to_parquet(F_PQ)

    y = load_excess()
    df = pd.concat([y.rename("y"), F], axis=1).dropna()
    m = nw_ols(df["y"], df[["SMB", "MOM", "VOL", "LIQ"]])
    ann_a = m.params["const"] * 244
    t_a = m.tvalues["const"]
    res = {
        "n_days": int(m.nobs),
        "window": f"{df.index[0].date()} ~ {df.index[-1].date()}",
        "excess_ann_raw": float(y.mean() * 244),
        "alpha_ann": float(ann_a),
        "alpha_t_nw": float(t_a),
        "r2": float(m.rsquared),
        "betas": {k: {"beta": float(m.params[k]), "t_nw": float(m.tvalues[k])}
                  for k in ("SMB", "MOM", "VOL", "LIQ")},
        "annualized_factor_mean_bp": {k: float(df[k].mean() * 244 * 1e4) for k in ("SMB", "MOM", "VOL", "LIQ")},
    }
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def dsr_only():
    sys.path.insert(0, os.path.join(WS, "qs-skills", "skill-backtest-overfit", "scripts"))
    from deflated_sharpe import deflated_sharpe_ratio
    import candidate_ledger as led
    y = load_excess()
    n = led.status()["ledger_entries"]          # 账本 9 条（判决收据数），不是 fold×模型数
    r = deflated_sharpe_ratio(y.to_numpy(dtype=float), n_trials=n)
    out = {"series": "daily strategy excess return (wf folds 8-11, costed, T+1 open fill)",
           "n_obs": int(len(y)), "n_trials": n,
           "observed_sr_per_period": float(y.mean() / y.std(ddof=1)),
           "observed_sr_ann": float(y.mean() / y.std(ddof=1) * np.sqrt(244)),
           **r.to_dict(),
           "note": "对比旧版：旧 DSR 用 IC 序列+n_trials=folds×models；本版用策略收益序列+n_trials=账本候选数"}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    json.dump(out, open(os.path.join(WS, "data", "backtest", "dsr_recomputed.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dsr":
        dsr_only()
    else:
        main()
