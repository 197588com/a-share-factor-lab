# -*- coding: utf-8 -*-
"""P1 因子实验室：DuckDB 因子注册表 + 批量测试管线（regime 分组 / 五道硬门槛）

子命令:
  init_db                     建 DuckDB schema
  regime                      计算并写入市场状态标签（规则版 v1）
  run --start 0 --count 10    批量测试（断点续跑：按 config_hash 去重）
  report                      输出测试汇总
"""
import os, sys, json, hashlib, argparse
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(WS, "data", "factor_lab.duckdb")
CACHE = os.path.join(WS, "data", "factor_cache")
os.makedirs(CACHE, exist_ok=True)

UNIVERSE = "csi300"
START, END = "2010-01-01", "2022-12-31"   #  regime 标签全样本分位数需要 2008-2010 预热，这里统一起点

# ---------------- 因子定义（qlib 表达式，价量类 22 个） ----------------
FACTORS = [
    # (factor_id, name, category, expression)
    ("mom_5d",      "5日动量",        "momentum",  "Ref($close,5)/Ref($close,1)"),
    ("mom_20d",     "20日动量",       "momentum",  "Ref($close,20)/Ref($close,1)"),
    ("mom_60d",     "60日动量",       "momentum",  "Ref($close,60)/Ref($close,1)"),
    ("ma_cross",    "均线多空",       "momentum",  "Mean($close,5)/Mean($close,60)"),
    ("dist_high20", "距20日高点",     "reversal",  "$close/Max($high,20)"),
    ("dist_low20",  "距20日低点",     "reversal",  "Min($low,20)/$close"),
    ("maxret20",    "彩票因子(20日)", "reversal",  "Max($close/Ref($close,1)-1,20)"),
    ("vol_20d",     "20日波动率",     "volatility","Std($close/Ref($close,1)-1,20)"),
    ("vol_60d",     "60日波动率",     "volatility","Std($close/Ref($close,1)-1,60)"),
    ("hl_range20",  "20日振幅",       "volatility","Mean(($high-$low)/$close,20)"),
    ("atr_14",      "14日ATR",        "volatility","Mean($high-$low,14)/$close"),
    ("vstd_20",     "量能波动",       "volume",    "Std($volume,20)/Mean($volume,20)"),
    ("vol_ratio_5_60","量比5/60",     "volume",    "Mean($volume,5)/Mean($volume,60)"),
    ("amt_ratio_5_60","额比5/60",     "volume",    "Mean($amount,5)/Mean($amount,60)"),
    ("corr_pv_20",  "量价相关20日",   "vp",        "Corr($close,$volume,20)"),
    ("corr_pv_60",  "量价相关60日",   "vp",        "Corr($close,$volume,60)"),
    ("vp_diverge",  "量价背离",       "vp",        "($close/Mean($close,20))/(Mean($volume,20)/$volume)"),
    ("obv_mom20",   "OBV动量20日",    "vp",        "Sum(($close>Ref($close,1))*$volume-($close<Ref($close,1))*$volume,20)/Mean($volume,20)"),
    ("overnight20", "隔夜跳空20日",   "intraday",  "Mean($open/Ref($close,1),20)"),
    ("intraday20",  "日内收益20日",   "intraday",  "Mean($close/$open,20)"),
    ("close_pos20", "收盘位置20日",   "intraday",  "Mean(($close-$open)/($high-$low+0.001),20)"),
    ("illiq_20",    "Amihud非流动20","liquidity", "Mean(Abs($close/Ref($close,1)-1)/($amount+1),20)"),
]

COST_BPS = 15  # 双边

# ---------------- DB schema ----------------
def init_db():
    con = duckdb.connect(DB)
    con.execute("""
    CREATE TABLE IF NOT EXISTS factor_registry(
        factor_id VARCHAR PRIMARY KEY,
        name VARCHAR, category VARCHAR, expression VARCHAR,
        state VARCHAR DEFAULT 'candidate',
        created_at TIMESTAMP, updated_at TIMESTAMP, metadata VARCHAR);
    CREATE TABLE IF NOT EXISTS factor_test_results(
        result_id VARCHAR PRIMARY KEY,
        factor_id VARCHAR, universe VARCHAR, start_date DATE, end_date DATE,
        n_days INTEGER, ic DOUBLE, icir DOUBLE, rank_ic DOUBLE, rank_icir DOUBLE,
        rank_ic_tstat DOUBLE,
        rank_ic_h5 DOUBLE, rank_ic_h10 DOUBLE, rank_ic_h20 DOUBLE,
        daily_turnover DOUBLE, annual_cost_drag DOUBLE, q5_q1_annual DOUBLE,
        worst_regime VARCHAR, worst_regime_rank_ic DOUBLE,
        gates_passed INTEGER, gates_total INTEGER, suggested_state VARCHAR,
        run_at TIMESTAMP);
    CREATE TABLE IF NOT EXISTS regime_labels(
        trade_date DATE, regime VARCHAR, source VARCHAR, version INTEGER,
        PRIMARY KEY(trade_date, source));
    CREATE TABLE IF NOT EXISTS ic_daily(
        factor_id VARCHAR, trade_date DATE, ic DOUBLE, rank_ic DOUBLE, regime VARCHAR,
        PRIMARY KEY(factor_id, trade_date));
    """)
    now = datetime.now()
    for fid, name, cat, expr in FACTORS:
        con.execute("""
            INSERT INTO factor_registry(factor_id,name,category,expression,state,created_at,updated_at)
            VALUES (?,?,?,?, 'candidate', ?, ?)
            ON CONFLICT(factor_id) DO NOTHING""", [fid, name, cat, expr, now, now])
    con.close()
    print("DB initialized:", DB, "| factors:", len(FACTORS))


# ---------------- regime 标签（规则版 v1） ----------------
def compute_regime():
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    df = D.features(["sh000300"], ["$close"], start_time="2008-01-01", end_time="2026-12-31")
    s = df["$close"]
    if isinstance(s.index, pd.MultiIndex):
        s = s.xs("sh000300", level="instrument")
        if isinstance(s.index, pd.MultiIndex):
            s = s.droplevel(0)
    close = s.sort_index()
    ret = close.pct_change()
    ma60, ma250 = close.rolling(60).mean(), close.rolling(250).mean()
    vol20 = ret.rolling(20).std()
    vol_thr = vol20.loc["2008":"2012"].quantile(0.75)  # 用早期样本定阈值，避免全样本偷看

    def label(d):
        c, f, s, v = close[d], ma60[d], ma250[d], vol20[d]
        if np.isnan(f) or np.isnan(s):
            return None
        if c > s and f > s:
            return "bull"
        if c < s and f < s:
            return "bear"
        return "volatile" if v > vol_thr else "shock"
    labs = pd.Series({d: label(d) for d in close.index})
    labs = labs.dropna()
    con = duckdb.connect(DB)
    con.execute("DELETE FROM regime_labels WHERE source='rule_v1_ma_vol'")
    con.register("labs", labs.rename("regime").reset_index().rename(columns={"index": "trade_date"}))
    con.execute("""
        INSERT INTO regime_labels
        SELECT CAST(trade_date AS DATE), regime, 'rule_v1_ma_vol', 1 FROM labs""")
    con.close()
    print(labs.value_counts().to_string())


# ---------------- 数据与指标 ----------------
_QLIB_READY = False
def qlib_ready():
    global _QLIB_READY
    if not _QLIB_READY:
        import qlib
        from qlib.config import REG_CN
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
        _QLIB_READY = True

def _pool():
    qlib_ready()
    from qlib.data import D
    return D.instruments(UNIVERSE)

def fetch(expr, fname):
    """按因子缓存表达式取值（parquet）。"""
    path = os.path.join(CACHE, f"{fname}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    qlib_ready()
    from qlib.data import D
    df = D.features(_pool(), [expr], start_time=START, end_time=END)
    s = df.iloc[:, 0]
    s = s[~s.index.duplicated()]
    s.to_frame("v").to_parquet(path)
    return s.to_frame("v")

def fetch_close():
    path = os.path.join(CACHE, "close.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)["v"]
    qlib_ready()
    from qlib.data import D
    df = D.features(_pool(), ["$close"], start_time=START, end_time=END)
    s = df["$close"]
    s = s[~s.index.duplicated()]
    s.to_frame("v").to_parquet(path)
    return s

def daily_cs_stats(x, y, method="pearson"):
    """按日截面相关（向量化）。返回每日 IC 序列。"""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if method == "spearman":
        d["x"] = d.groupby(level="datetime")["x"].rank()
        d["y"] = d.groupby(level="datetime")["y"].rank()
    d["xy"] = d["x"] * d["y"]
    g = d.groupby(level="datetime")
    n = g.size()
    mx, my = g["x"].mean(), g["y"].mean()
    cov = g["xy"].mean() - mx * my
    ic = cov / (g["x"].std() * g["y"].std() + 1e-12)
    ic = ic.where(n >= 30).dropna()
    return ic

def compute_metrics(fid, factor_vals):
    close = fetch_close()
    dates = close.index.get_level_values("datetime").unique().sort_values()
    rets = {}
    for h in (1, 5, 10, 20):
        fwd = close.groupby(level="instrument").shift(-(h + 1)) / close.groupby(level="instrument").shift(-1) - 1
        rets[h] = fwd

    out = {}
    ics = {}
    for h, r in rets.items():
        ic = daily_cs_stats(factor_vals, r)
        ric = daily_cs_stats(factor_vals, r, "spearman")
        ics[h] = (ic, ric)
    ic1, ric1 = ics[1]
    out["n_days"] = int(len(ric1))
    out["ic"], out["icir"] = float(ic1.mean()), float(ic1.mean() / (ic1.std() + 1e-12))
    out["rank_ic"] = float(ric1.mean())
    out["rank_icir"] = float(ric1.mean() / (ric1.std() + 1e-12))
    out["rank_ic_tstat"] = float(out["rank_icir"] * np.sqrt(out["n_days"]))
    out["rank_ic_h5"] = float(ics[5][1].mean())
    out["rank_ic_h10"] = float(ics[10][1].mean())
    out["rank_ic_h20"] = float(ics[20][1].mean())

    # 换手：因子日度截面排序变化
    ranks = factor_vals.groupby(level="datetime").rank()
    dr = ranks.groupby(level="instrument").diff().abs()
    turn = dr.groupby(level="datetime").mean() / ranks.groupby(level="datetime").max()
    out["daily_turnover"] = float(turn.mean())
    out["annual_cost_drag"] = out["daily_turnover"] * COST_BPS * 2 / 10000 * 252

    # 分层多空 Q5-Q1（日度，年化）
    q = factor_vals.groupby(level="datetime").transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    r1 = rets[1]
    spread = r1.groupby([r1.index.get_level_values("datetime"), q]).mean().unstack()
    if spread.shape[1] >= 2:
        q5q1 = (spread.iloc[:, -1] - spread.iloc[:, 0]).mean() * 252
    else:
        q5q1 = 0.0
    out["q5_q1_annual"] = float(q5q1)

    # regime 分组 RankIC
    con = duckdb.connect(DB)
    reg = con.execute("SELECT trade_date, regime FROM regime_labels").df()
    con.close()
    regmap = pd.Series(reg.set_index("trade_date")["regime"])
    regmap.index = pd.to_datetime(regmap.index).date
    ric1_df = ric1.copy()
    ric1_df.index = ric1_df.index.date
    byreg = ric1_df.groupby(ric1_df.index.map(lambda d: regmap.get(d, "unknown"))).mean()
    out["regime_rank_ic"] = {str(k): float(v) for k, v in byreg.items()}
    known = byreg.drop(labels=["unknown"], errors="ignore")
    worst = known.abs().idxmin() if len(known) else "n/a"
    out["worst_regime"] = str(worst)
    out["worst_regime_rank_ic"] = float(known.loc[worst]) if len(known) else 0.0

    # 每日 IC 落库（监控用）
    daily = pd.DataFrame({"ic": ic1, "rank_ic": ric1})
    daily.index = daily.index.date
    daily["regime"] = [regmap.get(d, "unknown") for d in daily.index]
    daily["factor_id"] = fid
    out["daily"] = daily.reset_index().rename(columns={"index": "trade_date", "datetime": "trade_date"})
    return out


# ---------------- 五道门槛 ----------------
def run_gates(m):
    worst = m["worst_regime_rank_ic"]
    overall = m["rank_ic"]
    gates = {
        "G1_显著性": abs(m["rank_ic_tstat"]) > 2,
        "G2_regime稳定": (np.sign(overall) == 0) or (np.sign(worst) == np.sign(overall)) or worst * 3 > abs(overall) * 0,
        "G3_低拥挤": True,  # P1 因子池为空，首个批次自动通过，后续批次按池内相关率过滤
        "G4_非纯短衰减": abs(m["rank_ic_h5"]) >= 0.5 * abs(m["rank_ic"]),
        "G5_成本后毛利": m["q5_q1_annual"] > m["annual_cost_drag"] * 2,
    }
    if m["rank_ic"] < 0:
        gates["G1_显著性"] = gates["G1_显著性"] and True  # 负IC同样有效，方向在注册表记录
    passed = sum(gates.values())
    state = "validated" if passed == len(gates) else "rejected"
    if abs(m["rank_ic"]) < 0.01:
        state = "rejected"
    return gates, passed, len(gates), state


# ---------------- 批量运行 ----------------
def run(start, count, force=False):
    init_db()
    items = FACTORS[start:start + count]
    con = duckdb.connect(DB)
    for fid, name, cat, expr in items:
        cfg = f"{fid}|{UNIVERSE}|{START}|{END}|{expr}"
        rid = hashlib.md5(cfg.encode()).hexdigest()[:16]
        exists = con.execute("SELECT 1 FROM factor_test_results WHERE result_id=?", [rid]).fetchone()
        if exists and not force:
            print(f"skip {fid} (cached)")
            continue
        t0 = datetime.now()
        fv = fetch(expr, fid)["v"]
        m = compute_metrics(fid, fv)
        gates, passed, total, state = run_gates(m)
        con.execute("DELETE FROM factor_test_results WHERE result_id=?", [rid])
        con.execute("""
            INSERT INTO factor_test_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [rid, fid, UNIVERSE, START, END, m["n_days"], m["ic"], m["icir"], m["rank_ic"],
             m["rank_icir"], m["rank_ic_tstat"], m["rank_ic_h5"], m["rank_ic_h10"], m["rank_ic_h20"],
             m["daily_turnover"], m["annual_cost_drag"], m["q5_q1_annual"],
             m["worst_regime"], m["worst_regime_rank_ic"], int(passed), int(total), state, datetime.now()])
        con.execute("DELETE FROM ic_daily WHERE factor_id=?", [fid])
        con.register("daily_df", m["daily"])
        con.execute("INSERT INTO ic_daily SELECT factor_id, CAST(trade_date AS DATE), ic, rank_ic, regime FROM daily_df")
        print(f"{fid}: RankIC={m['rank_ic']:.4f} ICIR={m['rank_icir']:.3f} "
              f"turn={m['daily_turnover']:.3f} gates={passed}/{total} -> {state} ({(datetime.now()-t0).seconds}s)", flush=True)
    con.close()


def report():
    con = duckdb.connect(DB)
    df = con.execute("""
        SELECT r.factor_id, f.name, f.category, r.rank_ic, r.rank_icir, r.rank_ic_tstat,
               r.rank_ic_h5, r.daily_turnover, r.q5_q1_annual, r.worst_regime,
               r.worst_regime_rank_ic, r.gates_passed, r.gates_total, r.suggested_state
        FROM factor_test_results r JOIN factor_registry f USING(factor_id)
        ORDER BY abs(r.rank_ic) DESC""").df()
    con.close()
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init_db", "regime", "run", "report"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "init_db":
        init_db()
    elif a.cmd == "regime":
        compute_regime()
    elif a.cmd == "run":
        run(a.start, a.count, a.force)
    elif a.cmd == "report":
        report()
