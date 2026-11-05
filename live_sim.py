# -*- coding: utf-8 -*-
"""live_sim.py — L2 可执行模拟账户：每日 top10 信号的真实可交易口径记账。

借鉴 vn.py PaperAccount 的设计思想（事件驱动订单 → 成交 → 持仓 → 账户），
把"信号质量计量器"(live_ledger.py) 升级为"模拟实盘"：
  初始资金 100 万，每日按最新 top10 等权调仓（与模型 TopkDropout 思路一致：
  仍在榜上的继续持有，不在榜的卖出，新上榜的买入）。

可执行口径 v2（借鉴 t1overnight 执行纪律，与 v1 收盘价口径差异全部显性入账）：
  成交价   = 执行日开盘价 ± 滑点 0.10%（v1 为收盘价；开盘价次日可真实成交）
  跳空上限 = 买入开盘价 > 昨收×1.015 → 当日放弃该票，不顺延次选
  佣金     = 0.025%，最低 5 元（双边）；印花税 0.05%（卖出）；过户费 0.001%（沪市双边）
  手数     = 100 股整手向下取整
  涨停买不进 = 执行日一字涨停（high==low 且 close≥preclose*(1+涨停幅)-0.005）→ 订单挂起
  跌停卖不出 = 执行日一字跌停 → 持仓保留，次日继续尝试卖出
  停牌     = 当日无行情 → 无法交易，按最后价估值
  资金约束 = 现金不够时按可用资金缩减买入数量

用法：
  python3 live_sim.py backfill   # 从全部信号文件重建账户（首次用）
  python3 live_sim.py update     # 增量：处理新信号日 + 逐日盯市（每日任务用）
  python3 live_sim.py report     # 打印账户摘要
"""
import os, sys, glob, json, argparse, datetime
import numpy as np
import pandas as pd
import duckdb

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
DB = os.path.join(WS, "data", "factor_lab.duckdb")
SIGNAL_GLOB = os.path.join(WS, "data", "live", "signal_*.csv")
NAME_MAP = os.path.join(WS, "data", "live", "name_map.csv")
CAL = os.path.expanduser("~/.qlib/qlib_data/cn_data/calendars/day.txt")
CURVE_CSV = os.path.join(WS, "data", "live", "sim_nav_curve.csv")

# ---- 账户参数（账户参数版本化，改动即重建） ----
PARAMS = dict(capital=1_000_000, n_target=10, slip=0.001, comm=0.00025, comm_min=5.0,
              stamp=0.0005, transfer=0.00001, lot=100,
              exec_mode="open", gap_max=0.015, params_ver="sim_v2")

DDL = """
CREATE TABLE IF NOT EXISTS sim_meta(key VARCHAR PRIMARY KEY, value VARCHAR);
CREATE TABLE IF NOT EXISTS sim_orders(
  exec_date DATE, signal_date DATE, code VARCHAR, side VARCHAR,
  qty BIGINT, px DOUBLE, value DOUBLE, fee DOUBLE, status VARCHAR, note VARCHAR);
CREATE TABLE IF NOT EXISTS sim_positions(
  code VARCHAR PRIMARY KEY, qty BIGINT, cost_px DOUBLE, entry_date DATE, last_px DOUBLE);
CREATE TABLE IF NOT EXISTS sim_nav(
  date DATE PRIMARY KEY, cash DOUBLE, mkt_value DOUBLE, nav DOUBLE,
  bench_nav DOUBLE, n_pos INT, turnover DOUBLE, fee_cum DOUBLE);
"""


def connect():
    con = duckdb.connect(DB)
    con.execute(DDL)
    return con


def trading_days():
    days = open(CAL).read().strip().split("\n")
    return pd.to_datetime(pd.Series(days)).sort_values().reset_index(drop=True)


def load_names():
    if not os.path.exists(NAME_MAP):
        return {}
    df = pd.read_csv(NAME_MAP, dtype=str)
    return {str(r["ts_code"]).split(".")[0]: str(r["name"]) for _, r in df.iterrows()}


def limit_ratio(code, names):
    """涨停幅度：创业板/科创板 20%，ST 5%，主板 10%。"""
    nm = names.get(code, "")
    if "ST" in nm.upper():
        return 0.05
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def price_frame(codes, days):
    """qlib 拉 OHLC（股票 + SH000300），(date, code) -> DataFrame。"""
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    from qlib.data import D
    insts = [("SH" + c if c.startswith("6") else "SZ" + c) for c in codes] + ["SH000300"]
    df = D.features(insts, ["$open", "$high", "$low", "$close"],
                    start_time=str(days.iloc[0].date()), end_time=str(days.iloc[-1].date()))
    out = {}
    for c in codes:
        col = "SH" + c if c.startswith("6") else "SZ" + c
        sub = df.xs(col, level="instrument") if col in df.index.get_level_values("instrument") else None
        out[c] = sub
    bench = df.xs("SH000300", level="instrument") if "SH000300" in df.index.get_level_values("instrument") else None
    return out, bench


def get_signals():
    """signal_*.csv -> {signal_date: [(rank, code, score)]}（每日 top10）。"""
    sig = {}
    for f in sorted(glob.glob(SIGNAL_GLOB)):
        d = os.path.basename(f).replace("signal_", "").replace(".csv", "")
        df = pd.read_csv(f, dtype=str)
        df["score"] = df["score"].astype(float)
        df = df.sort_values("score", ascending=False).head(PARAMS["n_target"])
        sig[pd.Timestamp(d)] = [(str(r["code"]), float(r["score"])) for _, r in df.iterrows()]
    return sig


# ---------------- 账户核心 ----------------

def fee_of(side, code, value):
    fee = max(PARAMS["comm_min"], value * PARAMS["comm"])
    if side == "SELL":
        fee += value * PARAMS["stamp"]
    if code.startswith("6"):
        fee += value * PARAMS["transfer"]
    return fee


class Account:
    def __init__(self, con):
        self.con = con
        self.cash = float(con.execute("SELECT value FROM sim_meta WHERE key='cash'").fetchone()[0]) \
            if con.execute("SELECT COUNT(*) FROM sim_meta WHERE key='cash'").fetchone()[0] else PARAMS["capital"]
        self.fee_cum = float(con.execute("SELECT value FROM sim_meta WHERE key='fee_cum'").fetchone()[0]) \
            if con.execute("SELECT COUNT(*) FROM sim_meta WHERE key='fee_cum'").fetchone()[0] else 0.0
        self.pos = {r[0]: dict(qty=int(r[1]), cost=float(r[2]), entry=str(r[3]), last=float(r[4]))
                    for r in con.execute("SELECT code, qty, cost_px, entry_date, last_px FROM sim_positions").fetchall()}

    def save(self):
        self.con.execute("INSERT OR REPLACE INTO sim_meta VALUES ('cash', ?), ('fee_cum', ?), ('params_ver', ?)",
                         [str(self.cash), str(self.fee_cum), PARAMS["params_ver"]])
        self.con.execute("DELETE FROM sim_positions")
        for c, p in self.pos.items():
            self.con.execute("INSERT INTO sim_positions VALUES (?,?,?,?,?)",
                             [c, p["qty"], p["cost"], p["entry"], p["last"]])

    def equity(self):
        return self.cash + sum(p["qty"] * p["last"] for p in self.pos.values())

    def log_order(self, d, sd, code, side, qty, px, value, fee, status, note):
        self.con.execute("INSERT INTO sim_orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                         [str(d.date()), str(sd.date()) if sd else None, code, side,
                          qty, px, value, fee, status, note])

    def sell(self, d, sd, code, bar, names, forced=True):
        p = self.pos[code]
        opx = float(bar["$open"])
        px = opx * (1 - PARAMS["slip"])
        ratio = limit_ratio(code, names)
        one_way = float(bar["$high"]) == float(bar["$low"])
        # 开盘价执行：一字跌停（开盘即封死）卖不出
        if one_way and opx <= bar["__pre"] * (1 - ratio) + 0.005:
            self.log_order(d, sd, code, "SELL", 0, px, 0, 0, "BLOCKED", "一字跌停卖不出")
            return False
        value = p["qty"] * px
        fee = fee_of("SELL", code, value)
        self.cash += value - fee
        self.fee_cum += fee
        self.log_order(d, sd, code, "SELL", p["qty"], px, value, fee, "FILLED", "")
        del self.pos[code]
        return True

    def buy(self, d, sd, code, target_value, bar, names):
        opx = float(bar["$open"])
        px = opx * (1 + PARAMS["slip"])
        ratio = limit_ratio(code, names)
        one_way = float(bar["$high"]) == float(bar["$low"])
        # 跳空上限纪律（t1overnight 口径）：开盘 > 昨收×(1+gap_max) 当日放弃，不顺延次选
        if opx > bar["__pre"] * (1 + PARAMS["gap_max"]):
            self.log_order(d, sd, code, "BUY", 0, px, 0, 0, "BLOCKED",
                           f"跳空超限放弃 open>{PARAMS['gap_max']*100:.1f}%")
            return False
        # 一字涨停（开盘即封死）买不进
        if one_way and opx >= bar["__pre"] * (1 + ratio) - 0.005:
            self.log_order(d, sd, code, "BUY", 0, px, 0, 0, "BLOCKED", "一字涨停买不进")
            return False
        qty = int(min(target_value, self.cash) / px / PARAMS["lot"]) * PARAMS["lot"]
        if qty <= 0:
            self.log_order(d, sd, code, "BUY", 0, px, 0, 0, "SKIPPED", "现金不足")
            return False
        value = qty * px
        fee = fee_of("BUY", code, value)
        while value + fee > self.cash and qty > 0:
            qty -= PARAMS["lot"]
            value = qty * px
            fee = fee_of("BUY", code, value)
        if qty <= 0:
            self.log_order(d, sd, code, "BUY", 0, px, 0, 0, "SKIPPED", "现金不足")
            return False
        self.cash -= value + fee
        self.fee_cum += fee
        if code in self.pos:
            old = self.pos[code]
            new_qty = old["qty"] + qty
            self.pos[code] = dict(qty=new_qty, cost=(old["cost"] * old["qty"] + px * qty) / new_qty,
                                  entry=old["entry"], last=px)
        else:
            self.pos[code] = dict(qty=qty, cost=px, entry=str(d.date()), last=px)
        note = "" if target_value <= self.cash + value else "按可用资金缩减"
        self.log_order(d, sd, code, "BUY", qty, px, value, fee, "FILLED", note)
        return True

    def rebalance(self, d, sd, targets, bars, names):
        """按目标 top10 等权调仓：先卖后买。返回当日成交额。"""
        tgt_codes = [c for c, _ in targets]
        turnover = 0.0
        # 卖出：持仓不在目标里的全部卖（开盘价执行）
        for code in list(self.pos.keys()):
            if code not in tgt_codes:
                bar = bars.get(code)
                if bar is None or pd.isna(bar.get("$open")):
                    self.log_order(d, sd, code, "SELL", 0, 0, 0, 0, "HELD", "停牌，继续持有")
                    continue
                self.sell(d, sd, code, bar, names)
        # 等权目标市值（按昨收估值，开盘价下单）
        eq = self.equity()
        tgt_value = eq / PARAMS["n_target"]
        # 买入：目标里没持仓的买
        for code, score in targets:
            if code in self.pos:
                continue
            bar = bars.get(code)
            if bar is None or pd.isna(bar.get("$open")):
                self.log_order(d, sd, code, "BUY", 0, 0, 0, 0, "HELD", "停牌，无法买入")
                continue
            self.buy(d, sd, code, tgt_value, bar, names)
        tv = self.con.execute(
            "SELECT COALESCE(SUM(value),0) FROM sim_orders WHERE exec_date=? AND status='FILLED'",
            [str(d.date())]).fetchone()[0]
        return float(tv)

    def mark(self, d, bars):
        for code, p in self.pos.items():
            bar = bars.get(code)
            if bar is not None and not pd.isna(bar.get("$close")):
                p["last"] = float(bar["$close"])


# ---------------- 主流程 ----------------

def run(con, rebuild=False):
    days = trading_days()
    sig = get_signals()
    names = load_names()
    if rebuild:
        for t in ("sim_orders", "sim_positions", "sim_nav", "sim_meta"):
            con.execute(f"DELETE FROM {t}")
    acct = Account(con)

    done_dates = set(pd.Timestamp(x[0]) for x in
                     con.execute("SELECT DISTINCT exec_date FROM sim_orders").fetchall())
    # 每个信号日对应一个执行日 T+1
    plans = []
    for sd in sorted(sig.keys()):
        i = days.searchsorted(sd)
        if i + 1 < len(days):
            plans.append((sd, days.iloc[i + 1]))
    latest = days.iloc[-1]

    # 需要的股票 = 所有信号 top10 ∪ 持仓
    all_codes = sorted({c for s in sig.values() for c, _ in s} | set(acct.pos.keys()))
    bars_by_day = {}  # 惰性按日取
    pf_cache = {}

    def bars_on(d):
        if d not in bars_by_day:
            out = {}
            for c in all_codes:
                sub = pf_cache.get(c)
                if sub is not None and d in sub.index:
                    row = sub.loc[d].copy()
                    i = sub.index.get_loc(d)
                    row["__pre"] = float(sub["$close"].iloc[i - 1]) if i > 0 else float(row["$open"])
                    if not pd.isna(row["$close"]):
                        out[c] = row
            bars_by_day[d] = out
        return bars_by_day[d]

    pf_cache, bench = price_frame(all_codes, days)
    bench = bench["$close"] if bench is not None else None

    n_new = 0
    # 单笔交易的执行日 -> (signal_date, targets)
    plan_by_day = {ed: (sd, sig[sd]) for sd, ed in plans}

    # 逐日推进：从最后一条 nav 的次日（或首个执行日）到 latest
    # 每个交易日：先按当日计划调仓（若有），再盯市，再落当日 nav 行——
    # 保证 nav 表里的 cash/持仓/费用是"当日收盘后"的真实状态
    last_nav = con.execute("SELECT MAX(date) FROM sim_nav").fetchone()[0]
    first_nav = con.execute("SELECT MIN(date) FROM sim_nav").fetchone()[0]
    start_i = days.searchsorted(pd.Timestamp(last_nav) + pd.Timedelta(days=1)) if last_nav else \
        days.searchsorted(plans[0][1]) if plans else len(days)
    # 基准锚点固定为账户首日（首次运行确定，之后增量沿用，保证曲线可比）
    anchor = pd.Timestamp(first_nav) if first_nav else (days[start_i] if start_i < len(days) else None)
    if anchor is not None and con.execute("SELECT COUNT(*) FROM sim_meta WHERE key='bench_anchor'").fetchone()[0] == 0:
        con.execute("INSERT INTO sim_meta VALUES ('bench_anchor', ?)", [str(anchor.date())])
    anchor = pd.Timestamp(con.execute("SELECT value FROM sim_meta WHERE key='bench_anchor'").fetchone()[0])
    bench0 = bench.at[anchor] if bench is not None and anchor in bench.index else np.nan
    rows = []
    for i in range(start_i, len(days)):
        d = days.iloc[i]
        if d in plan_by_day and d not in done_dates and d <= latest:
            sd, targets = plan_by_day[d]
            # 开盘价执行：持仓已按昨收估值，先调仓再按收盘盯市
            acct.rebalance(d, sd, targets, bars_on(d), names)
            acct.mark(d, bars_on(d))  # 新仓按收盘价盯市，滑点成本当日显性入账
            acct.save()
            done_dates.add(d)
            n_new += 1
        else:
            acct.mark(d, bars_on(d))
        nav = acct.equity()
        mv = sum(p["qty"] * p["last"] for p in acct.pos.values())
        b_nav = float(bench.at[d] / bench0 * PARAMS["capital"]) if bench is not None and d in bench.index else None
        rows.append((str(d.date()), acct.cash, mv, nav, b_nav,
                     len(acct.pos),
                     float(con.execute("SELECT COALESCE(SUM(value),0) FROM sim_orders WHERE exec_date=? AND status='FILLED'",
                                       [str(d.date())]).fetchone()[0]),
                     acct.fee_cum))
    if rows:
        df = pd.DataFrame(rows, columns=["date", "cash", "mkt_value", "nav", "bench_nav",
                                         "n_pos", "turnover", "fee_cum"])
        con.register("nv", df)
        con.execute("""INSERT OR REPLACE INTO sim_nav
                       SELECT date, cash, mkt_value, nav, bench_nav, n_pos, turnover, fee_cum FROM nv""")
    acct.save()
    print(f"sim: 新执行日={n_new} nav_rows={len(rows)} pos={len(acct.pos)} "
          f"cash={acct.cash:,.0f} fee_cum={acct.fee_cum:,.0f}")


def report(con):
    nav = con.execute("SELECT * FROM sim_nav ORDER BY date").fetchdf()
    if nav.empty:
        print("sim_nav 为空")
        return
    nav.to_csv(CURVE_CSV, index=False)
    last = nav.iloc[-1]
    ret = last.nav / PARAMS["capital"] - 1
    b_ret = last.bench_nav / PARAMS["capital"] - 1 if last.bench_nav else np.nan
    cummax = nav.nav.cummax()
    mdd = (nav.nav / cummax - 1).min()
    n_days = len(nav)
    ann = (1 + ret) ** (250 / n_days) - 1 if n_days else 0
    orders = con.execute("""SELECT status, COUNT(*) FROM sim_orders GROUP BY status""").fetchall()
    blk = con.execute("""SELECT exec_date, code, side, note FROM sim_orders
                         WHERE status IN ('BLOCKED','HELD','SKIPPED') ORDER BY exec_date""").fetchdf()
    to = con.execute("SELECT COALESCE(SUM(turnover),0) FROM sim_nav").fetchone()[0]
    print(f"== 模拟账户（{PARAMS['params_ver']}，本金 {PARAMS['capital']:,.0f}）==")
    print(f"净值 {last.nav:,.0f}  总收益 {ret*100:+.2f}%  基准 {b_ret*100:+.2f}%  超额 {(ret-b_ret)*100:+.2f}%")
    print(f"年化 {ann*100:+.2f}%  最大回撤 {mdd*100:.2f}%  交易日 {n_days}")
    print(f"费用累计 {last.fee_cum:,.0f}  换手累计 {to:,.0f}  订单状态 {dict(orders)}")
    print(f"持仓 {int(last.n_pos)} 只，现金 {last.cash:,.0f}")
    if len(blk):
        print(f"-- 受阻/挂起订单 {len(blk)} 笔 --")
        print(blk.to_string(index=False))


def artifact(con):
    nav = con.execute("SELECT * FROM sim_nav ORDER BY date").fetchdf()
    last = nav.iloc[-1] if len(nav) else None
    ret = last.nav / PARAMS["capital"] - 1 if last is not None else 0
    b_ret = last.bench_nav / PARAMS["capital"] - 1 if last is not None and last.bench_nav else None
    mdd = float((nav.nav / nav.nav.cummax() - 1).min()) if len(nav) else 0
    blk = con.execute("""SELECT COUNT(*) FROM sim_orders
                         WHERE status IN ('BLOCKED','HELD','SKIPPED')""").fetchone()[0]
    art = {
        "summary": f"模拟账户 净值{last.nav:,.0f} 收益{ret*100:+.2f}% 超额{((ret-b_ret) if b_ret is not None else 0)*100:+.2f}%",
        "ranAt": datetime.datetime.now().isoformat(timespec="seconds"),
        "paramsVer": PARAMS["params_ver"],
        "nav": round(float(last.nav), 2) if last is not None else None,
        "totalRet": round(float(ret), 6),
        "benchRet": round(float(b_ret), 6) if b_ret is not None else None,
        "excessRet": round(float(ret - b_ret), 6) if b_ret is not None else None,
        "maxDD": round(mdd, 6),
        "feeCum": round(float(last.fee_cum), 2) if last is not None else 0,
        "nPos": int(last.n_pos) if last is not None else 0,
        "cash": round(float(last.cash), 2) if last is not None else None,
        "blockedOrders": int(blk),
        "asOf": str(last.date) if last is not None else None,
        "curve": ([{**r, "date": str(r["date"])} for r in
                   nav[["date", "nav", "bench_nav"]].tail(120).to_dict("records")]
                  if len(nav) else []),
    }
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backfill", "update", "report"])
    a = ap.parse_args()
    con = connect()
    if a.cmd == "report":
        report(con)
        return
    run(con, rebuild=(a.cmd == "backfill"))
    report(con)
    art = artifact(con)
    try:
        import event_bus
        event_bus.publish("sim_done", {"asOf": art["asOf"], "nav": art["nav"],
                                       "excessRet": art["excessRet"], "blockedOrders": art["blockedOrders"]})
    except Exception as e:
        print(f"event publish skipped: {e}")
    print("ARTIFACT " + json.dumps(art, ensure_ascii=False))


if __name__ == "__main__":
    main()
