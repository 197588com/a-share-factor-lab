# -*- coding: utf-8 -*-
"""live_ledger.py — L1 实盘表现台账：每日信号的 realized 收益自动记账。

设计口径（与模型标签 walkforward.yaml 完全一致，保证"适应度函数"与训练同尺度）：
  信号日 T（收盘后出 top10）→ 入场 = T+1 收盘价 →  horizon h 收益 = close(T+1+h)/close(T+1)-1
  基准 = SH000300 同窗口收益；超额 = 个股收益 - 基准收益。
  horizon ∈ {1, 5, 21} 对应标签口径 / 周度 / 月度三条适应度曲线。

数据流：
  data/live/signal_*.csv（每日信号任务落盘）→ live_signals 表（去重）
  → live_ledger 表（逐笔持仓结算，unsettled 行待行情到位后自动结转）

用法：
  python3 live_ledger.py backfill   # 扫描全部信号文件 + 结算所有可结算（首次建账用）
  python3 live_ledger.py update     # 只结算存量 unsettled（每日任务用这个）
  python3 live_ledger.py report     # 打印台账摘要（均值超额 / 累计曲线 / 待结笔数）
"""
import os, sys, glob, json, argparse, datetime
import numpy as np
import pandas as pd
import duckdb

WS = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(WS, "data", "factor_lab.duckdb")
SIGNAL_GLOB = os.path.join(WS, "data", "live", "signal_*.csv")
NAME_MAP = os.path.join(WS, "data", "live", "name_map.csv")
HORIZONS = (1, 5, 21)
CAL = os.path.expanduser("~/.qlib/qlib_data/cn_data/calendars/day.txt")

DDL = """
CREATE TABLE IF NOT EXISTS live_signals(
  signal_date DATE, rank INT, code VARCHAR, instrument VARCHAR,
  score DOUBLE, name VARCHAR, source VARCHAR,
  PRIMARY KEY(signal_date, code));
CREATE TABLE IF NOT EXISTS live_ledger(
  signal_date DATE, code VARCHAR, horizon INT,
  entry_date DATE, entry_px DOUBLE, exit_date DATE, exit_px DOUBLE,
  ret DOUBLE, bench_ret DOUBLE, excess DOUBLE, settled BOOLEAN,
  PRIMARY KEY(signal_date, code, horizon));
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
    m = {}
    for _, r in df.iterrows():
        c = str(r["ts_code"]).split(".")[0]
        m[c] = str(r["name"])
    return m


def ingest_signals(con):
    """扫描 signal_*.csv → live_signals（按 signal_date+code 去重）。"""
    names = load_names()
    files = sorted(glob.glob(SIGNAL_GLOB))
    rows = []
    for f in files:
        d = os.path.basename(f).replace("signal_", "").replace(".csv", "")
        df = pd.read_csv(f, dtype=str)
        df["score"] = df["score"].astype(float)
        # 信号文件是全 universe 打分，台账只记实际信号 top10（与小组件口径一致）
        df = df.sort_values("score", ascending=False).head(10).reset_index(drop=True)
        for i, r in df.iterrows():
            rows.append((d, i + 1, str(r["code"]), str(r["instrument"]),
                         float(r["score"]), names.get(str(r["code"]), ""),
                         os.path.basename(f)))
    ins = pd.DataFrame(rows, columns=["signal_date", "rank", "code", "instrument",
                                      "score", "name", "source"])
    con.register("ins", ins)
    # 清掉旧口径的全 universe 行，再写入 top10
    con.execute("DELETE FROM live_signals")
    con.execute("""INSERT INTO live_signals
                   SELECT signal_date, rank, code, instrument, score, name, source FROM ins""")
    print(f"signals: {len(ins)} 行 top10（files={len(files)}）")
    return len(ins)


def price_frame(codes, days):
    """qlib 拉 close（股票 + SH000300 基准），返回 (date, code) -> close 的 Series。"""
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    from qlib.data import D
    insts = [c if "." in c or c.startswith("SH") else
             ("SH" + c if c.startswith("6") else "SZ" + c) for c in codes] + ["SH000300"]
    df = D.features(insts, ["$close"], start_time=str(days.iloc[0].date()),
                    end_time=str(days.iloc[-1].date()))
    df = df["$close"].unstack("instrument")
    return df


def settle(con):
    """结算所有 unsettled：入场=T+1 收盘，exit=T+1+h 收盘，全部有行情才结转。"""
    days = trading_days()
    d_idx = {d: i for i, d in enumerate(days)}
    # 三类待结算：①未结转 ②已结转但 excess 空（基准数据迟到，如指数停更后补齐，需回填）
    # ③信号表有而台账没有的全新行
    pend = con.execute("""SELECT DISTINCT signal_date, code FROM live_ledger WHERE NOT settled
                          UNION
                          SELECT DISTINCT signal_date, code FROM live_ledger
                          WHERE settled AND excess IS NULL
                          UNION
                          SELECT ls.signal_date, ls.code FROM live_signals ls
                          WHERE (SELECT COUNT(DISTINCT horizon) FROM live_ledger l
                                 WHERE l.signal_date=ls.signal_date AND l.code=ls.code)
                                < %(nh)d""" % {"nh": len(HORIZONS)}).fetchdf()
    if pend.empty:
        print("nothing to settle")
        return
    px = price_frame(pend["code"].tolist(), days)
    bench = px["SH000300"] if "SH000300" in px else None
    rows = []
    skipped = 0
    for _, r in pend.iterrows():
        sd = pd.Timestamp(r["signal_date"])
        code = r["code"]
        col = "SH" + code if code.startswith("6") else "SZ" + code
        if col not in px.columns or sd not in d_idx:
            skipped += 1
            continue
        i = d_idx[sd]
        if i + 1 >= len(days):
            continue  # T+1 还没到
        ed = days.iloc[i + 1]
        e_px = px.at[ed, col] if pd.notna(px.at[ed, col]) else None
        if e_px is None:
            skipped += 1
            continue
        for h in HORIZONS:
            if i + 1 + h >= len(days):
                # 行情日历还没到 exit 日：占位 pending 行（否则该信号从台账静默消失，
                # pending 计数少报、胜率样本永远滞后一天——审计 2026-11-05 发现的缺陷）
                rows.append((str(sd.date()), code, h, str(ed.date()),
                             float(e_px), None, None, None, None, None, False))
                continue
            xd = days.iloc[i + 1 + h]
            x_px = px.at[xd, col]
            if pd.isna(x_px):
                # exit 日在日历内但行情未发布（最新交易日）：同样占位 pending
                rows.append((str(sd.date()), code, h, str(ed.date()),
                             float(e_px), str(xd.date()), None, None, None, None, False))
                continue
            b_e = bench.at[ed] if bench is not None and ed in bench.index else np.nan
            b_x = bench.at[xd] if bench is not None and xd in bench.index else np.nan
            ret = float(x_px / e_px - 1) if pd.notna(x_px) else np.nan
            b_ret = float(b_x / b_e - 1) if pd.notna(b_e) and pd.notna(b_x) else np.nan
            settled = pd.notna(ret)
            rows.append((str(sd.date()), code, h, str(ed.date()),
                         float(e_px), str(xd.date()),
                         float(x_px) if pd.notna(x_px) else None,
                         ret if pd.notna(ret) else None,
                         b_ret if pd.notna(b_ret) else None,
                         (ret - b_ret) if pd.notna(ret) and pd.notna(b_ret) else None,
                         settled))
    if rows:
        df = pd.DataFrame(rows, columns=["signal_date", "code", "horizon", "entry_date",
                                         "entry_px", "exit_date", "exit_px", "ret",
                                         "bench_ret", "excess", "settled"])
        con.register("st", df)
        con.execute("""INSERT OR REPLACE INTO live_ledger
                       SELECT signal_date, code, horizon, entry_date, entry_px, exit_date,
                              exit_px, ret, bench_ret, excess, settled FROM st""")
    ns = con.execute("SELECT COUNT(*) FROM live_ledger WHERE NOT settled").fetchone()[0]
    nf = con.execute("SELECT COUNT(*) FROM live_ledger WHERE settled").fetchone()[0]
    print(f"ledger rows={len(rows)} settled_total={nf} pending={ns} skipped={skipped}")


def report(con):
    df = con.execute("""SELECT signal_date, horizon, ret, excess, settled
                        FROM live_ledger ORDER BY signal_date""").fetchdf()
    pend = con.execute("SELECT COUNT(*) FROM live_ledger WHERE NOT settled").fetchone()[0]
    print(f"== 实盘台账（settled={len(df[df.settled])} / pending={pend}）==")
    for h in HORIZONS:
        d = df[(df.horizon == h) & df.settled]
        if d.empty:
            print(f"h={h:2d}d: 无已结算样本")
            continue
        ex = d.excess.dropna()
        line = f"h={h:2d}d: n={len(d)}  mean_ret={d.ret.mean()*100:+.2f}%"
        if len(ex):
            t = ex.mean() / (ex.std() / np.sqrt(len(ex))) if ex.std() > 0 else np.nan
            line += f"  mean_excess={ex.mean()*100:+.2f}%  t={t:.2f}"
        # 等权 top10 组合口径：按信号日分组等权
        g = d.groupby("signal_date").excess.mean().dropna()
        if len(g) >= 2:
            line += f"  组合累计超额={((1+g).prod()-1)*100:+.2f}%"
        print(line)
    days = df[df.settled].groupby("signal_date").size()
    if not days.empty:
        print(f"信号日覆盖: {days.index.min().date()} ~ {days.index.max().date()} "
              f"({len(days)} 天)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backfill", "update", "report"])
    a = ap.parse_args()
    con = connect()
    if a.cmd == "report":
        report(con)
        return
    ingest_signals(con)
    settle(con)
    if a.cmd == "backfill":
        report(con)
    artifact = {
        "summary": "实盘台账已更新",
        "ranAt": datetime.datetime.now().isoformat(timespec="seconds"),
        "signals": con.execute("SELECT COUNT(*) FROM live_signals").fetchone()[0],
        "settled": con.execute("SELECT COUNT(*) FROM live_ledger WHERE settled").fetchone()[0],
        "pending": con.execute("SELECT COUNT(*) FROM live_ledger WHERE NOT settled").fetchone()[0],
    }
    print("ARTIFACT " + json.dumps(artifact, ensure_ascii=False))


if __name__ == "__main__":
    main()
