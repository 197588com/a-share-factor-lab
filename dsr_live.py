# -*- coding: utf-8 -*-
"""dsr_live.py — DSR 实盘复算例行（P21 后续：上线闸门 0.95 的可执行口径复核）。

口径：live 模拟账户（sim_v2，T+1 开盘价成交+跳空纪律+全费用）日度净值对沪深300锚的
日超额收益序列，n_trials=候选账本累计判决数（与 wf 口径一致，随账本增长自动收紧）。
样本 <60 交易日不出结论（insufficient，只报进度）；≥60 后计算 DSR，≥0.95 判 passed。

NOTE：sim_v2 的成交价偏离模型 close 标签口径（P18 已立约），故本复算与 wf 口径 DSR
（attribution.py dsr，0.786）是两数并读的纪律，不互相替代。

每月 2 日 18:52 由 automation 触发；短样本月自动短路为进度汇报。
"""
import json, os, sys, datetime

import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(WS, "data", "factor_lab.duckdb")
HIST = os.path.join(WS, "data", "live", "dsr_live_history.jsonl")
REVIEW_DIR = os.path.join(WS, "data", "live")

MIN_DAYS = 60                    # 预注册：不足 60 个交易日不出 DSR 结论
PASS_GATE = 0.95                 # P4 上线四闸门之一：DSR ≥ 0.95
STATUS_CN = {"insufficient": "样本积累中", "passed": "过闸", "failed": "未过闸",
             "error": "运行失败"}


def load_excess():
    import duckdb
    con = duckdb.connect(DB, read_only=True)
    nav = con.execute("SELECT date, nav, bench_nav FROM sim_nav ORDER BY date").fetchdf()
    if len(nav) < 2:
        return None, nav
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.set_index("date")
    ex = nav["nav"].pct_change() - nav["bench_nav"].pct_change()
    return ex.dropna(), nav


def run_review():
    today = datetime.date.today().isoformat()
    try:
        ex, nav = load_excess()
        if ex is None:
            art = {"asOf": today, "status": "insufficient", "nDays": 0,
                   "needDays": MIN_DAYS,
                   "headline": "模拟账户净值不足 2 行，无法构成日收益序列"}
            return art, None

        # 账本 trials（与 wf DSR 同口径：累计判决数）
        sys.path.insert(0, WS)
        import candidate_ledger as led
        n_trials = led.status()["ledger_entries"]

        n = len(ex)
        nav_total = float(nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1)
        bench_total = float(nav["bench_nav"].iloc[-1] / nav["bench_nav"].iloc[0] - 1)

        if n < MIN_DAYS:
            art = {
                "asOf": today, "status": "insufficient",
                "nDays": n, "needDays": MIN_DAYS,
                "headline": f"样本积累中：{n}/{MIN_DAYS} 交易日，预计还需约 {MIN_DAYS - n} 个交易日",
                "nTrials": n_trials,
                "navTotal": round(nav_total, 6), "benchTotal": round(bench_total, 6),
                "excessTotal": round(nav_total - bench_total, 6),
                "window": f"{nav.index[0].date()} ~ {nav.index[-1].date()}",
            }
        else:
            sys.path.insert(0, os.path.join(WS, "qs-skills", "skill-backtest-overfit", "scripts"))
            from deflated_sharpe import deflated_sharpe_ratio
            r = deflated_sharpe_ratio(ex.to_numpy(dtype=float), n_trials=n_trials).to_dict()
            sr_ann = float(ex.mean() / ex.std(ddof=1) * np.sqrt(244))
            dsr = float(r.get("dsr", r.get("deflated_sharpe_ratio", np.nan)))
            psr = float(r.get("psr", r.get("probabilistic_sharpe_ratio", np.nan)))
            status = "passed" if dsr >= PASS_GATE else "failed"
            art = {
                "asOf": today, "status": status, "nDays": n, "needDays": MIN_DAYS,
                "headline": (f"DSR {dsr:.3f} ≥ {PASS_GATE}，上线闸门通过"
                             if status == "passed" else
                             f"DSR {dsr:.3f} < {PASS_GATE}，上线闸门未过（PSR {psr:.3f}）"),
                "dsr": round(dsr, 4), "psr": round(psr, 4), "srAnn": round(sr_ann, 3),
                "nTrials": n_trials,
                "navTotal": round(nav_total, 6), "benchTotal": round(bench_total, 6),
                "excessTotal": round(nav_total - bench_total, 6),
                "window": f"{ex.index[0].date()} ~ {ex.index[-1].date()}",
            }

        # 历史沉淀
        os.makedirs(os.path.dirname(HIST), exist_ok=True)
        prev = []
        if os.path.exists(HIST):
            prev = [json.loads(l) for l in open(HIST) if l.strip()]
        rec = {"date": today, "nDays": n, "status": art["status"]}
        for k in ("dsr", "psr", "excessTotal"):
            if k in art:
                rec[k] = art[k]
        if not prev or prev[-1]["date"] != today:
            with open(HIST, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            prev.append(rec)
        art["history"] = prev[-12:]
        art["statusCN"] = STATUS_CN[art["status"]]

        md = [
            f"# DSR 实盘复算 · {today}", "",
            f"- 口径：sim_v2 模拟账户（T+1 开盘价、跳空上限、全费用）日超额 vs 沪深300锚",
            f"- 窗口：{art.get('window', '—')}，日收益 n={n}（预注册 ≥{MIN_DAYS} 才出结论）",
            f"- trials：候选账本累计 {n_trials} 条（随账本增长自动收紧）",
            f"- 累计：账户 {nav_total:+.2%} / 基准 {bench_total:+.2%} / 超额 {art.get('excessTotal', nav_total - bench_total):+.2%}",
            f"- 判定：**{art['statusCN']}** —— {art['headline']}",
            "",
            "注：本口径成交价偏离模型 close 标签，与 wf 口径 DSR（attribution.py，"
            "封存线内 0.786）两数并读，不互相替代。",
        ]
        if "dsr" in art:
            md.append(f"- DSR={art['dsr']:.4f} PSR={art['psr']:.4f} 年化SR={art.get('srAnn', float('nan')):.3f}")
        md_path = os.path.join(REVIEW_DIR, f"dsr_live_review_{today}.md")
        open(md_path, "w").write("\n".join(md))
        art["review"] = "\n".join(md)
        return art, md_path
    except Exception as e:
        return {"asOf": today, "status": "error",
                "headline": f"复算失败：{type(e).__name__}: {e}"}, None


def main():
    art, _ = run_review()
    print(json.dumps(art, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
