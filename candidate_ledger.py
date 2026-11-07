# -*- coding: utf-8 -*-
"""candidate_ledger.py — U2/U3/U4：候选账本 + Bonferroni 阈值 + 月度改动预算。

设计（final_review_audit.md）：
- append-only JSONL，每条带 sha256 链（prev_hash 链接上一条），防事后篡改——
  对应 Bounded Research Loop 的 hash-chained ledger 纪律。
- U3 族错误率：判决阈值 p < 0.05/n，n = 账本中对同一检验窗口已有判决的候选数。
- U4 预算：kind ∈ {signal, structure} 的改动每月 ≤ MAX_PER_MONTH 次进入正式检验，
  超限 gate() 拒绝并要求排队到下月；factor 类（走三层闸门）不计入此预算。

用法：
  from candidate_ledger import record, gate, status
  gate("sig_blend_v1", p_value=0.003, window="2023-2026H1", kind="signal")
    -> {"verdict": "pass", ...}  或 {"verdict": "budget_exceeded"/"insufficient_evidence"}

  python candidate_ledger.py backfill   # 历史回填（幂等，按 candidate_id 去重）
  python candidate_ledger.py status     # 当前阈值与预算状态
"""
import os, sys, json, hashlib
from datetime import date

WS = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(WS, "data", "live", "candidate_ledger.jsonl")
ALPHA = 0.05
MAX_PER_MONTH = {"signal": 2, "structure": 2}

# kind 分类：factor=三层闸门候选；signal=信号层改动；structure=模型结构改动；control=对照轨（不计 n 也不计预算）
COUNTS_TOWARD_N = {"factor", "signal", "structure"}


def _hash(entry: dict, prev_hash: str) -> str:
    payload = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()[:16]


def _last():
    if not os.path.exists(LEDGER):
        return None, "GENESIS"
    last, prev = None, "GENESIS"
    with open(LEDGER, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
                prev = last["hash"]
    return last, prev


def record(candidate_id, kind, window, decision, p_value=None, prespecified=None, notes=""):
    """落账一条判决。幂等：candidate_id 已存在则跳过并返回已有记录。"""
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    _, prev = _last()
    existing = {json.loads(l)["candidate_id"]
                for l in open(LEDGER, encoding="utf-8")} if os.path.exists(LEDGER) else set()
    if candidate_id in existing:
        return None
    entry = {
        "candidate_id": candidate_id, "kind": kind, "window": window,
        "decision": decision, "p_value": p_value,
        "prespecified": prespecified or {},
        "notes": notes, "date": date.today().isoformat(),
        "prev_hash": prev,
    }
    entry["hash"] = _hash(entry, prev)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _entries():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER, encoding="utf-8") if l.strip()]


def threshold(window):
    """U3：该检验资源（window 键）上的当前 Bonferroni 判决阈值。

    window 是「被消耗的检验资源」键，不是日期范围：
      wf_folds8-11  近期四 fold 测试数据（含 increment fold11 口径）
      wf_folds0-11  全量 12 fold 测试数据
    凡在该资源上做过判决（含单因子门禁——IC 评估同样消耗测试年份信息）
    的候选都计入 n。
    """
    n = sum(1 for e in _entries()
            if e["window"] == window and e["kind"] in COUNTS_TOWARD_N)
    return ALPHA / max(n, 1), n


def gate(candidate_id, p_value, window, kind, prespecified=None):
    """U4 + U3：预算检查 + 证据检查，落账后返回判决。"""
    if kind in MAX_PER_MONTH:
        month = date.today().strftime("%Y-%m")
        used = sum(1 for e in _entries()
                   if e["kind"] == kind and e["date"].startswith(month))
        if used >= MAX_PER_MONTH[kind]:
            return {"verdict": "budget_exceeded",
                    "message": f"{kind} 类改动本月预算 {MAX_PER_MONTH[kind]} 已用尽，排队至下月",
                    "used_this_month": used}
    th, n = threshold(window)
    verdict = "pass" if (p_value is not None and p_value < th) else "insufficient_evidence"
    record(candidate_id, kind, window, verdict, p_value, prespecified,
           notes=f"Bonferroni 阈值 {th:.5f} (n={n})")
    return {"verdict": verdict, "threshold": th, "n": n, "p_value": p_value}


def verify_chain():
    prev = "GENESIS"
    for e in _entries():
        h = e.pop("hash")
        if h != _hash(e, prev) or e["prev_hash"] != prev:
            return False
        prev = h
    return True


def status():
    es = _entries()
    th_all, n_all = threshold("wf_folds0-11")
    th_recent, n_recent = threshold("wf_folds8-11")
    month = date.today().strftime("%Y-%m")
    used = {k: sum(1 for e in es if e["kind"] == k and e["date"].startswith(month))
            for k in MAX_PER_MONTH}
    return {"ledger_entries": len(es), "chain_ok": verify_chain(),
            "wf_folds0-11": {"n": n_all, "alpha": ALPHA, "threshold": round(th_all, 5)},
            "wf_folds8-11": {"n": n_recent, "alpha": ALPHA, "threshold": round(th_recent, 5)},
            "budget_used_this_month": used, "budget_max": MAX_PER_MONTH}


# ---------------- 历史回填（2026-11-03 ~ 11-05 终审的七单候选 + 两条对照轨） ----------------
BACKFILL = [
    ("margin_161d", "factor", "wf_folds8-11", "rejected", None,
     {"gate": "三层闸门", "标准": "increment fold11 + 全量WF 配对检验"},
     "MBR/MBAL/MSRT 两融族；fold11 口径淘汰（2026-11-04）"),
    ("dragon_160d", "factor", "wf_folds8-11", "rejected", 0.48,
     {"gate": "全量WF配对检验", "标准": "逐年 p<0.05 一致改善"},
     "LBCNT/LIMUC 龙虎榜族；2023-2026 逐年 t=-0.61/-0.44/+0.70/-0.09 全 p>0.48；转观察名单"),
    ("mf_moneyflow", "factor", "wf_folds0-11", "rejected_gate1", 0.05,
     {"gate": "单因子IC门禁"},
     "主力净流入 4 因子；12 年 |IC|≤0.021 符号漂移，未进模型层；终止该类数据源"),
    ("corrx_rc20", "factor", "wf_folds8-11", "rejected", 0.36,
     {"gate": "全量WF配对检验"},
     "RC20 价量差分相关；逐年 t=-1.14/-0.36/-0.25/-0.34 全不显著；同源冗余注入噪声"),
    ("min_ivol_headr", "factor", "wf_folds0-11", "rejected", 0.72,
     {"gate": "全量WF配对检验", "pool_t": 0.35, "cluster_p": 0.82, "win_rate": "6/12"},
     "IVOL/HEADR 分钟族；单因子 12 年最强（IVOL 均值 -0.028）但模型增量不显著；观察名单"),
    ("evolve1_rpos", "factor", "wf_folds0-11", "rejected", 0.249,
     {"gate": "全量WF配对检验", "pool_t": 1.15, "cluster_p": 0.284, "win_rate": "7/12",
      "预注册": "|IC|≥0.015 且 ≥9/12 同号 且 corr<0.8"},
     "range_pos 族；六单中方向性最强（2026 +0.0072 改善）；留进化池驱动第二批"),
    ("evolve2_rvol_rgap", "factor", "wf_folds0-11", "rejected", 0.939,
     {"gate": "全量WF配对检验", "pool_t": 0.08, "cluster_p": 0.946, "win_rate": "4/12",
      "预注册": "同 evolve1"},
     "强度增强交互批；盲堆交互反噬，比单母本更差；下一代变异改「正交性优先」"),
    ("master_arch_compare", "control", "wf_folds0-11", "rejected", 0.24,
     {"gate": "架构对照轨"},
     "MASTER vs ALSTM 12 年配对 t=1.19 p=0.24，无优势；极限形态定稿选 ALSTM"),
    ("hgb_linear_track", "control", "wf_folds0-11", "rejected", None,
     {"gate": "架构对照轨"},
     "HGB 线性轨样本外 ≈ 0；确认非线性收益真实但有限"),
]


def backfill():
    for cid, kind, win, dec, p, pre, notes in BACKFILL:
        r = record(cid, kind, win, dec, p, pre, f"[历史回填] {notes}")
        print(("+ " if r else "= ") + cid)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        backfill()
    print(json.dumps(status(), ensure_ascii=False, indent=1))
