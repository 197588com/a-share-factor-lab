# -*- coding: utf-8 -*-
"""attribution_review.py — 月度超额归因例行审查（P21 后续长期机制）。

每月重跑 attribution.py（数据全缓存，秒级），对照预注册阈值判定：
  trigger  alpha < 5% 或 |VOL beta| > 0.40   → 触发结构审查（人机决策，不自动改模型）
  watch    alpha < 7.5% 或 |VOL beta| > 0.30 → 关注，下月复查
  healthy  其余
判定只驱动审查报告，任何结构改动必须重走三层闸门 + 候选账本（Bonferroni 预算）。
历史沉淀 data/live/attribution_history.jsonl（供小组件趋势带）。
"""
import json, os, subprocess, sys, datetime

WS = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(WS, "data", "backtest", "attribution.json")
HIST = os.path.join(WS, "data", "live", "attribution_history.jsonl")
REVIEW_DIR = os.path.join(WS, "data", "live")

THRESH = {"alpha_trigger": 0.05, "alpha_watch": 0.075,
          "volbeta_trigger": 0.40, "volbeta_watch": 0.30}
STATUS_CN = {"healthy": "健康", "watch": "关注", "trigger": "触发审查", "error": "运行失败"}


def run_review():
    today = datetime.date.today().isoformat()
    # 1. 重跑归因（幂等，缓存命中时秒级）
    r = subprocess.run([sys.executable, os.path.join(WS, "attribution.py")],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        art = {"asOf": today, "status": "error",
               "headline": f"归因重跑失败：{r.stderr.strip()[-200:]}"}
        return art, None
    a = json.load(open(OUT_JSON))
    alpha = a["alpha_ann"]
    volb = a["betas"]["VOL"]["beta"]
    volt = a["betas"]["VOL"]["t_nw"]
    smb = a["betas"]["SMB"]["beta"]

    # 2. 预注册阈值判定
    reasons = []
    if alpha < THRESH["alpha_trigger"]:
        reasons.append(f"年化 alpha {alpha:.1%} 跌破 5% 触发线")
    if abs(volb) > THRESH["volbeta_trigger"]:
        reasons.append(f"|VOL beta| {abs(volb):.2f} 突破 0.40 触发线（做空波动率脐带过紧）")
    if reasons:
        status = "trigger"
    elif alpha < THRESH["alpha_watch"] or abs(volb) > THRESH["volbeta_watch"]:
        status = "watch"
        if alpha < THRESH["alpha_watch"]:
            reasons.append(f"年化 alpha {alpha:.1%} 低于 7.5% 关注线")
        if abs(volb) > THRESH["volbeta_watch"]:
            reasons.append(f"|VOL beta| {abs(volb):.2f} 超过 0.30 关注线")
    else:
        status = "healthy"
        reasons.append("alpha 与 VOL 暴露均在安全区内")

    # 3. 历史沉淀（断点追加，保留全部）
    os.makedirs(os.path.dirname(HIST), exist_ok=True)
    rec = {"date": today, "alpha": round(alpha, 6), "vol_beta": round(volb, 6),
           "r2": round(a["r2"], 4), "status": status}
    prev = []
    if os.path.exists(HIST):
        prev = [json.loads(l) for l in open(HIST) if l.strip()]
    if not prev or prev[-1]["date"] != today:
        with open(HIST, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        prev.append(rec)
    history = prev[-12:]

    # 4. 审查报告（md 工件，供人工翻阅）
    md = [
        f"# 超额归因月度审查 · {today}",
        "",
        f"- 窗口：{a['window']}（封存线内，n={a['n_days']} 日）",
        f"- 原始超额年化：{a['excess_ann_raw']:.2%}",
        f"- **四因子剥离后 alpha：{alpha:.2%}/年（NW t={a['alpha_t_nw']:.2f}）**",
        f"- R²（风格解释份额）：{a['r2']:.1%}",
        f"- 风格暴露：SMB {smb:+.3f}（t={a['betas']['SMB']['t_nw']:.2f}，"
        f"小市值伪装检验）/ VOL {volb:+.3f}（t={volt:.2f}，做空波动率脐带）",
        f"- 判定：**{STATUS_CN[status]}** —— {'；'.join(reasons)}",
        "",
        "## 阈值约定（预注册）",
        f"- 触发结构审查：alpha < {THRESH['alpha_trigger']:.0%} 或 |VOL beta| > {THRESH['volbeta_trigger']:.2f}",
        f"- 关注复查：alpha < {THRESH['alpha_watch']:.1%} 或 |VOL beta| > {THRESH['volbeta_watch']:.2f}",
        "- 触发后的任何结构改动须重走三层闸门并计入党选账本月度预算",
    ]
    md_path = os.path.join(REVIEW_DIR, f"attribution_review_{today}.md")
    open(md_path, "w").write("\n".join(md))

    art = {
        "asOf": today, "window": a["window"], "nDays": a["n_days"],
        "status": status, "statusCN": STATUS_CN[status],
        "headline": f"{STATUS_CN[status]}：{'；'.join(reasons)}",
        "alphaAnn": round(alpha, 6), "alphaTNW": round(a["alpha_t_nw"], 3),
        "r2": round(a["r2"], 4), "excessAnnRaw": round(a["excess_ann_raw"], 6),
        "betas": {k: {"beta": round(v["beta"], 4), "t": round(v["t_nw"], 2)}
                  for k, v in a["betas"].items()},
        "thresholds": THRESH, "history": history,
        "review": "\n".join(md),
    }
    return art, md_path


def main():
    art, _ = run_review()
    print(json.dumps(art, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
