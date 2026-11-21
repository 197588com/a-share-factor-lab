# -*- coding: utf-8 -*-
"""从终审 JSON 和执行产物生成 P27/P30 报告，禁止手工转录核心数字。"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import research_verdict as rv
import shadow_execution as se

WS = Path(__file__).resolve().parent
P25 = WS / "data" / "backtest" / "execution_replay"
P25X2 = WS / "data" / "backtest" / "execution_replay_cost_x2"
P27 = WS / "data" / "p27_universe_expansion"
P30 = WS / "data" / "p30_execution_aware"


def label_diagnostics(scores, min_names=20):
    """逐日标签 RankIC 与等权五分位标签收益；绝非可执行组合收益。"""
    required = {"date", "instrument", "score", "label"}
    if not required.issubset(scores.columns):
        raise ValueError(f"评分缺字段：{sorted(required - set(scores.columns))}")
    if scores.duplicated(["date", "instrument"]).any():
        raise ValueError("评分存在重复的日期／股票")
    clean = scores.dropna(subset=["date", "instrument", "score", "label"]).copy()
    clean = clean[np.isfinite(clean["score"]) & np.isfinite(clean["label"])]
    clean["date"] = pd.to_datetime(clean["date"])
    ic = clean.groupby("date", sort=True).filter(lambda g: len(g) >= min_names).groupby("date").apply(
        lambda g: g["score"].corr(g["label"], method="spearman"), include_groups=False).dropna()
    ranked = clean.groupby("date", sort=True).filter(lambda g: len(g) >= 5 * min_names).copy()
    if ranked.empty:
        raise ValueError("没有足够的逐日有效标签构造五分位诊断")
    ranked["quintile"] = ranked.groupby("date")["score"].rank(method="first").groupby(ranked["date"]).transform(
        lambda rank: pd.qcut(rank, 5, labels=False))
    quintiles = ranked.groupby("quintile")["label"].mean().reindex(range(5))
    if quintiles.isna().any():
        raise ValueError("五分位标签收益不完整")
    return ic, quintiles


def diagnostic_overview(candidate, daily_path, score_paths, output_path):
    """四联图只读取冻结执行／评分产物，不修改候选、参数或判决。"""
    daily = pd.read_parquet(daily_path).sort_values("datetime")
    if daily.empty or daily["datetime"].duplicated().any():
        raise ValueError(f"{candidate} 日收益为空或日期重复")
    scores = pd.concat([pd.read_parquet(p) for p in score_paths], ignore_index=True)
    ic, quintiles = label_diagnostics(scores)
    dates = pd.to_datetime(daily["datetime"])
    strategy = (1 + daily["strategy_return"].astype(float)).cumprod()
    benchmark = (1 + daily["bench"].astype(float)).cumprod()
    relative = strategy / benchmark
    strategy_dd = strategy / strategy.cummax() - 1
    benchmark_dd = benchmark / benchmark.cummax() - 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(dates, strategy, label="Strategy net", linewidth=1.3)
    ax.plot(dates, benchmark, label="Benchmark", linewidth=1.1)
    ax.plot(dates, relative, label="Relative (strategy / benchmark)", linewidth=1.1)
    ax.set_title(f"{candidate}: executable NAV")
    ax.set_ylabel("NAV, start = 1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.plot(ic.index, ic.cumsum(), color="darkorange", linewidth=1.3)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Cumulative daily label RankIC (diagnostic only)")
    ax.set_ylabel("Sum of daily Spearman IC")
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    values = quintiles.to_numpy() * 1e4
    ax.bar([f"Q{i}" for i in range(1, 6)], values,
           color=["#c44e52" if x < 0 else "#4c72b0" for x in values])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Mean next-open label by score quintile (not returns)")
    ax.set_ylabel("Mean label (bp)")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    ax.plot(dates, strategy_dd, label="Strategy net", linewidth=1.3)
    ax.plot(dates, benchmark_dd, label="Benchmark", linewidth=1.1)
    ax.set_title("Executable drawdown")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {"candidate": candidate, "output": str(output_path), "daily_rows": len(daily),
            "rank_ic_days": len(ic), "score_rows": len(scores)}


def generate_diagnostics():
    specs = {
        "p27_csi500_alstm158": (
            P27 / "execution" / "capital_1000000" / "daily_all.parquet",
            [P27 / "scores" / f"csi500_alstm_fold{i}.parquet" for i in (8, 9, 10, 11)],
            WS / "p27_csi500_diagnostic_overview.png"),
        "p30_open_gross": (
            P30 / "execution" / "p30_open_gross" / "daily_all.parquet",
            [P30 / "scores" / f"p30_open_gross_fold{i}.parquet" for i in (8, 9, 10, 11)],
            WS / "p30_open_gross_diagnostic_overview.png"),
        "p30_open_net_masked": (
            P30 / "execution" / "p30_open_net_masked" / "daily_all.parquet",
            [P30 / "scores" / f"p30_open_net_masked_fold{i}.parquet" for i in (8, 9, 10, 11)],
            WS / "p30_open_net_masked_diagnostic_overview.png"),
    }
    return {candidate: diagnostic_overview(candidate, *paths) for candidate, paths in specs.items()}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}%"


def num(x, digits=3):
    if x is None:
        return "—"
    if x != 0 and abs(x) < 0.001:
        return f"{x:.3g}"
    return f"{x:.{digits}f}"


def p_value(x):
    return "—" if x is None else f"{x:.6g}"


def main_rows():
    frozen = load(P25 / "summary.json")
    frozen_dsr = load(P25 / "dsr_execution_replay.json")
    frozen_daily = pd.read_parquet(P25 / "daily_all.parquet")
    frozen_orders = se.order_stats(pd.read_parquet(P25 / "orders_all.parquet"))
    stress = load(P25X2 / "summary.json")
    verdicts = load(WS / "data" / "p27_p30_final_verdicts.json")
    rows = {
        "P25（CSI300 冻结基线）": {
            "overall": frozen["overall"], "stress": stress["overall"], "orders": frozen_orders,
            "dsr": frozen_dsr["deflated_sharpe_ratio"], "dsr_trials": frozen_dsr["n_trials"],
            "cluster": rv.annual_cluster_positive(frozen_daily),
            "cluster_kind": "自身超额>0", "decision": "冻结主口径；DSR 未过闸",
        }
    }
    labels = {"p27_csi500_alstm158": "P27 CSI500／ALSTM158",
              "p30_open_gross": "P30 开盘毛收益目标",
              "p30_open_net_masked": "P30 可执行净成本目标"}
    for candidate, label in labels.items():
        v = verdicts[candidate]
        rows[label] = {
            "overall": v["overall"], "stress": v["double_cost_overall"], "orders": v["orders"],
            "dsr": v["dsr"]["deflated_sharpe_ratio"], "dsr_trials": v["dsr"]["n_trials"],
            "cluster": v["annual_cluster"],
            "cluster_kind": "自身超额>0" if candidate.startswith("p27") else "相对P25增量>0",
            "decision": v["decision"],
        }
    return rows, verdicts


def comparison_table(rows):
    lines = [
        "| 候选 | 年化净超额 | 净 IR | MDD | 平均单边换手 | 平均现金权重 | 买入未成交率 | 双倍成本年化净超额 | DSR（trials） | 年度聚类 p／口径 | 判决 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for label, r in rows.items():
        o, x2, orders = r["overall"], r["stress"], r["orders"]
        lines.append(
            f"| {label} | {pct(o['annual_net_excess'])} | {num(o['net_ir'])} | {pct(o['mdd'])} | "
            f"{pct(o['mean_one_way_turnover'])} | {pct(o['mean_cash_weight'])} | "
            f"{pct(orders['unfilled_buy_rate'])} | {pct(x2['annual_net_excess'])} | "
            f"{num(r['dsr'])}（{r['dsr_trials']}） | {p_value(r['cluster']['p_one_sided_positive'])}／{r['cluster_kind']} | {r['decision']} |")
    return "\n".join(lines)


def yearly_table(verdicts):
    lines = ["| 候选 | 年度／fold | 年化净超额 | 净 IR | MDD | 买入未成交率 |", "|---|---|---:|---:|---:|---:|"]
    labels = {"p27_csi500_alstm158": "P27 CSI500", "p30_open_gross": "P30 开盘毛收益",
              "p30_open_net_masked": "P30 可执行净成本"}
    for candidate, label in labels.items():
        if candidate not in verdicts:
            continue
        for fold in verdicts[candidate]["folds"]:
            lines.append(f"| {label} | {fold['year']}／{fold['fold']} | {pct(fold['annual_net_excess'])} | "
                         f"{num(fold['net_ir'])} | {pct(fold['mdd'])} | {pct(fold['unfilled_buy_rate'])} |")
    return "\n".join(lines)


def execution_diagnostics_table(rows):
    """补齐预注册要求的累计收益与实际账户摩擦，不只展示超额和 IR。"""
    lines = ["| 候选 | 累计策略净收益 | 累计费用（元） | 累计滑点（元） | 平均持仓 |",
             "|---|---:|---:|---:|---:|"]
    for label, r in rows.items():
        o = r["overall"]
        lines.append(f"| {label} | {pct(o['total_net_return'])} | {o['fees_sum']:.2f} | "
                     f"{o['slippage_sum']:.2f} | {o['mean_positions']:.2f} |")
    return "\n".join(lines)


def p30_rank_ic_table():
    """逐折原样读取评估 JSON；RankIC 仅诊断，不参与绩效晋升。"""
    lines = ["| 候选 | fold | 最佳 epoch | 验证集 IC | 测试 RankIC（诊断） | 测试 IC（诊断） |",
             "|---|---:|---:|---:|---:|---:|"]
    for candidate in ("p30_open_gross", "p30_open_net_masked"):
        for fold in (8, 9, 10, 11):
            result = load(P30 / "metrics" / f"eval_{candidate}_fold{fold}.json")
            metrics = result["metrics"]
            lines.append(f"| {candidate} | {fold} | {result['best_epoch']} | "
                         f"{num(result['best_valid_ic'], 6)} | {num(metrics['RankIC'], 6)} | "
                         f"{num(metrics['IC'], 6)} |")
    return "\n".join(lines)


def capacity_table(verdicts):
    capacity = verdicts["p27_csi500_alstm158"]["capacity"]
    lines = ["| 初始资金 | 年化净超额 | 净 IR | MDD | 平均／最大持仓 | 平均现金权重 | 买入未成交率 | 一手跳过率 | 成交额占比 p95／最大 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in capacity["capacities"]:
        o, orders = item["overall"], item["orders"]
        part = item["participation"]
        p95 = "—" if part["participation_p95"] is None else pct(part["participation_p95"])
        maximum = "—" if part["participation_max"] is None else pct(part["participation_max"])
        lines.append(f"| {item['capital'] / 1e6:.0f} 百万元 | {pct(o['annual_net_excess'])} | {num(o['net_ir'])} | "
                     f"{pct(o['mdd'])} | {o['mean_positions']:.1f}／{o['max_positions']} | "
                     f"{pct(o['mean_cash_weight'])} | {pct(orders['unfilled_buy_rate'])} | "
                     f"{pct(orders['one_lot_skip_rate'])} | {p95}／{maximum} |")
    return "\n".join(lines)


def gate_lines(verdicts):
    lines = []
    for candidate, v in verdicts.items():
        failed = "、".join(v["failed_gates"]) if v["failed_gates"] else "无（仍须独立复核）"
        lines.append(f"- `{candidate}`：**{v['decision']}**；失败闸门：{failed}。")
    return "\n".join(lines)


def generate():
    rows, verdicts = main_rows()
    health = load(P27 / "meta" / "data_health.json")
    limits = load(P27 / "meta" / "limit_cross_validation.json")
    actions = load(P27 / "meta" / "action_cross_validation.json")
    execution_gates = load(P27 / "meta" / "execution_data_gates.json")
    membership = load(P27 / "meta" / "constituent_summary.json")
    multiplicity = next(iter(verdicts.values()))["multiplicity"]
    trial_sr_count = next(iter(verdicts.values()))["dsr"]["trial_sr_count"]
    comparison = comparison_table(rows)
    diagnostics = generate_diagnostics()
    if set(diagnostics) != set(verdicts):
        raise AssertionError("诊断图与终审候选不一致")
    common = f"""

## 累计收益与账户摩擦

{execution_diagnostics_table(rows)}

费用和滑点是四个独立 fold 账户的累计名义金额，不是年化比率；累计策略净收益按逐日收益复利链接。

## 统计口径

- 正式窗口为 2023-01-03～2026-06-30，folds 8–11；P25 数值保持冻结，压力结果来自独立目录。
- 当前整批 DSR `n_trials={multiplicity['n_trials']}`；Bonferroni 基数 `n={multiplicity['bonferroni_n']}`，阈值 `{multiplicity['bonferroni_threshold']:.12g}`。
- P25 的 DSR 仍是其冻结时 `n_trials=9` 的历史判决；P27／P30 用当前整批真实可执行 SR 估计，不为数据闸失败项伪造收益。
- 试验间 SR 方差仅有 {trial_sr_count}/{multiplicity['n_trials']} 条真实可执行序列覆盖；旧候选 SR 未完整留存是残余限制，不能把这称作完整的历史 SR 台账，也不使用单序列方差回退。
- 所有候选即使全过也只进入 `pending_independent_review`，不自动解除生产冻结。
"""
    p27_diagnostic = """
## 可执行净值与标签诊断

![P27 可执行净值、累计标签 RankIC、五分位标签收益与回撤](p27_csi500_diagnostic_overview.png)

左上与右下来自逐日可执行账户；累计 RankIC 与五分位柱图仅是模型标签诊断，不是可成交收益，不参与调参或晋升。
"""
    p30_diagnostics = """
## 可执行净值与标签诊断

![P30 开盘毛收益目标诊断](p30_open_gross_diagnostic_overview.png)

![P30 可执行净成本目标诊断](p30_open_net_masked_diagnostic_overview.png)

左上与右下来自逐日可执行账户；累计 RankIC 与五分位柱图仅是模型标签诊断，不是可成交收益，不参与调参或晋升。
"""
    p27 = f"""# P27 宇宙扩展终审报告

生成日期：{pd.Timestamp.now().date().isoformat()}。本报告由 `p27_p30_report.py` 从执行产物生成。

## 结论

{comparison}
{common}
{p27_diagnostic}

## 数据闸事实

- 中证 500：{membership['days']} 个交易日、{membership['instruments']} 只历史成分，每日 {membership['count_min']}–{membership['count_max']} 只；真实快照 {membership['source_snapshot_days']} 日，{membership['imputed_days']} 个缺日只沿用此前快照，最大陈旧 {membership['max_staleness_calendar_days']} 个自然日。
- 数据健康：`passed={health['passed']}`；真实快照覆盖率 {pct(health['source_snapshot_coverage'])}；正式执行期必需原始行情 {health['formal_execution_instruments']} 只，缺失 {len(health['missing_raw'])}。
- 涨跌停跨源核验：{limits['rows']} 行、{limits['instruments']} 只重叠股票，不一致 {limits['mismatch_rows']} 行；最大绝对差 {limits['up_abs_diff_max']:.2f} 元。
- 公司行动外连接：{actions['events']} 个事件、{actions['matched_events']} 个匹配事件；裁决前候选单边 {actions['raw_missing_in_candidate']}、参考单边 {actions['raw_missing_in_reference']}，裁决后未决缺失均为 {actions['missing_in_candidate']}／{actions['missing_in_reference']}；现金／送转／支付日差异 {actions['cash_mismatch_over_0_011']}／{actions['stock_mismatch']}／{actions['pay_date_mismatch']}。
- 独立复原：参考缺失但由交易所参考价证实 {actions['reference_missing_exchange_verified']} 件，候选缺失后按参考价补记 {actions['exchange_reference_imputed']} 件，官方实施公告合并 {actions['official_implementation_combined']} 件。证据与审计分别见 `data/p27_universe_expansion/meta/action_official_evidence.json`、`action_official_evidence_audit.json` 和 `action_cross_validation_detail.parquet`。
- 执行强制数据闸：`passed={execution_gates['passed']}`；评分覆盖 {execution_gates['score_instruments']} 只股票，公司行动缓存缺失 {len(execution_gates['missing_action_files'])} 只。缺文件或任何跨源冲突都会在执行前抛错。
- 中证 1000：`rejected_gate0_data_health`，全历史 PIT 成分不足，**未训练、未看收益**。

## 分年可执行结果

{yearly_table({'p27_csi500_alstm158': verdicts['p27_csi500_alstm158']})}

## 容量与可成交性

{capacity_table(verdicts)}

成交额占比是事后诊断，不回填成交通知，也不冒充市场冲击模型。三档资金共享同一信号，不作为挑选参数。

## 终审闸门

{gate_lines({'p27_csi500_alstm158': verdicts['p27_csi500_alstm158']})}

事实源：`data/p27_universe_expansion/metrics/final_verdict.json`、`data/p27_universe_expansion/capacity/summary.json`。
"""
    p30 = f"""# P30 可执行成本感知目标终审报告

生成日期：{pd.Timestamp.now().date().isoformat()}。本报告由 `p27_p30_report.py` 从执行产物生成。

## 结论

{comparison}
{common}
{p30_diagnostics}

## 分年可执行结果

{yearly_table({'p30_open_gross': verdicts['p30_open_gross'], 'p30_open_net_masked': verdicts['p30_open_net_masked']})}

## 逐 fold 标签预测诊断

{p30_rank_ic_table()}

这些 RankIC／IC 只用于判断标签学习目标是否实现，不替代组合净收益，也不作为晋升证据。

## 终审闸门

{gate_lines({'p30_open_gross': verdicts['p30_open_gross'], 'p30_open_net_masked': verdicts['p30_open_net_masked']})}

## 解释边界

- `open_gross` 只改变训练目标为 T+1 开盘到 T+2 开盘毛收益；执行仍完整扣费。
- `open_net_masked` 额外扣除固定参考成本并剔除不可买入样本；标签 RankIC 不是晋升依据。
- 两个候选的年度聚类检验均为与 P25 同日超额差的配对结果；不能和 P27 的自身超额检验混读。

事实源：`data/p30_execution_aware/metrics/final_p30_open_gross.json`、`data/p30_execution_aware/metrics/final_p30_open_net_masked.json`。
"""
    (WS / "p27_universe_expansion_report.md").write_text(p27, encoding="utf-8")
    (WS / "p30_execution_aware_report.md").write_text(p30, encoding="utf-8")
    (WS / "data" / "p27_p30_comparison.md").write_text(comparison + common, encoding="utf-8")
    print(comparison)


if __name__ == "__main__":
    generate()
