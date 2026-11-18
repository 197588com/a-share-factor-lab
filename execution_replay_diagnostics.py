# -*- coding: utf-8 -*-
"""P25 执行损耗算术分解；事件计数不冒充收益贡献。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from execution_replay import OUT, PARAMS


def cost_decomposition(daily):
    prev = daily["nav"].shift(1).fillna(PARAMS["capital"])
    delta = float(daily["excess"].sub(daily["baseline_excess"]).mean() * 244)
    fee = float((daily["fee"] / prev).mean() * 244)
    slip = float((daily["slippage"] / prev).mean() * 244)
    baseline = float(daily["baseline_cost"].mean() * 244)
    incremental = fee + slip - baseline
    return {
        "days": int(len(daily)),
        "annual_excess_delta": delta,
        "annual_fee_drag": fee,
        "annual_slippage_drag": slip,
        "annual_baseline_cost_drag": baseline,
        "annual_incremental_direct_friction": incremental,
        "annual_residual_allocation_price_opportunity": delta - incremental,
        "mean_cash_weight": float((daily["cash"] / daily["nav"]).mean()),
        "mean_positions": float(daily["n_pos"].mean()),
        "mean_one_way_turnover": float(daily["turnover"].mean()),
    }


def order_decomposition(orders):
    buys = orders[orders["side"].eq("BUY")]
    sells = orders[orders["side"].eq("SELL")]
    skipped = buys[buys["status"].eq("SKIPPED")].copy()
    if not skipped.empty:
        one_lot = skipped["px"] * PARAMS["lot"]
        fee = np.maximum(PARAMS["comm_min"], one_lot * PARAMS["comm"])
        fee += np.where(skipped["code"].str.startswith("SH"), one_lot * PARAMS["transfer"], 0)
        skipped["skip_cash_insufficient"] = skipped["cash_after"] < one_lot + fee
    blocked = buys[buys["status"].eq("BLOCKED")]
    filled = orders[orders["status"].eq("FILLED")].copy()
    commission = float((filled["value"] * PARAMS["comm"]).clip(lower=PARAMS["comm_min"]).sum())
    stamp = float((filled.loc[filled["side"].eq("SELL"), "value"] * PARAMS["stamp"]).sum())
    transfer = float((filled.loc[filled["code"].str.startswith("SH"), "value"] * PARAMS["transfer"]).sum())
    return {
        "planned_buys": int(len(buys)), "filled_buys": int(buys["status"].eq("FILLED").sum()),
        "blocked_buys": int(len(blocked)), "skipped_buys": int(len(skipped)),
        "unfilled_buy_ratio": float(1 - buys["status"].eq("FILLED").mean()) if len(buys) else 0.0,
        "blocked_gap_over_1_5pct": int(blocked["note"].str.contains("跳空", na=False).sum()),
        "blocked_open_limit_up": int(blocked["note"].str.contains("涨停", na=False).sum()),
        "skipped_cash_insufficient": int(skipped.get("skip_cash_insufficient", pd.Series(dtype=bool)).sum()),
        "skipped_target_allocation_too_small": int((~skipped.get("skip_cash_insufficient", pd.Series(dtype=bool))).sum()),
        "skipped_open_price_median": float(skipped["raw_open"].median()) if len(skipped) else None,
        "skipped_open_price_p90": float(skipped["raw_open"].quantile(.9)) if len(skipped) else None,
        "filled_sells": int(sells["status"].eq("FILLED").sum()),
        "blocked_sells_open_limit_down": int(sells["note"].str.contains("跌停", na=False).sum()),
        "held_no_quote_events": int(orders["status"].eq("HELD").sum()),
        "fee_commission_reconstructed": commission, "fee_stamp_reconstructed": stamp,
        "fee_transfer_reconstructed": transfer,
        "fee_total_reconstructed": commission + stamp + transfer,
        "slippage_buy": float(((filled.loc[filled["side"].eq("BUY"), "px"] - filled.loc[filled["side"].eq("BUY"), "raw_open"])
                               * filled.loc[filled["side"].eq("BUY"), "qty"]).sum()),
        "slippage_sell": float(((filled.loc[filled["side"].eq("SELL"), "raw_open"] - filled.loc[filled["side"].eq("SELL"), "px"])
                                * filled.loc[filled["side"].eq("SELL"), "qty"]).sum()),
    }


def main():
    daily = pd.read_parquet(OUT / "daily_all.parquet")
    orders = pd.read_parquet(OUT / "orders_all.parquet")
    folds = {}
    for fold in sorted(daily["fold"].unique()):
        d = daily[daily["fold"].eq(fold)]
        o = orders[orders["fold"].eq(fold)]
        folds[str(fold)] = {"cost": cost_decomposition(d), "orders": order_decomposition(o)}
    overall_cost = cost_decomposition(daily)
    # 每 fold 是独立 100 万账户，总体直接损耗按日数加权，不跨 fold 错接 NAV。
    weights = {f: folds[f]["cost"]["days"] / len(daily) for f in folds}
    for key in ["annual_fee_drag", "annual_slippage_drag", "annual_baseline_cost_drag",
                "annual_incremental_direct_friction", "annual_residual_allocation_price_opportunity"]:
        overall_cost[key] = float(sum(folds[f]["cost"][key] * weights[f] for f in folds))
    actions = pd.concat([pd.read_parquet(OUT / f"corporate_actions_fold{f}.parquet").assign(fold=f)
                         for f in sorted(daily["fold"].unique())], ignore_index=True)
    imputed = actions[actions["cash_source"].eq("exchange_reference_imputed")]
    result = {
        "method_note": "直接摩擦为日度费用/滑点除以上日 NAV 的算术拖曳；残差同时包含成交组合差异、整手/现金滞留、跳空弃买和原始价路径，不是严格反事实归因。受阻事件不计作确定收益损失。",
        "overall": {"cost": overall_cost, "orders": order_decomposition(orders)},
        "folds": folds,
        "corporate_actions_on_holdings": {
            "events": int(len(actions)), "imputed_events": int(len(imputed)),
            "imputed_cash_amount": float(imputed["cash_amount"].sum()),
            "imputed_detail": imputed.to_dict("records"),
            "stock_reference_mismatch_events": int((actions["stk_div"].gt(0) & actions["cash_reconciliation_gap"].abs().gt(.011)).sum())
                if "stk_div" in actions else 0,
        },
    }
    (OUT / "execution_cost_decomposition.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame({"fold": f, **v["cost"], **v["orders"]} for f, v in {**folds, "overall": result["overall"]}.items()).to_csv(
        OUT / "execution_cost_decomposition.csv", index=False)
    actions.to_parquet(OUT / "corporate_actions_all.parquet", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
