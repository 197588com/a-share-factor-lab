# -*- coding: utf-8 -*-
"""P25：folds 8–11 生产 Top50/Drop5 的原始价可执行历史回灌。

独立审计，不修改 live_sim.py、生产 DuckDB、模型评分或既有 DSR。
正式窗口最晚 2026-06-30，holdout_guard 硬拦截；参数见 p25_execution_replay.md。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from holdout_guard import assert_window_allowed

WS = Path(__file__).resolve().parent
OUT = WS / "data" / "backtest" / "execution_replay"
RAW = OUT / "raw"
DAILY_BASE = WS / "data" / "backtest" / "daily"
SCORES = WS / "data" / "scores"
PARAMS = dict(capital=1_000_000.0, topk=50, n_drop=5, risk_degree=0.95,
              slip=0.001, comm=0.00025, comm_min=5.0, stamp=0.0005,
              transfer=0.00001, lot=100, gap_max=0.015, n_trials=9,
              formal_end="2026-06-30")


def fee_of(side, code, value):
    fee = max(PARAMS["comm_min"], value * PARAMS["comm"])
    if side == "SELL":
        fee += value * PARAMS["stamp"]
    if code.startswith("SH"):
        fee += value * PARAMS["transfer"]
    return float(fee)


def topk_plan(scores: pd.Series, positions: dict, topk=50, n_drop=5):
    """逐行复刻 Qlib TopkDropout 的候选、合并排序、卖出和买入数量。"""
    scores = scores.astype(float)
    current = pd.Index(list(positions.keys()))
    last = scores.reindex(current).sort_values(ascending=False).index
    need = max(n_drop + topk - len(last), 0)
    today = scores[~scores.index.isin(last)].sort_values(ascending=False).index[:need]
    comb = scores.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
    tail = comb[-min(n_drop, len(comb)):] if len(comb) else comb
    sell = last[last.isin(tail)].tolist()
    buy_n = max(len(sell) + topk - len(last), 0)
    buy = today[:buy_n].tolist()
    return sell, buy


class ReplayAccount:
    def __init__(self, capital=PARAMS["capital"]):
        self.cash = float(capital)
        self.fee_cum = 0.0
        self.pos = {}
        self.receivables = []
        self.orders = []
        self.action_log = []
        self.corporate_actions = 0
        self.dividend_receivable_new = 0.0

    def receivable_total(self):
        return float(sum(x["amount"] for x in self.receivables))

    def equity(self):
        return float(self.cash + sum(p["qty"] * p["last_px"] for p in self.pos.values())
                     + self.receivable_total())

    def _log(self, d, sd, code, side, qty, px, value, fee, status, note, bar=None):
        self.orders.append({
            "exec_date": str(pd.Timestamp(d).date()),
            "signal_date": str(pd.Timestamp(sd).date()) if sd is not None else None,
            "code": code, "side": side, "qty": float(qty), "raw_open": float(bar["open"]) if bar else np.nan,
            "px": float(px), "value": float(value), "fee": float(fee), "status": status, "note": note,
            "cash_after": float(self.cash), "limit_source": bar.get("limit_source", "missing") if bar else "missing",
        })

    def process_payables(self, d):
        d = pd.Timestamp(d)
        keep = []
        for item in self.receivables:
            if pd.Timestamp(item["pay_date"]) <= d:
                self.cash += item["amount"]
            else:
                keep.append(item)
        self.receivables = keep

    def apply_actions(self, d, actions_by_code):
        """除权日开盘前处理既有持仓；分红先计应收，支付日才转可用现金。"""
        d = pd.Timestamp(d)
        self.corporate_actions = 0
        self.corporate_action_imputed = 0
        self.corporate_action_stock_mismatch = 0
        self.dividend_receivable_new = 0.0
        for code, action in actions_by_code.items():
            if code not in self.pos or action is None or action.empty:
                continue
            rows = action[pd.to_datetime(action["ex_date"], errors="coerce").eq(d)]
            for _, r in rows.iterrows():
                p = self.pos[code]
                old_qty = float(p["qty"])
                cash_div = float(r.get("cash_div", 0.0)) if pd.notna(r.get("cash_div")) else 0.0
                # Tushare 实施记录中 stk_div 是每股送转总比例；stk_bo_rate/stk_co_rate
                # 是其拆分项，不能再相加或相减（已用 pre_close 与复权因子交叉核实）。
                stk_div = float(r.get("stk_div", 0.0)) if pd.notna(r.get("stk_div")) else 0.0
                multiplier = 1.0 + stk_div
                if multiplier <= 0 or cash_div < 0:
                    raise ValueError(f"非法公司行动 {code} {d.date()} multiplier={multiplier} cash={cash_div}")
                if r.get("cash_source") == "exchange_reference_imputed":
                    self.corporate_action_imputed += 1
                if stk_div and abs(float(r.get("cash_reconciliation_gap", 0.0))) > 0.011:
                    self.corporate_action_stock_mismatch += 1
                # 无行情停牌时也把最后价调整为除权参考价，NAV 不出现伪亏损。
                p["last_px"] = max((float(p["last_px"]) - cash_div) / multiplier, 0.0)
                p["qty"] = old_qty * multiplier
                amount = old_qty * cash_div
                self.action_log.append({
                    "ex_date": str(d.date()), "code": code, "old_qty": old_qty,
                    "new_qty": float(p["qty"]), "cash_div_per_old_share": cash_div,
                    "cash_amount": float(amount), "cash_source": r.get("cash_source", "legacy_unknown"),
                    "raw_cash_sum": float(r.get("raw_cash_sum", np.nan)),
                    "reference_implied_cash": float(r.get("reference_implied_cash", np.nan)),
                    "cash_reconciliation_gap": float(r.get("cash_reconciliation_gap", np.nan)),
                    "pay_date": str(pd.Timestamp(r["pay_date"]).date()) if pd.notna(r.get("pay_date")) else str(d.date()),
                })
                if amount:
                    pay_date = pd.Timestamp(r["pay_date"]) if pd.notna(r.get("pay_date")) else d
                    item = {"code": code, "amount": float(amount), "ex_date": str(d.date()),
                            "pay_date": str(pay_date.date())}
                    if pay_date <= d:
                        self.cash += amount
                    else:
                        self.receivables.append(item)
                    self.dividend_receivable_new += amount
                self.corporate_actions += 1

    def sell(self, d, sd, code, bar):
        p = self.pos[code]
        opx = float(bar["open"])
        px = opx * (1 - PARAMS["slip"])
        if pd.notna(bar.get("down_limit")) and opx <= float(bar["down_limit"]) + 0.005:
            self._log(d, sd, code, "SELL", 0, px, 0, 0, "BLOCKED", "开盘跌停卖不出", bar)
            return False
        qty = float(p["qty"])
        value = qty * px
        fee = fee_of("SELL", code, value)
        self.cash += value - fee
        self.fee_cum += fee
        self._log(d, sd, code, "SELL", qty, px, value, fee, "FILLED", "", bar)
        del self.pos[code]
        return True

    def buy(self, d, sd, code, target_value, bar):
        opx = float(bar["open"])
        px = opx * (1 + PARAMS["slip"])
        pre = float(bar["pre_close"])
        if opx > pre * (1 + PARAMS["gap_max"]):
            self._log(d, sd, code, "BUY", 0, px, 0, 0, "BLOCKED", "开盘跳空>1.5%，放弃且不顺延", bar)
            return False
        if pd.notna(bar.get("up_limit")) and opx >= float(bar["up_limit"]) - 0.005:
            self._log(d, sd, code, "BUY", 0, px, 0, 0, "BLOCKED", "开盘涨停买不进", bar)
            return False
        qty = int(min(target_value, self.cash) / px / PARAMS["lot"]) * PARAMS["lot"]
        if qty <= 0:
            self._log(d, sd, code, "BUY", 0, px, 0, 0, "SKIPPED", "现金不足一手", bar)
            return False
        value = qty * px
        fee = fee_of("BUY", code, value)
        while value + fee > self.cash + 1e-9 and qty > 0:
            qty -= PARAMS["lot"]
            value = qty * px
            fee = fee_of("BUY", code, value) if qty else 0.0
        if qty <= 0:
            self._log(d, sd, code, "BUY", 0, px, 0, 0, "SKIPPED", "含费现金不足一手", bar)
            return False
        self.cash -= value + fee
        self.fee_cum += fee
        self.pos[code] = {"qty": float(qty), "last_px": px, "entry": str(pd.Timestamp(d).date())}
        self._log(d, sd, code, "BUY", qty, px, value, fee, "FILLED", "", bar)
        return True


class RawStore:
    def __init__(self, instruments):
        names_path = WS / "data" / "live" / "name_map.csv"
        self.names = {}
        if names_path.exists():
            names = pd.read_csv(names_path, dtype=str)
            self.names = dict(zip(names["ts_code"].str[:6], names["name"]))
        self.frames = {}
        self.actions = {}
        for code in sorted(instruments):
            p = RAW / f"{code.lower()}.parquet"
            if not p.exists():
                raise FileNotFoundError(f"缺原始行情：{p}；先运行 python execution_data.py")
            df = pd.read_parquet(p)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                if "pre_close" not in df:
                    df["pre_close"] = df["close"].shift(1)
            self.frames[code] = df
            ap = RAW / f"{code.lower()}_actions.parquet"
            action = pd.read_parquet(ap) if ap.exists() else pd.DataFrame()
            if not action.empty:
                action["ex_date"] = pd.to_datetime(action["ex_date"], errors="coerce")
                action["pay_date"] = pd.to_datetime(action["pay_date"], errors="coerce")
            self.actions[code] = action

    def row(self, code, d):
        df = self.frames.get(code)
        if df is None or df.empty or pd.Timestamp(d) not in df.index:
            return None
        row = df.loc[pd.Timestamp(d)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row

    def _fallback_limit(self, code, pre, up):
        number = code[2:]
        name = self.names.get(number, "")
        ratio = 0.05 if "ST" in name.upper() else (0.20 if number.startswith(("300", "301", "688")) else 0.10)
        return pre * (1 + ratio) if up else pre * (1 - ratio)

    def bar(self, code, d):
        row = self.row(code, d)
        if row is None or pd.isna(row.get("open")) or pd.isna(row.get("pre_close")):
            return None
        bar = row.to_dict()
        fallback = pd.isna(bar.get("up_limit")) or pd.isna(bar.get("down_limit"))
        if fallback:
            bar["up_limit"] = self._fallback_limit(code, float(bar["pre_close"]), True)
            bar["down_limit"] = self._fallback_limit(code, float(bar["pre_close"]), False)
        bar["limit_source"] = "fallback_board_name" if fallback else "tushare_stk_limit"
        return bar


def load_adjusted_features(instruments, start, end):
    """本地前复权 CSV 复刻 backtest_p4.mask_limit_up，不读取外部 qlib 库。"""
    frames = []
    for code in sorted(instruments):
        p = WS / "data" / "csv" / f"{code.lower()}.csv"
        if not p.exists() or p.stat().st_size == 0:
            continue
        df = pd.read_csv(p, usecols=["date", "open", "close"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        # D.features 原调用只取评分最小/最大日，首日 chg 与末日 next_open 因而为 NaN。
        df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date")
        df["chg"] = df["close"].pct_change()
        df["nxt_open"] = df["open"].shift(-1)
        df["ogap"] = df["nxt_open"] / df["close"] - 1
        df["instrument"] = code
        frames.append(df[["date", "instrument", "close", "chg", "ogap"]])
    if not frames:
        raise ValueError("前复权行情为空")
    return pd.concat(frames, ignore_index=True)


def production_mask(score, adjusted):
    df = score.merge(adjusted, on=["date", "instrument"], how="left", validate="many_to_one")
    code = df["instrument"].str[2:]
    date = df["date"]
    board = np.where(code.str.startswith("68"), 0.198,
                     np.where(code.str.startswith("30"), 0.198, 0.098))
    gem20 = code.str.startswith("30") & date.ge(pd.Timestamp("2020-08-24"))
    thr = np.where(gem20, 0.198, board)
    mask_t = df["chg"].ge(thr).fillna(False)
    close_t = df["close"].to_numpy(dtype=float)
    ogap_thr = thr - 0.005 / np.clip(close_t, 1e-9, None)
    mask_open = (~mask_t) & df["ogap"].to_numpy(dtype=float).__ge__(ogap_thr)
    mask_open = pd.Series(mask_open, index=df.index).fillna(False)
    events = pd.concat([
        df.loc[mask_t, ["date", "instrument", "score"]].assign(mask_reason="T_close_limit"),
        df.loc[mask_open, ["date", "instrument", "score"]].assign(mask_reason="T1_open_limit"),
    ], ignore_index=True)
    kept = df.loc[~(mask_t | mask_open), ["date", "instrument", "score"]].copy()
    return kept, events


def _metric_row(daily, prefix=""):
    ret = daily["strategy_return"].astype(float)
    excess = daily["excess"].astype(float)
    nav = daily["nav"].astype(float)
    # 合并净值仅用于跨 fold 链接收益/回撤；现金权重必须除以各 fold 自己的账户 NAV。
    cash_nav = daily["account_nav"].astype(float) if "account_nav" in daily else nav
    return {
        "days": int(len(daily)),
        "annual_net_excess": float(excess.mean() * 244),
        "net_ir": float(excess.mean() / (excess.std(ddof=1) + 1e-12) * np.sqrt(244)),
        "mdd": float((nav / nav.cummax() - 1).min()),
        "total_net_return": float(nav.iloc[-1] / PARAMS["capital"] - 1),
        "benchmark_return": float((1 + daily["bench"]).prod() - 1),
        "mean_one_way_turnover": float(daily["turnover"].mean()),
        "fees_sum": float(daily["fee"].sum()),
        "slippage_sum": float(daily["slippage"].sum()),
        "mean_positions": float(daily["n_pos"].mean()),
        "mean_cash_weight": float((daily["cash"] / cash_nav).mean()),
    }


def _baseline_metric(daily):
    ret = daily["baseline_return"] - daily["baseline_cost"]
    excess = daily["baseline_excess"]
    nav = (1 + ret).cumprod()
    return {
        "days": int(len(daily)), "annual_net_excess": float(excess.mean() * 244),
        "net_ir": float(excess.mean() / (excess.std(ddof=1) + 1e-12) * np.sqrt(244)),
        "mdd": float((nav / nav.cummax() - 1).min()),
        "total_net_return": float(nav.iloc[-1] - 1),
        "benchmark_return": float((1 + daily["bench"]).prod() - 1),
        "fees_sum": float(daily["baseline_cost"].sum() * PARAMS["capital"]),
    }


def run_fold(fold, store):
    baseline = pd.read_parquet(DAILY_BASE / f"alstm_fold{fold}_oc0.0005_cc0.0015.parquet")
    baseline["datetime"] = pd.to_datetime(baseline["datetime"])
    baseline = baseline[baseline["datetime"] <= PARAMS["formal_end"]].sort_values("datetime").reset_index(drop=True)
    assert_window_allowed(baseline["datetime"].min(), baseline["datetime"].max(), purpose="candidate")
    raw_score = pd.read_parquet(SCORES / f"alstm_fold{fold}.parquet")
    raw_score["date"] = pd.to_datetime(raw_score["date"])
    raw_score["instrument"] = raw_score["instrument"].str.upper()
    raw_score = raw_score[raw_score["date"].isin(set(baseline["datetime"]))]
    # 使用随审计归档的 Qlib 快照原算法输出，避免 CSV 复刻在临界格上产生集合差。
    kept_path = OUT / f"mask_kept_qlib_fold{fold}.parquet"
    events_path = OUT / f"mask_events_qlib_fold{fold}.parquet"
    if not kept_path.exists() or not events_path.exists():
        raise FileNotFoundError(f"缺 fold{fold} Qlib 屏蔽快照；先运行 validate_mask_qlib.py --fold {fold}")
    score = pd.read_parquet(kept_path)
    events = pd.read_parquet(events_path)
    score["date"] = pd.to_datetime(score["date"])
    events["date"] = pd.to_datetime(events["date"])
    check = score.merge(raw_score, on=["date", "instrument"], how="left",
                        suffixes=("_masked", "_raw"), validate="one_to_one")
    if len(check) != len(score) or check["score_raw"].isna().any() or not np.allclose(
            check["score_masked"], check["score_raw"], rtol=0, atol=1e-12):
        raise AssertionError(f"fold{fold} Qlib 屏蔽键或评分与原始评分不一致")
    if score["score"].isna().any():
        raise AssertionError(f"fold{fold} 屏蔽后仍有空评分")
    scores_by_date = {d: g.set_index("instrument")["score"] for d, g in score.groupby("date")}
    acct = ReplayAccount()
    rows = []
    calendar = baseline["datetime"].tolist()
    for i, d in enumerate(calendar):
        acct.process_payables(d)
        acct.apply_actions(d, store.actions)
        order_start = len(acct.orders)
        fee_before = acct.fee_cum
        if i > 0:
            sd = calendar[i - 1]
            signal = scores_by_date.get(sd)
            if signal is not None:
                sells, buys = topk_plan(signal, acct.pos, PARAMS["topk"], PARAMS["n_drop"])
                for code in sells:
                    if code not in acct.pos:
                        continue
                    bar = store.bar(code, d)
                    if bar is None:
                        acct._log(d, sd, code, "SELL", 0, 0, 0, 0, "HELD", "停牌或无有效开盘，继续持有")
                    else:
                        acct.sell(d, sd, code, bar)
                target = acct.cash * PARAMS["risk_degree"] / len(buys) if buys else 0.0
                for code in buys:
                    bar = store.bar(code, d)
                    if bar is None:
                        acct._log(d, sd, code, "BUY", 0, 0, 0, 0, "HELD", "停牌或无有效开盘，无法买入")
                    else:
                        acct.buy(d, sd, code, target, bar)
        day_orders = acct.orders[order_start:]
        filled = [x for x in day_orders if x["status"] == "FILLED"]
        gross = sum(x["value"] for x in filled)
        slippage = sum((x["px"] - x["raw_open"]) * x["qty"] if x["side"] == "BUY"
                       else (x["raw_open"] - x["px"]) * x["qty"] for x in filled)
        for code, p in acct.pos.items():
            row = store.row(code, d)
            if row is not None and pd.notna(row.get("close")):
                p["last_px"] = float(row["close"])
        nav = acct.equity()
        prev_nav = rows[-1]["nav"] if rows else PARAMS["capital"]
        mkt_value = float(sum(p["qty"] * p["last_px"] for p in acct.pos.values()))
        b = baseline.iloc[i]
        strategy_return = nav / prev_nav - 1
        rows.append({
            "datetime": d, "fold": fold, "cash": acct.cash, "mkt_value": mkt_value,
            "receivables": acct.receivable_total(), "nav": nav, "strategy_return": strategy_return,
            "bench": float(b["bench"]), "excess": strategy_return - float(b["bench"]),
            "baseline_return": float(b["return"]), "baseline_cost": float(b["cost"]),
            "baseline_excess": float(b["excess"]), "n_pos": len(acct.pos),
            "turnover": gross / (2 * prev_nav) if prev_nav else 0.0,
            "fee": acct.fee_cum - fee_before, "fee_cum": acct.fee_cum,
            "slippage": slippage, "corporate_actions": acct.corporate_actions,
            "corporate_action_imputed": acct.corporate_action_imputed,
            "corporate_action_stock_mismatch": acct.corporate_action_stock_mismatch,
            "dividend_receivable_new": acct.dividend_receivable_new,
            "blocked": sum(x["status"] == "BLOCKED" for x in day_orders),
            "held": sum(x["status"] == "HELD" for x in day_orders),
            "skipped": sum(x["status"] == "SKIPPED" for x in day_orders),
        })
        if not np.isfinite(nav) or nav <= 0 or acct.cash < -0.01:
            raise AssertionError(f"账户健康失败 fold={fold} date={d.date()} nav={nav} cash={acct.cash}")
    daily = pd.DataFrame(rows)
    orders = pd.DataFrame(acct.orders)
    action_log = pd.DataFrame(acct.action_log)
    return daily, orders, events, action_log


def _dsr(excess):
    sys.path.insert(0, str(WS / "qs-skills" / "skill-backtest-overfit" / "scripts"))
    from deflated_sharpe import deflated_sharpe_ratio
    r = deflated_sharpe_ratio(np.asarray(excess, dtype=float), n_trials=PARAMS["n_trials"])
    return {"series": "P25 raw-price executable daily excess return; folds 8-11; formal window only",
            "n_obs": int(len(excess)), "n_trials": PARAMS["n_trials"],
            "observed_sr_per_period": float(np.mean(excess) / np.std(excess, ddof=1)),
            "observed_sr_ann": float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(244)),
            **r.to_dict()}


def validate_corporate_actions(store):
    """实施明细、除权参考昨收和复权因子三分核验；不把不同收益定义混为一谈。"""
    rows = []
    for code, actions in store.actions.items():
        if actions is None or actions.empty:
            continue
        frame = store.frames.get(code)
        if frame is None or frame.empty:
            continue
        for _, r in actions.iterrows():
            d = pd.Timestamp(r["ex_date"]) if pd.notna(r["ex_date"]) else pd.NaT
            if pd.isna(d) or d not in frame.index:
                continue
            i = frame.index.get_loc(d)
            if not isinstance(i, (int, np.integer)) or i <= 0:
                continue
            ex, prev = frame.iloc[i], frame.iloc[i - 1]
            cash = float(r.get("cash_div", 0.0)) if pd.notna(r.get("cash_div")) else 0.0
            stock = float(r.get("stk_div", 0.0)) if pd.notna(r.get("stk_div")) else 0.0
            mult = 1 + stock
            prev_close, pre_ex, close_ex = float(prev["close"]), float(ex["pre_close"]), float(ex["close"])
            implied = prev_close - pre_ex * mult
            economic = (close_ex * mult + cash) / prev_close - 1
            reference = close_ex / pre_ex - 1
            adjusted = np.nan
            if "adj_factor" in frame and pd.notna(ex["adj_factor"]) and pd.notna(prev["adj_factor"]):
                adjusted = close_ex * float(ex["adj_factor"]) / (prev_close * float(prev["adj_factor"])) - 1
            rows.append({"code": code, "ex_date": d, "prev_close": prev_close,
                         "pre_close_ex": pre_ex, "close_ex": close_ex,
                         "cash_div": cash, "stk_div": stock, "implied_cash": implied,
                         "cash_gap": cash - implied, "economic_return": economic,
                         "reference_return": reference, "adj_factor_return": adjusted,
                         "abs_economic_minus_adj": abs(economic - adjusted) if pd.notna(adjusted) else np.nan,
                         "abs_reference_minus_adj": abs(reference - adjusted) if pd.notna(adjusted) else np.nan,
                         "cash_source": r.get("cash_source", "legacy_unknown"),
                         "raw_cash_sum": r.get("raw_cash_sum", np.nan), "n_raw_rows": r.get("n_raw_rows", np.nan),
                         "n_plans": r.get("n_plans", np.nan), "n_pay_dates": r.get("n_pay_dates", np.nan),
                         "pay_date": r.get("pay_date")})
    if not rows:
        return {"events_checked": 0}
    detail = pd.DataFrame(rows).sort_values(["ex_date", "code"])
    detail.to_csv(OUT / "corporate_action_validation.csv", index=False)

    def error_stats(s):
        s = s.dropna()
        return {"median_abs_error": float(s.median()), "p99_abs_error": float(s.quantile(.99)),
                "max_abs_error": float(s.max()), "over_10bp": int((s > .001).sum())}

    gap = detail["cash_gap"].abs()
    summary = {
        "events_checked": int(len(detail)),
        "cash_sources": {str(k): int(v) for k, v in detail["cash_source"].value_counts().items()},
        "exchange_reference_vs_adj_factor": error_stats(detail["abs_reference_minus_adj"]),
        "economic_total_return_vs_adj_factor_descriptive": error_stats(detail["abs_economic_minus_adj"]),
        "cash_vs_exchange_reference": {
            "median_abs_cny_per_share": float(gap.median()), "p99_abs_cny_per_share": float(gap.quantile(.99)),
            "max_abs_cny_per_share": float(gap.max()), "over_0_011_cny": int((gap > .011).sum()),
            "stock_plan_over_0_011_cny": int(((gap > .011) & detail["stk_div"].gt(0)).sum()),
        },
        "multi_pay_date_events": int((pd.to_numeric(detail.get("n_pay_dates"), errors="coerce").fillna(0) > 1).sum())
            if "n_pay_dates" in detail else 0,
        "worst_cash_gaps": detail.assign(_gap=gap).nlargest(10, "_gap").drop(columns="_gap").to_dict("records"),
    }
    (OUT / "corporate_action_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def _plots(all_daily):
    import matplotlib.pyplot as plt
    d = all_daily.sort_values("datetime")
    nav = (1 + d["strategy_return"]).cumprod()
    bench = (1 + d["bench"]).cumprod()
    baseline = (1 + d["baseline_return"] - d["baseline_cost"]).cumprod()
    dd = nav / nav.cummax() - 1
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.3, 1.2, 1.5]})
    axes[0].plot(d["datetime"], nav, label="P25 executable", lw=1.5)
    axes[0].plot(d["datetime"], baseline, label="Qlib baseline", lw=1.1, alpha=0.85)
    axes[0].plot(d["datetime"], bench, label="CSI300", lw=1.0, alpha=0.75)
    axes[0].set_title("P25 execution replay: net value")
    axes[0].legend(); axes[0].grid(alpha=0.2)
    axes[1].plot(d["datetime"], (1 + d["excess"]).cumprod(), label="Executable excess compound")
    axes[1].plot(d["datetime"], (1 + d["baseline_excess"]).cumprod(), label="Baseline excess compound", alpha=0.8)
    axes[1].legend(); axes[1].grid(alpha=0.2)
    axes[2].fill_between(d["datetime"], dd, 0, color="firebrick", alpha=0.35)
    axes[2].set_title("Executable drawdown"); axes[2].grid(alpha=0.2)
    axes[3].plot(d["datetime"], d["turnover"].rolling(20).mean(), label="20d one-way turnover")
    axes[3].plot(d["datetime"], d["fee"] / d["nav"].shift(1) * 1e4, label="Daily fee (bp)", alpha=0.55)
    axes[3].legend(); axes[3].grid(alpha=0.2)
    fig.tight_layout()
    path = OUT / "execution_replay_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="8-11")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    if "-" in args.folds:
        lo, hi = map(int, args.folds.split("-")); folds = list(range(lo, hi + 1))
    else:
        folds = [int(x) for x in args.folds.split(",")]
    if not (WS / "p25_execution_replay.md").exists():
        raise FileNotFoundError("P25 预注册缺失")
    OUT.mkdir(parents=True, exist_ok=True)
    all_instruments = set()
    for fold in folds:
        s = pd.read_parquet(SCORES / f"alstm_fold{fold}.parquet", columns=["instrument"])
        all_instruments.update(s["instrument"].str.upper())
    store = RawStore(all_instruments)
    summaries, daily_all, order_counts = [], [], {}
    health = {"folds": {}, "checks": {}}
    for fold in folds:
        daily, orders, events, action_log = run_fold(fold, store)
        daily.to_parquet(OUT / f"daily_fold{fold}.parquet", index=False)
        orders.to_parquet(OUT / f"orders_fold{fold}.parquet", index=False)
        events.to_parquet(OUT / f"mask_events_fold{fold}.parquet", index=False)
        action_log.to_parquet(OUT / f"corporate_actions_fold{fold}.parquet", index=False)
        m = _metric_row(daily)
        b = _baseline_metric(daily)
        summaries.append({"fold": fold, "year": int(daily["datetime"].dt.year.iloc[0]),
                          **{f"replay_{k}": v for k, v in m.items()},
                          **{f"baseline_{k}": v for k, v in b.items()},
                          "delta_annual_net_excess": m["annual_net_excess"] - b["annual_net_excess"],
                          "delta_ir": m["net_ir"] - b["net_ir"]})
        daily_all.append(daily)
        order_counts[str(fold)] = orders.groupby(["side", "status"]).size().to_dict() if not orders.empty else {}
        mask_diff_path = OUT / f"mask_diff_fold{fold}.csv"
        mask_diff = pd.read_csv(mask_diff_path) if mask_diff_path.exists() else pd.DataFrame()
        health["folds"][str(fold)] = {
            "dates_match_baseline": bool(daily["datetime"].equals(pd.read_parquet(
                DAILY_BASE / f"alstm_fold{fold}_oc0.0005_cc0.0015.parquet")["datetime"].pipe(pd.to_datetime).loc[lambda x: x <= PARAMS["formal_end"]].reset_index(drop=True))),
            "max_positions": int(daily["n_pos"].max()), "median_positions": float(daily["n_pos"].median()),
            "mean_one_way_turnover": float(daily["turnover"].mean()),
            "mask_source": "archived_qlib_snapshot_original_algorithm",
            "mask_key_diff_vs_csv_replica": int(len(mask_diff)),
            "mask_T_close": int((events["mask_reason"] == "T_close_limit").sum()) if not events.empty else 0,
            "mask_T1_open": int((events["mask_reason"] == "T1_open_limit").sum()) if not events.empty else 0,
        }
        print(f"fold{fold}: replay ann={m['annual_net_excess']*100:.2f}% IR={m['net_ir']:.3f} "
              f"MDD={m['mdd']*100:.1f}% | baseline ann={b['annual_net_excess']*100:.2f}% IR={b['net_ir']:.3f}", flush=True)
    daily = pd.concat(daily_all, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    daily.to_parquet(OUT / "daily_all.parquet", index=False)
    overall = _metric_row(daily.assign(account_nav=daily["nav"],
                                       nav=(1 + daily["strategy_return"]).cumprod() * PARAMS["capital"]))
    baseline_overall = _baseline_metric(daily)
    dsr = _dsr(daily["excess"].to_numpy())
    (OUT / "dsr_execution_replay.json").write_text(json.dumps(dsr, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"params": PARAMS, "formal_window": [str(daily["datetime"].min().date()), str(daily["datetime"].max().date())],
               "overall": overall, "baseline_overall": baseline_overall, "dsr": dsr,
               "folds": summaries, "order_counts": {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in order_counts.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(OUT / "fold_summary.csv", index=False)
    orders_all = pd.concat([pd.read_parquet(OUT / f"orders_fold{f}.parquet").assign(fold=f) for f in folds], ignore_index=True)
    orders_all.to_parquet(OUT / "orders_all.parquet", index=False)
    filled_buys = orders_all[(orders_all["side"] == "BUY") & (orders_all["status"] == "FILLED")]
    corporate_validation = validate_corporate_actions(store)
    health["corporate_action_independent_validation"] = corporate_validation
    health["checks"] = {
        "expected_843_days": bool(len(daily) == 843),
        "no_sealed_dates": bool(daily["datetime"].max() <= pd.Timestamp(PARAMS["formal_end"])),
        "no_negative_cash": bool((daily["cash"] >= -0.01).all()),
        "nav_identity": bool(np.allclose(daily["cash"] + daily["mkt_value"] + daily["receivables"], daily["nav"], atol=1e-6)),
        "all_filled_buys_round_lot": bool((filled_buys["qty"] % 100 == 0).all()),
        "no_future_signal": bool((pd.to_datetime(orders_all["signal_date"]) < pd.to_datetime(orders_all["exec_date"])).all()),
        "finite_returns": bool(np.isfinite(daily[["strategy_return", "bench", "excess"]].to_numpy()).all()),
        "turnover_plausible_0_to_30pct": bool(0 < daily["turnover"].mean() < 0.30),
        "all_fold_dates_match": bool(all(x["dates_match_baseline"] for x in health["folds"].values())),
        "all_folds_use_qlib_snapshot": bool(all(x["mask_source"] == "archived_qlib_snapshot_original_algorithm"
                                                for x in health["folds"].values())),
        "corporate_action_sources_known": bool("legacy_unknown" not in corporate_validation.get("cash_sources", {})),
    }
    health["passed"] = bool(all(health["checks"].values()))
    health["n_orders"] = int(len(orders_all))
    health["limit_fallback_orders"] = int((orders_all["limit_source"] == "fallback_board_name").sum())
    health["corporate_action_events_on_holdings"] = int(daily["corporate_actions"].sum())
    health["corporate_action_imputed_on_holdings"] = int(daily["corporate_action_imputed"].sum())
    health["corporate_action_stock_mismatch_on_holdings"] = int(daily["corporate_action_stock_mismatch"].sum())
    health["dividend_receivables_total"] = float(daily["dividend_receivable_new"].sum())
    health["position_summary"] = {"mean": float(daily["n_pos"].mean()), "median": float(daily["n_pos"].median()),
                                  "min": int(daily["n_pos"].min()), "max": int(daily["n_pos"].max())}
    (OUT / "health.json").write_text(json.dumps(health, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not health["passed"]:
        raise AssertionError(f"P25 健康检查失败：{json.dumps(health['checks'], ensure_ascii=False)}")
    plot = None if args.skip_plots else str(_plots(daily))
    print(json.dumps({"overall": overall, "baseline": baseline_overall, "dsr": dsr,
                      "health": health, "plot": plot}, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
