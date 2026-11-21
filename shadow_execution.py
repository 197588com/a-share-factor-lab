# -*- coding: utf-8 -*-
"""P27/P30 共用 P25 账户语义的 shadow 执行器；不修改生产账务。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import execution_replay as er
from holdout_guard import assert_window_allowed

WS = Path(__file__).resolve().parent


class P27RawStore(er.RawStore):
    def __init__(self, instruments):
        self.frames, self.actions, self.names = {}, {}, {}
        raw = WS / "data" / "p27_universe_expansion" / "raw_daily"
        actions = WS / "data" / "p27_universe_expansion" / "actions"
        for code in sorted(instruments):
            p = raw / f"{code.lower()}.parquet"
            if not p.exists():
                raise FileNotFoundError(f"缺 P27 原始行情 {p}")
            df = pd.read_parquet(p)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                if "pre_close" not in df and "preclose" in df:
                    df["pre_close"] = df["preclose"]
            self.frames[code] = df
            ap = actions / f"{code.lower()}.parquet"
            if not ap.exists():
                raise FileNotFoundError(f"缺 P27 公司行动缓存 {ap}；先运行 p27_actions.py")
            action = pd.read_parquet(ap)
            if not action.empty:
                action["ex_date"] = pd.to_datetime(action["ex_date"], errors="coerce")
                action["pay_date"] = pd.to_datetime(action["pay_date"], errors="coerce")
            self.actions[code] = action

    def bar(self, code, d):
        row = self.row(code, d)
        if row is None:
            return None
        if "tradestatus" in row and not pd.notna(row.get("tradestatus")):
            return None
        if "tradestatus" in row and int(row.get("tradestatus")) != 1:
            return None
        if not pd.notna(row.get("open")) or not pd.notna(row.get("pre_close")):
            return None
        if float(row["open"]) <= 0 or float(row["pre_close"]) <= 0:
            return None
        bar = super().bar(code, d)
        if bar is not None and pd.notna(row.get("limit_source", np.nan)):
            bar["limit_source"] = row["limit_source"]
        return bar


def adjusted_features(instruments, start, end, csv_dir):
    frames = []
    for code in sorted(instruments):
        p = Path(csv_dir) / f"{code.lower()}.csv"
        if not p.exists() or p.stat().st_size == 0:
            continue
        df = pd.read_csv(p, usecols=["date", "open", "close"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date")
        df["chg"] = df["close"].pct_change()
        df["ogap"] = df["open"].shift(-1) / df["close"] - 1
        df["instrument"] = code
        frames.append(df[["date", "instrument", "close", "chg", "ogap"]])
    if not frames:
        raise ValueError("前复权行情为空")
    return pd.concat(frames, ignore_index=True)


def index_baseline(index_instrument, csv_dir, start, end):
    d = pd.read_csv(Path(csv_dir) / f"{index_instrument.lower()}.csv", usecols=["date", "close"])
    d["datetime"] = pd.to_datetime(d["date"])
    d = d[(d["datetime"] >= start) & (d["datetime"] <= end)].sort_values("datetime")
    d["bench"] = d["close"].pct_change().fillna(0.0)
    d["return"] = d["bench"]
    d["cost"] = 0.0
    d["excess"] = 0.0
    return d[["datetime", "bench", "return", "cost", "excess"]].reset_index(drop=True)


def prepare_scores(score, baseline, csv_dir):
    score = score.copy()
    score["date"] = pd.to_datetime(score["date"])
    score["instrument"] = score["instrument"].str.upper()
    score = score[score["date"].isin(set(baseline["datetime"]))]
    adjusted = adjusted_features(score["instrument"].unique(), score["date"].min(), score["date"].max(), csv_dir)
    kept, events = er.production_mask(score[["date", "instrument", "score"]], adjusted)
    if kept.empty or kept["score"].isna().any():
        raise AssertionError("评分屏蔽后为空或仍有 NaN")
    return kept, events


def validate_score_frame(score):
    """评分键唯一且数值有限；坏评分在任何行情读取和账户撮合前失败。"""
    required = {"date", "instrument", "score"}
    if not required.issubset(score.columns):
        raise AssertionError(f"评分字段缺失：{required - set(score.columns)}")
    if score[["date", "instrument"]].isna().any().any():
        raise AssertionError("评分日期或股票代码为空")
    if score.duplicated(["date", "instrument"]).any():
        raise AssertionError("评分存在重复股票—日期键")
    values = pd.to_numeric(score["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("评分包含 NaN 或无穷值")


def prepare_scores_qlib(score, baseline, provider):
    """P30 与 P25 使用同一 Qlib 快照和原屏蔽算法，避免 CSV 临界格集合差。"""
    validate_score_frame(score)
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri=str(Path(provider).resolve()), region=REG_CN, kernels=1)
    score = score.copy()
    score["date"] = pd.to_datetime(score["date"])
    score["instrument"] = score["instrument"].str.upper()
    dates = pd.to_datetime(baseline["datetime"])
    score = score[score["date"].isin(set(dates))].copy()
    pred = score.set_index(["date", "instrument"])["score"]
    pred.index = pred.index.set_names(["datetime", "instrument"])
    insts = sorted(pred.index.get_level_values("instrument").unique())
    px = D.features(insts, ["$close", "$open"],
                    start_time=pred.index.get_level_values("datetime").min(),
                    end_time=pred.index.get_level_values("datetime").max())
    close = px["$close"].swaplevel().sort_index()
    open_ = px["$open"].swaplevel().sort_index()
    chg = close.groupby(level="instrument").pct_change()
    next_open = open_.groupby(level="instrument").shift(-1)
    x = pred.to_frame("score").join(chg.rename("chg"), how="left").join(
        (next_open / close - 1).rename("ogap"), how="left")
    code = x.index.get_level_values("instrument").str[2:]
    date = x.index.get_level_values("datetime")
    board = np.where(code.str.startswith("68"), 0.198,
                     np.where(code.str.startswith("30"), 0.198, 0.098))
    gem20 = code.str.startswith("30") & (date >= pd.Timestamp("2020-08-24"))
    threshold = np.where(gem20, 0.198, board)
    mask_t = x["chg"].ge(threshold).fillna(False)
    x.loc[mask_t, "score"] = np.nan
    close_t = close.reindex(x.index)
    open_threshold = threshold - 0.005 / close_t.to_numpy(dtype=float).clip(min=1e-9)
    mask_open = x["score"].notna() & pd.Series(
        x["ogap"].to_numpy(dtype=float) >= open_threshold, index=x.index)
    events = pd.concat([
        x.loc[mask_t, "score"].reset_index().assign(mask_reason="T_close_limit"),
        x.loc[mask_open, "score"].reset_index().assign(mask_reason="T1_open_limit"),
    ], ignore_index=True).rename(columns={"datetime": "date"})
    kept = x.loc[x["score"].notna(), "score"].reset_index().rename(columns={"datetime": "date"})
    if kept.empty:
        raise AssertionError("Qlib 评分屏蔽后为空")
    return kept, events


def run_account(score, baseline, store, capital=1_000_000.0):
    """逐行复用 P25 ReplayAccount/topk_plan，基准只提供日历和基准收益。"""
    baseline = baseline.copy()
    baseline["datetime"] = pd.to_datetime(baseline["datetime"])
    baseline = baseline[baseline["datetime"] <= er.PARAMS["formal_end"]].sort_values("datetime").reset_index(drop=True)
    assert_window_allowed(baseline["datetime"].min(), baseline["datetime"].max(), purpose="candidate")
    scores_by_date = {d: g.set_index("instrument")["score"] for d, g in score.groupby("date")}
    acct = er.ReplayAccount(capital=capital)
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
                sells, buys = er.topk_plan(signal, acct.pos, er.PARAMS["topk"], er.PARAMS["n_drop"])
                for code in sells:
                    if code not in acct.pos:
                        continue
                    bar = store.bar(code, d)
                    if bar is None:
                        acct._log(d, sd, code, "SELL", 0, 0, 0, 0, "HELD", "停牌或无有效开盘，继续持有")
                    else:
                        acct.sell(d, sd, code, bar)
                target = acct.cash * er.PARAMS["risk_degree"] / len(buys) if buys else 0.0
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
        prev_nav = rows[-1]["nav"] if rows else capital
        mkt_value = float(sum(p["qty"] * p["last_px"] for p in acct.pos.values()))
        b = baseline.iloc[i]
        strategy_return = nav / prev_nav - 1
        rows.append({
            "datetime": d, "cash": acct.cash, "mkt_value": mkt_value,
            "receivables": acct.receivable_total(), "nav": nav, "account_nav": nav,
            "strategy_return": strategy_return, "bench": float(b["bench"]),
            "excess": strategy_return - float(b["bench"]), "n_pos": len(acct.pos),
            "turnover": gross / (2 * prev_nav) if prev_nav else 0.0,
            "fee": acct.fee_cum - fee_before, "fee_cum": acct.fee_cum,
            "slippage": slippage, "corporate_actions": acct.corporate_actions,
            "corporate_action_imputed": getattr(acct, "corporate_action_imputed", 0),
            "corporate_action_stock_mismatch": getattr(acct, "corporate_action_stock_mismatch", 0),
            "dividend_receivable_new": acct.dividend_receivable_new,
            "blocked": sum(x["status"] == "BLOCKED" for x in day_orders),
            "held": sum(x["status"] == "HELD" for x in day_orders),
            "skipped": sum(x["status"] == "SKIPPED" for x in day_orders),
        })
        if not np.isfinite(nav) or nav <= 0 or acct.cash < -0.01:
            raise AssertionError(f"账户健康失败 {d.date()} nav={nav} cash={acct.cash}")
    return pd.DataFrame(rows), pd.DataFrame(acct.orders), pd.DataFrame(acct.action_log)


def metric(daily, capital=1_000_000.0):
    d = daily.copy()
    d["nav"] = (1 + d["strategy_return"]).cumprod() * capital
    out = er._metric_row(d)
    # _metric_row 的历史默认资金是 100 万；容量档必须用自身收益链计算总收益。
    out["total_net_return"] = float((1 + d["strategy_return"]).prod() - 1)
    out["max_positions"] = int(d["n_pos"].max())
    return out


def order_stats(orders):
    buys = orders[orders["side"].eq("BUY")] if not orders.empty else pd.DataFrame()
    planned = len(buys)
    unfilled = int(buys["status"].isin(["BLOCKED", "SKIPPED", "HELD"]).sum()) if planned else 0
    skipped = int(buys["status"].eq("SKIPPED").sum()) if planned else 0
    return {
        "planned_buys": int(planned), "filled_buys": int(buys["status"].eq("FILLED").sum()) if planned else 0,
        "unfilled_buys": unfilled, "unfilled_buy_rate": float(unfilled / planned) if planned else None,
        "blocked": int(orders["status"].eq("BLOCKED").sum()) if not orders.empty else 0,
        "skipped": skipped, "one_lot_skips": skipped,
        "one_lot_skip_rate": float(skipped / planned) if planned else None,
        "held": int(orders["status"].eq("HELD").sum()) if not orders.empty else 0,
    }


def health(daily, orders, expected_days=None):
    filled_buys = orders[(orders["side"] == "BUY") & (orders["status"] == "FILLED")] if not orders.empty else pd.DataFrame()
    checks = {
        "no_sealed_dates": bool(daily["datetime"].max() <= pd.Timestamp(er.PARAMS["formal_end"])),
        "no_negative_cash": bool((daily["cash"] >= -0.01).all()),
        "nav_identity": bool(np.allclose(daily["cash"] + daily["mkt_value"] + daily["receivables"], daily["nav"], atol=1e-6)),
        "all_filled_buys_round_lot": bool((filled_buys["qty"] % 100 == 0).all()) if len(filled_buys) else True,
        "no_future_signal": bool((pd.to_datetime(orders["signal_date"]) < pd.to_datetime(orders["exec_date"])).all()) if not orders.empty else True,
        "finite_returns": bool(np.isfinite(daily[["strategy_return", "bench", "excess"]].to_numpy()).all()),
        "turnover_plausible": bool(0 < daily["turnover"].mean() < 0.50),
    }
    if expected_days is not None:
        checks["expected_days"] = bool(len(daily) == expected_days)
    return {"passed": bool(all(checks.values())), "checks": checks}
