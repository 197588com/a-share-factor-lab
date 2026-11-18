# -*- coding: utf-8 -*-
"""一次性 P25 复核：Qlib 快照原函数 vs 本地 CSV 复刻。带 main guard 防 spawn。"""
from pathlib import Path
import numpy as np
import pandas as pd


def run(fold=9):
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    from execution_replay import load_adjusted_features, production_mask

    out = Path("data/backtest/execution_replay")
    qlib.init(provider_uri=str((out / "qlib_snapshot").resolve()), region=REG_CN, kernels=1)
    df = pd.read_parquet(f"data/scores/alstm_fold{fold}.parquet")
    df["instrument"] = df["instrument"].str.upper()
    df["date"] = pd.to_datetime(df["date"])
    baseline = pd.read_parquet(f"data/backtest/daily/alstm_fold{fold}_oc0.0005_cc0.0015.parquet")
    dates = pd.to_datetime(baseline["datetime"])
    df = df[df["date"].isin(set(dates[dates <= pd.Timestamp("2026-06-30")]))].copy()
    pred = df.set_index(["date", "instrument"])["score"]
    pred.index = pred.index.set_names(["datetime", "instrument"])
    insts = sorted(pred.index.get_level_values("instrument").unique())
    px = D.features(insts, ["$close", "$open"],
                    start_time=pred.index.get_level_values("datetime").min(),
                    end_time=pred.index.get_level_values("datetime").max())
    close = px["$close"].swaplevel().sort_index()
    open_ = px["$open"].swaplevel().sort_index()
    chg = close.groupby(level="instrument").pct_change()
    nxt_open = open_.groupby(level="instrument").shift(-1)
    x = pred.to_frame("score").join(chg.rename("chg"), how="left").join((nxt_open / close - 1).rename("ogap"), how="left")
    code = x.index.get_level_values("instrument").str[2:]
    board = np.where(code.str.startswith("68"), 0.198, np.where(code.str.startswith("30"), 0.198, 0.098))
    date = x.index.get_level_values("datetime")
    gem20 = code.str.startswith("30") & (date >= pd.Timestamp("2020-08-24"))
    thr = np.where(gem20, 0.198, board)
    mask_t = x["chg"].ge(thr).fillna(False)
    x.loc[mask_t, "score"] = np.nan
    close_t = close.reindex(x.index)
    ogap_th = thr - 0.005 / close_t.to_numpy(dtype=float).clip(min=1e-9)
    mask_open = x["score"].notna() & pd.Series(x["ogap"].to_numpy(dtype=float) >= ogap_th, index=x.index)
    q_events = pd.concat([
        x.loc[mask_t, "score"].reset_index().assign(mask_reason="T_close_limit"),
        x.loc[mask_open, "score"].reset_index().assign(mask_reason="T1_open_limit"),
    ], ignore_index=True).rename(columns={"datetime": "date"})
    q_kept = x.loc[x["score"].notna(), "score"].reset_index().rename(columns={"datetime": "date"})

    adjusted = load_adjusted_features(insts, df["date"].min(), df["date"].max())
    local_kept, local_events = production_mask(df, adjusted)
    q_keys = set(map(tuple, q_kept[["date", "instrument"]].to_numpy()))
    local_keys = set(map(tuple, local_kept[["date", "instrument"]].to_numpy()))
    diff = sorted(q_keys.symmetric_difference(local_keys))
    diff_df = pd.DataFrame(diff, columns=["date", "instrument"])
    if not diff_df.empty:
        diff_df["source"] = ["qlib_only" if k in q_keys else "local_only" for k in diff]
    q_kept.to_parquet(out / f"mask_kept_qlib_fold{fold}.parquet", index=False)
    q_events.to_parquet(out / f"mask_events_qlib_fold{fold}.parquet", index=False)
    diff_df.to_csv(out / f"mask_diff_fold{fold}.csv", index=False)
    print(f"QLIB_SNAPSHOT fold={fold} T={int(mask_t.sum())} T1={int(mask_open.sum())} "
          f"kept={len(q_kept)} local_kept={len(local_kept)} key_diff={len(diff)}")
    if len(diff):
        print(diff_df.head(20).to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=9)
    run(parser.parse_args().fold)
