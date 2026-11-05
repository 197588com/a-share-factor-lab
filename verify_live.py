# -*- coding: utf-8 -*-
"""verify_live.py — 交叉核对 daily 路径特征与最新 fold 测试面板逐值一致性。

用法: python verify_live.py [--fold 11] [--asof 2026-06-30]
asof 默认取该 fold 测试年中点，须落在面板日期内。
"""
import sys, pickle, json, argparse
sys.path.insert(0, ".")
import numpy as np
import live_infer as li


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--asof", default=None)
    a = ap.parse_args()
    cfg = li.load_cfg()
    folds = li.latest_fold(cfg) if a.fold is None else None
    fold_id = a.fold if a.fold is not None else folds["id"]
    test_start = a.asof or "2026-06-30"
    feat = li.fetch_live_frame_daily(cfg, test_start)
    F, dates, insts = li.build_panel(feat)
    t = [i for i, d in enumerate(dates) if str(d)[:10] == test_start][0]

    blob = pickle.load(open(f"data/prep_wf/fold_{fold_id}.pkl", "rb"))
    panels, meta = blob["panels"], blob["meta"]
    Ft = panels["test"][0]
    import datetime as dtm
    key = dtm.date(*map(int, test_start.split("-")))
    ti = {d: i for i, d in enumerate(meta["dates"])}[key]
    print(f"fold{fold_id} panel date idx: {ti}", flush=True)

    worst = (0.0, None)
    for code in ["sh600000", "sh600519", "sz000001", "sz300750", "sh601318"]:
        if code not in insts or code not in meta["insts"]:
            print(code, "skip (不在某侧宇宙)", flush=True)
            continue
        d_ = float(np.nanmax(np.abs(F[t, insts.index(code), :] - Ft[ti, meta["insts"].index(code), :])))
        if d_ > worst[0]:
            worst = (d_, code)
        print(code, "max|diff| =", round(d_, 6), flush=True)
    print("WORST:", worst, flush=True)


if __name__ == "__main__":
    main()
