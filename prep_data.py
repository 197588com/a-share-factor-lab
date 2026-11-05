# -*- coding: utf-8 -*-
"""准备 Alpha158 数据集（分段缓存），供 ALSTM / MASTER 训练共用。"""
import os, sys, time, pickle
import qlib
from qlib.config import REG_CN
from qlib.contrib.data.handler import Alpha158

WS = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(WS, "data", "prep")
os.makedirs(CACHE, exist_ok=True)

SEGMENTS = {
    "train": ("2015-01-01", "2019-12-31"),
    "valid": ("2020-01-01", "2020-12-31"),
    "test":  ("2021-01-01", "2022-12-31"),
}
FIT = ("2015-01-01", "2019-12-31")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    h = Alpha158(
        instruments="csi300",
        start_time=SEGMENTS["train"][0],
        end_time=SEGMENTS["test"][1],
        fit_start_time=FIT[0],
        fit_end_time=FIT[1],
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
        ],
    )
    for seg, (s, e) in SEGMENTS.items():
        if only and seg != only:
            continue
        out = os.path.join(CACHE, f"{seg}.pkl")
        if os.path.exists(out):
            print(f"skip {seg} (cached)")
            continue
        t0 = time.time()
        df = h.fetch(selector=slice(s, e), level="datetime", col_set=["feature", "label"])
        df.to_pickle(out)
        print(f"{seg}: {df.shape} fetched in {time.time()-t0:.0f}s -> {out}", flush=True)
    print("PREP DONE")


if __name__ == "__main__":
    main()
