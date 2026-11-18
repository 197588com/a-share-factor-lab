# -*- coding: utf-8 -*-
"""P26 输入覆盖审计：只统计缺失与日期覆盖，不计算任何候选 IC。"""
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent
FD = WS / "data" / "factor_decay"
START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2026-06-26")

ivol = pd.read_parquet(FD / "min_feat.parquet", columns=["IVOL"])
labels = pd.read_parquet(FD / "labels.parquet")
print("IVOL rows/date range/non-null:", len(ivol), ivol.index.get_level_values("datetime").min(),
      ivol.index.get_level_values("datetime").max(), int(ivol["IVOL"].notna().sum()))
print("label columns:", list(labels.columns))
print("labels rows/date range/non-null:", len(labels), labels.index.get_level_values("datetime").min(),
      labels.index.get_level_values("datetime").max(), int(labels.iloc[:, 0].notna().sum()))

for year in range(2015, 2027):
    feat = pd.read_parquet(FD / f"feat_{year}.parquet")
    dates = feat.index.get_level_values("datetime")
    feat = feat[(dates >= START) & (dates <= END)]
    all_empty = [c for c in feat.columns if not feat[c].notna().any()]
    if "VWAP0" in feat:
        feat["VWAP0"] = 0.0
    complete = np.isfinite(feat.to_numpy(dtype=np.float64)).all(axis=1)
    f = feat.loc[complete]
    i = ivol[(ivol.index.get_level_values("datetime") >= pd.Timestamp(year, 1, 1))
             & (ivol.index.get_level_values("datetime") <= min(pd.Timestamp(year, 12, 31), END))]
    y = labels[(labels.index.get_level_values("datetime") >= pd.Timestamp(year, 1, 1))
               & (labels.index.get_level_values("datetime") <= min(pd.Timestamp(year, 12, 31), END))]
    j = f.join(i, how="inner").join(y, how="inner").replace([np.inf, -np.inf], np.nan)
    complete_joint = j.dropna()
    counts = complete_joint.groupby(level="datetime").size()
    print({
        "year": year, "feature_rows": len(feat), "all_empty_features": all_empty,
        "complete_feature_rows": int(complete.sum()), "complete_feature_days": int(f.index.get_level_values("datetime").nunique()),
        "ivol_rows": len(i), "ivol_nonnull": int(i["IVOL"].notna().sum()),
        "ivol_first": str(i.index.get_level_values("datetime").min()), "ivol_last": str(i.index.get_level_values("datetime").max()),
        "label_nonnull": int(y.iloc[:, 0].notna().sum()), "joint_complete_rows": len(complete_joint),
        "feature_instruments": int(feat.index.get_level_values("instrument").nunique()),
        "ivol_instruments": int(i.index.get_level_values("instrument").nunique()),
        "joint_days": int(len(counts)), "joint_days_ge_200": int((counts >= 200).sum()),
        "joint_daily_n_min": int(counts.min()), "joint_daily_n_median": float(counts.median()),
        "joint_daily_n_max": int(counts.max()),
    })
