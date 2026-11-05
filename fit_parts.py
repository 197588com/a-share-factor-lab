# -*- coding: utf-8 -*-
"""fit_parts.py — 分年拉取 fold-7 训练窗原始特征并合成拟合处理器。

bootstrap 的计算拆解（每步 < 300s，免托管任务）：
  python fit_parts.py fetch 2015   # 逐年：原始特征存 data/live/fit_parts/
  ...
  python fit_parts.py stats        # 合成 median/MAD → processors_alstm.pkl

口径与 walk-forward 训练完全一致：RobustZScoreNorm(median, MAD×1.4826) + clip(-3,3)
+ Fillna(0)，fit 窗 = fold7 train 2015-01-01..2020-12-31。
"""
import os, sys, time, pickle, json
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(WS, "data", "live")
PARTS = os.path.join(LIVE, "fit_parts")
os.makedirs(PARTS, exist_ok=True)
FIT_START, FIT_END = "2015-01-01", "2020-12-31"


def main():
    cmd = sys.argv[1]
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    from qlib.data.dataset.loader import QlibDataLoader
    from qlib.contrib.data.handler import Alpha158
    a158 = Alpha158.__new__(Alpha158)
    loader = QlibDataLoader(config={"feature": a158.get_feature_config()})

    if cmd == "fetch":
        year = sys.argv[2]
        s, e = f"{year}-01-01", f"{year}-12-31"
        t0 = time.time()
        raw = loader.load(instruments="csi300", start_time=s, end_time=e)
        feat = raw["feature"]
        cols = list(feat.columns)
        np.save(os.path.join(PARTS, f"feat_{year}.npy"), feat.to_numpy(dtype=np.float32))
        json.dump(cols, open(os.path.join(PARTS, "cols.json"), "w"))
        json.dump([str(d)[:10] for d in sorted(feat.index.get_level_values("datetime").unique())],
                  open(os.path.join(PARTS, f"dates_{year}.json"), "w"))
        print(f"{year}: {feat.shape} {time.time()-t0:.0f}s -> feat_{year}.npy", flush=True)

    elif cmd == "stats":
        # 用法: python fit_parts.py stats [--fit-start 2019-01-01 --fit-end 2024-12-31]
        # 默认取 wf 最新 fold 的训练窗；选用完整落在区间内的年度分片
        import argparse as _ap
        _p = _ap.ArgumentParser()
        _p.add_argument("--fit-start", default=None)
        _p.add_argument("--fit-end", default=None)
        _a, _ = _p.parse_known_args(sys.argv[2:])
        if _a.fit_start is None:
            import wf_pipeline as wf
            _cfg = wf.load_config()
            _fold = wf.generate_folds(_cfg)[-1]
            fit_start, fit_end = _fold["train"]
            print(f"fit window from latest fold {_fold['id']}: {fit_start}..{fit_end}", flush=True)
        else:
            fit_start, fit_end = _a.fit_start, _a.fit_end
        y0, y1 = int(fit_start[:4]), int(fit_end[:4])
        from qlib.data.dataset.processor import RobustZScoreNorm, Fillna
        cols = json.load(open(os.path.join(PARTS, "cols.json")))
        years = [y for y in range(y0, y1 + 1)
                 if os.path.exists(os.path.join(PARTS, f"feat_{y}.npy"))]
        t0 = time.time()
        arrs = [np.load(os.path.join(PARTS, f"feat_{y}.npy")) for y in years]
        X = np.concatenate(arrs, axis=0)
        del arrs
        print(f"stacked {X.shape} years={years} in {time.time()-t0:.0f}s", flush=True)
        mean = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - mean), axis=0)
        std = (mad + 1e-12) * 1.4826
        z = RobustZScoreNorm.__new__(RobustZScoreNorm)
        z.cols = [("feature", c) for c in cols]   # 与 get_group_columns 返回口径一致（元组）
        z.mean_train = mean; z.std_train = std; z.clip_outlier = True
        z.fields_group = "feature"
        pickle.dump([z, Fillna(fields_group="feature")],
                    open(os.path.join(LIVE, "processors_alstm.pkl"), "wb"))
        json.dump(cols, open(os.path.join(LIVE, "feature_order.json"), "w"))
        print(f"processors saved: years={years} rows={X.shape[0]} cols={len(cols)}", flush=True)


if __name__ == "__main__":
    main()
