# -*- coding: utf-8 -*-
"""blend_test.py — 组合层第二刀：IVOL 独立组合验证 + 信号融合回测（P9）。

背景：P6 五单证明饱和模型装不下新因子（IVOL 单因子 12 年满分但模型增量为零）。
本脚本绕开饱和模型，在组合层直接消费 IVOL alpha：

第一步  IVOL 独立多空诊断（2015-2026）：
  每日按 IVOL 升序，做多最低 30% / 做空最高 30%，收益用 labels.parquet
  （T+1→T+2 前向收益，与全项目同口径）；不计成本（诊断用途，如实标注）。

第二步  信号融合回测（folds 8-11，qlib TopkDropout 同 backtest_p4 引擎）：
  blended = w·rank(模型分) + (1-w)·rank(-IVOL)
  w ∈ {1.0 基线校准, 0.7, 0.5, 0.0 纯 IVOL}，与基线年化净超额对照。

  python blend_test.py ls          # 第一步：IVOL 多空诊断
  python blend_test.py blend       # 第二步：融合回测
"""
import os, sys
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import wf_pipeline as wf
import backtest_p4 as bp

FD = os.path.join(WS, "data", "factor_decay")
FOLDS = [8, 9, 10, 11]
WEIGHTS = [1.0, 0.7, 0.5, 0.0]


def ivol_ls():
    feat = pd.read_parquet(os.path.join(FD, "min_feat.parquet"))[["IVOL"]]
    lab = pd.read_parquet(os.path.join(FD, "labels.parquet"))
    df = feat.join(lab, how="inner").dropna()
    dd = df.index.get_level_values("datetime")
    rows = []
    for y in range(2015, 2027):
        sub = df[(dd >= f"{y}-01-01") & (dd <= f"{y}-12-31")]
        if not len(sub):
            continue
        d2 = sub.index.get_level_values("datetime")
        parts = []
        for d, g in sub.groupby(d2):
            if len(g) < 100:
                continue
            k = max(10, int(len(g) * 0.3))
            s = g.sort_values("IVOL")
            parts.append(s["y"].iloc[:k].mean() - s["y"].iloc[-k:].mean())
        r = pd.Series(parts)
        rows.append({"year": y, "ls_ann": r.mean() * 244,
                     "ls_ir": r.mean() / (r.std() + 1e-12) * np.sqrt(244),
                     "days": len(r)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(WS, "data", "backtest", "ivol_ls.csv"), index=False)
    print(out.round(4).to_string(index=False))
    print(f"\n12年均值: ann={out['ls_ann'].mean()*100:.2f}%  IR={out['ls_ir'].mean():.2f}")


def load_scores_ivol(fid):
    """fold 测试窗：模型分 + IVOL，统一 (Timestamp, UPPER) 索引。"""
    df = pd.read_parquet(os.path.join(bp.SCORE_DIR, f"alstm_fold{fid}.parquet"))
    df["instrument"] = df["instrument"].str.upper()
    sc = df.set_index(["date", "instrument"])["score"]
    sc.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(sc.index.get_level_values(0)),
         sc.index.get_level_values(1)], names=["datetime", "instrument"])
    iv = pd.read_parquet(os.path.join(FD, "min_feat.parquet"))["IVOL"]
    iv.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(iv.index.get_level_values(0)),
         iv.index.get_level_values(1).str.upper()], names=["datetime", "instrument"])
    j = sc.to_frame("score").join(iv.to_frame("IVOL"), how="left")
    # IVOL 缺失（停牌段）：中性 0.5 秩
    j["IVOL"] = j["IVOL"].fillna(j["IVOL"].median())
    return j


def blend_pred(j, w):
    r_m = j.groupby(level="datetime")["score"].rank(pct=True)
    r_i = j.groupby(level="datetime")["IVOL"].rank(pct=True)
    if np.isscalar(w):
        b = w * r_m + (1 - w) * (1 - r_i)      # 低 IVOL 秩高
    else:
        # w 为逐日权重序列（adaptive 模式）：映射到 (datetime, instrument) 行
        wd = w.to_dict()
        dl = r_m.index.get_level_values("datetime")
        wrow = pd.Series(dl.map(lambda d: wd.get(d, 1.0)).to_numpy(dtype=float), index=r_m.index)
        b = wrow * r_m + (1 - wrow) * (1 - r_i)
    pred = b.dropna()
    pred = pred[pred.index.get_level_values("datetime").isin(
        j.index.get_level_values("datetime").unique())]
    return pred


def ivol_ls_daily():
    """逐日 IVOL 多空收益序列（2015-2026，用于自适应门控的历史判断）。"""
    fp = os.path.join(WS, "data", "backtest", "ivol_ls_daily.parquet")
    if os.path.exists(fp):
        return pd.read_parquet(fp)["ls"]
    feat = pd.read_parquet(os.path.join(FD, "min_feat.parquet"))[["IVOL"]]
    lab = pd.read_parquet(os.path.join(FD, "labels.parquet"))
    df = feat.join(lab, how="inner").dropna()
    dd = df.index.get_level_values("datetime")
    parts = []
    for d, g in df.groupby(dd):
        if len(g) < 100:
            continue
        k = max(10, int(len(g) * 0.3))
        s = g.sort_values("IVOL")
        parts.append({"d": d, "ls": s["y"].iloc[:k].mean() - s["y"].iloc[-k:].mean()})
    ls = pd.DataFrame(parts).set_index("d")["ls"]
    ls.to_frame().to_parquet(fp)
    return ls


def blend():
    folds = {f["id"]: f for f in wf.generate_folds(wf.load_config())}
    ls = ivol_ls_daily()
    ls_trail = ls.rolling(60, min_periods=20).mean()     # 仅用历史，无未来函数
    rows = []
    for fid in FOLDS:
        f = folds[fid]
        j = load_scores_ivol(fid)
        dates = j.index.get_level_values("datetime").unique().sort_values()
        tilt = (ls_trail.reindex(dates).shift(1) > 0).map({True: 0.7, False: 1.0})
        tilt.index = dates
        scenarios = [("w", w) for w in WEIGHTS] + [("adaptive", tilt)]
        for tag, w in scenarios:
            pred = blend_pred(j, w)
            pred = bp.mask_limit_up(pred.sort_index())
            last_dt = pred.index.get_level_values("datetime").max()
            cal = pd.read_csv(os.path.expanduser("~/.qlib/qlib_data/cn_data/calendars/day.txt"),
                              header=None)[0].tolist()
            if str(last_dt)[:10] >= cal[-1]:
                pred = pred[pred.index.get_level_values("datetime") < last_dt]
            r = bp.run_bt(pred, f["test"][0], f["test"][1], 0.0005, 0.0015)
            rows.append({"fold": fid, "year": f["test"][0][:4], "scenario": f"{tag}:{w}" if tag == "w" else "adaptive",
                         "ann_excess": r["ann_excess"], "ir": r["ir"],
                         "mdd": r["mdd"], "turnover": r["turnover"]})
            print(f"fold{fid} ({f['test'][0][:4]}) {tag}:{w if tag=='w' else '60d'}: "
                  f"ann={r['ann_excess']*100:+.2f}% IR={r['ir']:.2f} "
                  f"MDD={r['mdd']*100:.1f}% to={r['turnover']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(WS, "data", "backtest", "blend_test.csv"), index=False)
    print("\n年化净超额(%)：")
    print(out.pivot_table(index="scenario", columns="year",
                          values="ann_excess").round(4) * 100)


if __name__ == "__main__":
    if sys.argv[1] == "ls":
        ivol_ls()
    else:
        blend()
