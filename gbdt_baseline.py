# -*- coding: utf-8 -*-
"""gbdt_baseline.py — 线性/树模型轨道对照实验（框架双轨制）。

动机：P5 延伸发现 ALSTM 在 2025-2026 的 RankIC ≈ 0。按 1.3 双轨制与 weekly 的
维持/替换规则，用 LightGBM 在 fold11 训练窗（2019-2024）上训练同口径 Alpha158 标签，
分别评估验证窗（2025）与测试窗（2026）RankIC，与 ALSTM fold11 对比：
  - GBDT 明显更强 → live 候选切换（仍需过回测闸门）
  - 同样近零 → 衰减在因子/数据层，须上新因子（归 1.4 生命周期）

样本构造与 WindowDS 同口径（8 日特征完整 + 标签非 NaN），特征取最近一日向量
（Alpha158 本身为滚动统计量，无需序列编码）。
"""
import os, sys, time, json, pickle
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import wf_pipeline as wf
import train_models as tm


def samples(blob, seg, lookback=8):
    """面板 -> (X, y, dates, inst_idx)，口径同 WindowDS。"""
    panels, meta = blob["panels"], blob["meta"]
    F, L = panels[seg]
    idx = wf.seg_indices(blob, seg)
    X, Y, D, I = [], [], [], []
    Fm = np.isfinite(F).all(axis=2)
    for t in idx:
        if t < lookback - 1:
            continue
        win = Fm[t - lookback + 1: t + 1].all(axis=0)
        lab = np.isfinite(L[t])
        for i in np.where(win & lab)[0]:
            X.append(F[t, i])
            Y.append(L[t, i])
            D.append(meta["dates"][t])
            I.append(i)
    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64), \
        np.asarray(D), np.asarray(I)


def daily_rankic(pred, y, dates):
    df = pd.DataFrame({"d": dates, "p": pred, "y": y}).dropna()
    ric = df.groupby("d").apply(lambda g: g["p"].corr(g["y"], method="spearman"),
                                include_groups=False)
    ric = ric.dropna()
    return float(ric.mean()), float(ric.mean() / (ric.std() + 1e-12))


def main():
    cfg = wf.load_config()
    fold = wf.generate_folds(cfg)[-1]
    print(f"fold{fold['id']} train={fold['train']} valid={fold['valid']} test={fold['test']}", flush=True)
    blob = wf.load_fold(fold["id"])

    t0 = time.time()
    Xtr, ytr, dtr, itr = samples(blob, "train")
    Xva, yva, dva, iva = samples(blob, "valid")
    Xte, yte, dte, ite = samples(blob, "test")
    print(f"samples train={Xtr.shape} valid={Xva.shape} test={Xte.shape} ({time.time()-t0:.0f}s)", flush=True)

    # LightGBM 在本机 numpy2 环境下 Dataset 构造段错误（随机数据可过、真实数据必崩），
    # 改用 sklearn HistGradientBoostingRegressor（同族直方图提升，环境已验证）
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(42)
    sub = rng.choice(len(Xtr), size=min(250_000, len(Xtr)), replace=False)
    m = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.06, max_leaf_nodes=63,
        l2_regularization=10.0, early_stopping=True, validation_fraction=0.1,
        random_state=42)
    t0 = time.time()
    m.fit(Xtr[sub], ytr[sub])
    print(f"trained n_iter={m.n_iter_} ({time.time()-t0:.0f}s)", flush=True)

    out = {"fold": fold["id"], "model": "hgb_alpha158", "n_iter": int(m.n_iter_)}
    for seg, X, y, d in (("valid2025", Xva, yva, dva), ("test2026", Xte, yte, dte)):
        ric, ricir = daily_rankic(m.predict(X), y, d)
        out[seg] = {"RankIC": ric, "RankICIR": ricir}
        print(f"{seg}: RankIC={ric:.4f} RankICIR={ricir:.3f}", flush=True)

    os.makedirs(os.path.join(WS, "data", "live"), exist_ok=True)
    json.dump(out, open(os.path.join(WS, "data", "live", "lgbm_baseline_fold11.json"), "w"),
              ensure_ascii=False, indent=2)
    pickle.dump(m, open(os.path.join(WS, "ckpt", "wf", "hgb_fold11.pkl"), "wb"))


if __name__ == "__main__":
    main()
