# -*- coding: utf-8 -*-
"""regime 识别器 v2：Gaussian HMM 概率化市场状态。

设计要点（对应 framework_design.md 1.4b）：
- 特征：沪深300 日收益、日内振幅、量能变化（3 维观测）
- 参数只用 2008-2014 训练窗估计，之后不再重估（无全样本偷看；滚动重估归 P3 walk-forward）
- 历史标签用因果滤波概率（expanding window：每日只用当日及以前数据），
  而非全序列平滑概率——研究回测与实时输出同一口径
- 状态经济学命名按训练窗内特征签名自动映射：bull / bear / shock / quiet
- 硬标签 + 概率分别落 DuckDB：regime_labels(source='hmm_v2') / regime_probs

用法：
    python factor_lab/regime_hmm.py run      # 首次拟合 + 全历史解码；之后增量续算
    python factor_lab/regime_hmm.py status   # 打印当前状态与转移矩阵
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(WS, "data", "factor_lab.duckdb")
MODEL_PATH = os.path.join(WS, "data", "models", "regime_hmm_v2.pkl")
TRAIN_END = "2014-12-31"
N_STATES = 4
SOURCE, VERSION = "hmm_v2", 2


def load_features():
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    df = D.features(["sh000300"], ["$open", "$high", "$low", "$close", "$volume"],
                    start_time="2008-01-01", end_time="2026-12-31")
    df.columns = ["open", "high", "low", "close", "volume"]
    px = df.xs("sh000300", level="instrument").sort_index() if isinstance(df.index, pd.MultiIndex) else df.sort_index()
    px = px.dropna()
    feat = pd.DataFrame(index=px.index)
    feat["ret"] = px["close"].pct_change()
    feat["range"] = (px["high"] - px["low"]) / px["close"]
    feat["dvol"] = px["volume"].pct_change().clip(-1, 5)
    return feat.dropna()


def state_names(model, X_train):
    """按训练窗解码的特征签名自动命名状态。"""
    from hmmlearn.hmm import GaussianHMM
    post = model.predict_proba(X_train)
    hard = post.argmax(1)
    stats = []
    for s in range(N_STATES):
        m = hard == s
        stats.append({
            "state": s, "frac": m.mean(),
            "mean_ret": X_train[m, 0].mean(), "mean_range": X_train[m, 1].mean(),
        })
    st = pd.DataFrame(stats).set_index("state")
    names = {}
    shock = st["mean_range"].idxmax()                      # 振幅最高 -> shock
    names[shock] = "shock"
    rest = st.drop(index=shock)
    names[rest["mean_ret"].idxmax()] = "bull"
    names[rest["mean_ret"].idxmin()] = "bear"
    left = [s for s in st.index if s not in names][0]
    names[left] = "quiet"
    return names, st


def fit_or_load(X):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as fh:
            return pickle.load(fh)
    from hmmlearn.hmm import GaussianHMM
    Xtr = X.loc[:TRAIN_END].values
    model = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                        n_iter=300, random_state=7, tol=1e-4)
    model.fit(Xtr)
    names, st = state_names(model, Xtr)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": model, "names": names, "train_end": TRAIN_END,
                     "train_stats": st, "features": list(X.columns)}, fh)
    print(f"HMM fitted on {Xtr.shape[0]} days (..{TRAIN_END}); "
          f"loglik={model.score(Xtr):.1f}; states={names}")
    return {"model": model, "names": names, "train_end": TRAIN_END,
            "train_stats": st, "features": list(X.columns)}


def filtered_probs(model, X, start_pos):
    """expanding-window 因果滤波：第 t 日的后验只用 X[:t+1]。"""
    n = len(X)
    out = np.full((n, N_STATES), np.nan)
    Xv = X.values
    for t in range(start_pos, n):
        out[t] = model.predict_proba(Xv[:t + 1])[-1]
    return out


def run():
    import duckdb
    X = load_features()
    bundle = fit_or_load(X)
    model, names = bundle["model"], bundle["names"]

    con = duckdb.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS regime_probs(
        trade_date DATE, state VARCHAR, prob DOUBLE, source VARCHAR, version INT,
        PRIMARY KEY(trade_date, state, source))""")
    done = con.execute("SELECT COUNT(DISTINCT trade_date) FROM regime_probs WHERE source=?", [SOURCE]).fetchone()[0]

    if done == 0:
        # 训练窗内用平滑后验（研究惯例），训练窗外用因果滤波
        Xtr = X.loc[:TRAIN_END]
        post_tr = model.predict_proba(Xtr.values)
        n_tr = len(Xtr)
        probs = filtered_probs(model, X, n_tr)   # 全长数组，训练窗段为 NaN
        probs[:n_tr] = post_tr                   # 训练窗段替换为平滑后验
        print(f"decoded: train(smoothed) {n_tr}d + live(filtered) {len(X)-n_tr}d")
        con.execute("DELETE FROM regime_labels WHERE source=?", [SOURCE])
        con.execute("DELETE FROM regime_probs WHERE source=?", [SOURCE])
    else:
        # 增量：只对尚未解码的日期做滤波（从已完成日期的前一天起，保证窗口连续）
        last = con.execute("SELECT MAX(trade_date) FROM regime_probs WHERE source=?", [SOURCE]).fetchone()[0]
        if last:
            pos = X.index.searchsorted(pd.Timestamp(last))
            start_pos = max(0, int(pos) - 1)
        else:
            start_pos = 0
        probs = filtered_probs(model, X, start_pos)
        con.execute("DELETE FROM regime_probs WHERE source=? AND trade_date > ?", [SOURCE, last])

    dates = X.index.date
    long_rows, label_rows = [], []
    for i, d in enumerate(dates):
        if np.isnan(probs[i]).all():
            continue
        p = probs[i]
        for s in range(N_STATES):
            long_rows.append((d, names[s], float(p[s]), SOURCE, VERSION))
        label_rows.append((d, names[int(p.argmax())], SOURCE, VERSION))
    con.executemany("INSERT OR REPLACE INTO regime_probs VALUES (?,?,?,?,?)", long_rows)
    con.executemany("INSERT OR REPLACE INTO regime_labels VALUES (?,?,?,?)", label_rows)
    con.close()
    print(f"DB updated: {len(label_rows)} labels, {len(long_rows)} prob rows (source={SOURCE})")


def refit(years=None, force=False):
    """滚动重估（P3 walk-forward 口径）：每年初用此前全部数据重估 HMM，
    当年标签只用当年模型做因果滤波——与实时决策可用的信息严格一致。
    结果落 source='hmm_roll'（version=年份），不覆盖生产标签 hmm_v2。
    """
    import duckdb
    X = load_features()
    con = duckdb.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS regime_probs(
        trade_date DATE, state VARCHAR, prob DOUBLE, source VARCHAR, version INT,
        PRIMARY KEY(trade_date, state, source))""")
    if years is None:
        first = pd.Timestamp(TRAIN_END).year + 1
        last = X.index[-1].year
        years = list(range(first, last + 1))
    from hmmlearn.hmm import GaussianHMM
    for y in years:
        done = con.execute("SELECT COUNT(*) FROM regime_labels WHERE source='hmm_roll' AND version=?",
                           [y]).fetchone()[0]
        if done and not force:
            print(f"{y}: already refit ({done}d), skip")
            continue
        cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=1)
        Xtr = X.loc[:cutoff]
        if len(Xtr) < 500:
            print(f"{y}: insufficient train data ({len(Xtr)}d), skip")
            continue
        model = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                            n_iter=300, random_state=7, tol=1e-4)
        model.fit(Xtr.values)
        names, st = state_names(model, Xtr.values)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        mp = os.path.join(WS, "data", "models", f"regime_hmm_roll_{y}.pkl")
        with open(mp, "wb") as fh:
            pickle.dump({"model": model, "names": names, "train_end": str(cutoff.date()),
                         "train_stats": st, "features": list(X.columns)}, fh)
        # 当年日期因果滤波（只用当年模型 + 当日及以前数据）
        Xy = X.loc[f"{y}-01-01":]
        if len(Xy) == 0:
            print(f"{y}: no dates yet, model saved only")
            continue
        pos0 = X.index.searchsorted(Xy.index[0])
        probs = filtered_probs(model, X, pos0)
        dates = X.index.date
        long_rows, label_rows = [], []
        for i in range(pos0, len(X)):
            if np.isnan(probs[i]).all():
                continue
            d = dates[i]
            if d.year != y:      # 只写当年（searchsorted 起点可能含前一年末日）
                continue
            for s in range(N_STATES):
                long_rows.append((d, names[s], float(probs[i][s]), "hmm_roll", y))
            label_rows.append((d, names[int(probs[i].argmax())], "hmm_roll", y))
        con.execute("DELETE FROM regime_probs WHERE source='hmm_roll' AND version=?", [y])
        con.execute("DELETE FROM regime_labels WHERE source='hmm_roll' AND version=?", [y])
        con.executemany("INSERT OR REPLACE INTO regime_probs VALUES (?,?,?,?,?)", long_rows)
        con.executemany("INSERT OR REPLACE INTO regime_labels VALUES (?,?,?,?)", label_rows)
        print(f"{y}: refit on {len(Xtr)}d (..{cutoff.date()}), decoded {len(label_rows)}d, states={names}", flush=True)
    con.close()


def status():
    import duckdb
    con = duckdb.connect(DB, read_only=True)
    cur = con.execute("""SELECT trade_date, state, MAX(prob) FROM regime_probs
                         WHERE source=? AND trade_date >= CURRENT_DATE - 10
                         GROUP BY 1,2 ORDER BY 1""", [SOURCE]).fetchall()
    print("近10日 HMM 状态:")
    for d, s, p in cur:
        print(f"  {d}  {s:6s}  p={p:.3f}")
    con.close()
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as fh:
            b = pickle.load(fh)
        A = b["model"].transmat_
        order = [b["names"][i] for i in range(N_STATES)]
        print("\n状态命名(按隐藏状态号):", b["names"])
        print("\n训练窗特征签名:")
        print(b["train_stats"].round(4).to_string())
        print("\n转移矩阵(行=from, 列=to):")
        print(pd.DataFrame(A, index=order, columns=order).round(3).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "refit":
        yrs = [int(x) for x in sys.argv[2:]] or None
        refit(years=yrs)
    else:
        {"run": run, "status": status}[cmd]()
