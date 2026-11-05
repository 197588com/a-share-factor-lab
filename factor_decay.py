# -*- coding: utf-8 -*-
"""factor_decay.py — Alpha158 因子族 2015-2026 分段衰竭诊断（P1 注册表口径）。

回答：2025-2026 近零 alpha 是全部因子齐衰还是结构性的——哪些因子族还活着。

  python factor_decay.py fetch 2025      # 逐年拉原始特征存 parquet（含 datetime/instrument 行索引）
  python factor_decay.py labels          # 标签：Ref($close,-2)/Ref($close,-1)-1（T+1→T+2，同训练口径）
  python factor_decay.py report          # 逐日 Spearman IC × 因子族 × 年 → 衰减表

口径要点：与训练同标签（次日开盘到次日开盘的前向收益）；逐日横截面 RankIC 的
年均值与 ICIR；因子族按特征名前缀聚合（KMID/KLEN/…、ROC/MA/STD/BETA/…）。
"""
import os, sys, re, json, time
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(WS, "data", "factor_decay")
os.makedirs(FD, exist_ok=True)

FAMILY = [
    ("kbar", ["KMID", "KLEN", "KUP", "KLOW", "KSFT"]),
    ("price", ["OPEN", "HIGH", "LOW", "VWAP"]),
    ("roc", ["ROC"]),
    ("ma", ["MA"]),
    ("std", ["STD"]),
    ("risk", ["BETA", "RSQR", "RESI"]),
    ("range", ["MAX", "MIN", "QTLU", "QTLD"]),
    ("rank", ["RANK"]),
    ("rsv", ["RSV"]),
    ("imbal", ["IMAX", "IMIN", "IMXD"]),
    ("corr", ["CORR", "CORD"]),
    ("cnt", ["CNTP", "CNTN", "CNTD"]),
    ("sum", ["SUMP", "SUMN", "SUMD"]),
    ("vol", ["VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"]),
]


def family_of(col):
    base = re.sub(r"\d+$", "", col)
    for fam, prefs in FAMILY:
        if base in prefs:
            return fam
    return "other"


def qlib_ctx():
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    return qlib


def fetch_year(year):
    qlib_ctx()
    from qlib.data.dataset.loader import QlibDataLoader
    from qlib.contrib.data.handler import Alpha158
    a158 = Alpha158.__new__(Alpha158)
    loader = QlibDataLoader(config={"feature": a158.get_feature_config()})
    t0 = time.time()
    raw = loader.load(instruments="csi300", start_time=f"{year}-01-01", end_time=f"{year}-12-31")
    feat = raw["feature"]
    feat.to_parquet(os.path.join(FD, f"feat_{year}.parquet"))
    print(f"{year}: {feat.shape} {time.time()-t0:.0f}s -> feat_{year}.parquet", flush=True)


def make_labels():
    qlib_ctx()
    from qlib.data import D
    px = D.features(D.instruments("csi300"), ["$close"], start_time="2015-01-01",
                    end_time="2026-12-31").squeeze()
    close = px.unstack("instrument").sort_index()
    label = close.shift(-2) / close.shift(-1) - 1     # Ref($close,-2)/Ref($close,-1)-1
    label.stack().rename("y").to_frame().to_parquet(os.path.join(FD, "labels.parquet"))
    print(f"labels {label.shape} -> labels.parquet", flush=True)


def rank_ic_by_year(year):
    """单年：逐特征逐日 Spearman IC（groupby 秩 + 向量化 Pearson），返回 DataFrame[fam, mean, icir]。"""
    feat = pd.read_parquet(os.path.join(FD, f"feat_{year}.parquet"))
    lab = pd.read_parquet(os.path.join(FD, "labels.parquet"))
    df = feat.join(lab, how="inner")
    d = df.index.get_level_values("datetime")
    rows = []
    for col in feat.columns:
        sub = pd.DataFrame({"d": d, "f": df[col], "y": df["y"]}).dropna()
        if len(sub) < 1000:
            continue
        r = sub.groupby("d")[["f", "y"]].rank()
        sd = sub["d"]                       # rank 结果与 sub 行对齐，日期键取 sub
        g = r["f"].groupby(sd)
        fx, fy = r["f"] - g.transform("mean"), r["y"] - r["y"].groupby(sd).transform("mean")
        cov = (fx * fy).groupby(sd).sum()
        ic = cov / np.sqrt((fx ** 2).groupby(sd).sum() * (fy ** 2).groupby(sd).sum())
        ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
        ic = ic[sd.groupby(sd).size().reindex(ic.index) >= 30]   # 每日至少 30 只才计
        if len(ic) < 30:
            continue
        rows.append({"year": year, "feature": col, "family": family_of(col),
                     "IC": float(ic.mean()), "ICIR": float(ic.mean() / (ic.std() + 1e-12))})
    return pd.DataFrame(rows)


def report():
    years = sorted(int(f[5:9]) for f in os.listdir(FD)
                   if re.match(r"feat_\d{4}\.parquet", f))
    all_ic = []
    for y in years:
        t0 = time.time()
        r = rank_ic_by_year(y)
        all_ic.append(r)
        print(f"{y}: {len(r)} features ({time.time()-t0:.0f}s)", flush=True)
    ic = pd.concat(all_ic)
    ic.to_csv(os.path.join(FD, "factor_ic_by_year.csv"), index=False)

    fam = ic.groupby(["family", "year"])["IC"].mean().unstack()
    fam = fam.reindex(columns=years)
    fam.to_csv(os.path.join(FD, "family_ic_matrix.csv"))
    print("\n=== 因子族 × 年 平均 RankIC ===")
    print(fam.round(4).to_string())
    print("\nsaved:", os.path.join(FD, "family_ic_matrix.csv"))


def main():
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch_year(sys.argv[2])
    elif cmd == "labels":
        make_labels()
    elif cmd == "report":
        report()


if __name__ == "__main__":
    main()
