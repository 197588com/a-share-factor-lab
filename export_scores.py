# -*- coding: utf-8 -*-
"""导出 walk-forward 各 fold 模型的逐日逐股预测分数（P4 回测原料）。

从已训练的检查点（.best 最优权重）只做推理，不重训。
分数落两处：
  - data/scores/{model}_fold{i}.parquet（date, instrument, score）
  - DuckDB model_scores 表（trade_date, instrument, model, fold, score）

用法：
  python export_scores.py --model alstm [--folds 0-7]
"""
import os, sys, argparse
import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import wf_pipeline as wf
import train_models as tm

SCORE_DIR = os.path.join(WS, "data", "scores")
DB = os.path.join(WS, "data", "factor_lab.duckdb")


def export(cfg, model_name, fold):
    import torch
    from torch.utils.data import DataLoader
    blob = wf.load_fold(fold["id"])
    panels, G, meta = blob["panels"], blob["G"], blob["meta"]
    with_gate = model_name == "master"
    te = tm.DSWithDate(*panels["test"], G, wf.seg_indices(blob, "test"), with_gate)
    mc = cfg["models"][model_name]
    model = tm.ALSTM(hidden=mc.get("hidden", 64), num_layers=mc.get("num_layers", 2)).to(tm.DEVICE) \
        if model_name == "alstm" else tm.MASTER(d_model=mc.get("d_model", 128)).to(tm.DEVICE)
    ck = wf.ckpt_path(cfg, model_name, fold["id"])
    src = ck + ".best" if os.path.exists(ck + ".best") else ck
    st = torch.load(src, map_location=tm.DEVICE, weights_only=False)
    model.load_state_dict(st["model"])
    ps, ys, ds = tm.predict(model, DataLoader(te, batch_size=2048, shuffle=False))
    inst_idx = np.array([s[1] for s in te.samples])
    df = pd.DataFrame({"date": [meta["dates"][i] for i in ds],
                       "instrument": [meta["insts"][j] for j in inst_idx],
                       "score": ps})
    os.makedirs(SCORE_DIR, exist_ok=True)
    out = os.path.join(SCORE_DIR, f"{model_name}_fold{fold['id']}.parquet")
    df.to_parquet(out)
    import duckdb
    con = duckdb.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS model_scores(
        trade_date DATE, instrument VARCHAR, model VARCHAR, fold INTEGER, score DOUBLE,
        PRIMARY KEY(trade_date, instrument, model, fold))""")
    con.execute("DELETE FROM model_scores WHERE model=? AND fold=?", [model_name, fold["id"]])
    con.register("_df", df.rename(columns={"date": "trade_date"}))
    con.execute("INSERT INTO model_scores SELECT trade_date, instrument, ?, ?, score FROM _df",
                [model_name, fold["id"]])
    con.close()
    print(f"{model_name}_fold{fold['id']}: {len(df)} scores ({df['date'].nunique()}d) -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["alstm", "master"])
    ap.add_argument("--folds", default="0-7")
    a = ap.parse_args()
    cfg = wf.load_config()
    folds = wf.generate_folds(cfg)
    ids = set()
    for part in a.folds.split(","):
        if "-" in part:
            x, y = part.split("-"); ids.update(range(int(x), int(y) + 1))
        else:
            ids.add(int(part))
    for f in folds:
        if f["id"] in ids:
            export(cfg, a.model, f)
