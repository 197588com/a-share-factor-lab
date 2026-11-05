# -*- coding: utf-8 -*-
"""每周 walk-forward 增量再训练（framework 1.4d：每周五推进一窗）。

每个运行周期做四件事：
1. HMM 滚动重估补算（regime_hmm.py refit，source=hmm_roll）
2. 按 walkforward.yaml 顺序推进 fold：每 fold 每模型每周最多补 epochs_per_run 个 epoch（断点续训）
3. 到达目标 epochs 的 fold 立即做测试段评估 + MLflow 记录 + 增量 DSR 审计
4. 上线判定（1.4d）：新 fold 样本外 RankIC 对比上一 fold，给出"维持/替换"决策

用法：
  python weekly_retrain.py            # 手动跑一个周期（幂等，随时可重入）
  python weekly_retrain.py --dry      # 只打印将要做什么，不动手

自动化入口见 daily_patrol 同款包装（blueprint automation assets/weekly_retrain_entry.py），
最终一行输出 ARTIFACT_JSON: {...} 供包装层解析。
"""
import os, sys, json, time, argparse
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import wf_pipeline as wf
import train_models as tm

STATE_PATH = os.path.join(WS, "data", "wf_state.json")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"folds": {}, "decisions": []}


def save_state(st):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def epochs_done(cfg, model, fold_id):
    ck = wf.ckpt_path(cfg, model, fold_id)
    if not os.path.exists(ck):
        return 0
    import torch
    st = torch.load(ck, map_location="cpu", weights_only=False)
    return int(st["epoch"]) + 1


def ic_csv_exists(model, fold_id):
    return os.path.exists(os.path.join(WS, "data", "wf_ic", f"{model}_fold{fold_id}.csv"))


def run(dry=False):
    cfg = wf.load_config()
    folds = wf.generate_folds(cfg)
    epr = cfg["retrain"]["epochs_per_run"]
    gate_end = pd.Timestamp(cfg["retrain"]["gate_csv_end"])
    st = load_state()
    progress, metrics_all = [], []

    # 1) HMM 滚动重估补算（幂等：已 refit 的年份自动跳过）
    if cfg["retrain"].get("hmm_roll", True):
        print("== HMM refit ==", flush=True)
        if not dry:
            import subprocess
            r = subprocess.run([sys.executable, os.path.join(WS, "factor_lab", "regime_hmm.py"), "refit"],
                               capture_output=True, text=True)
            for line in (r.stdout + r.stderr).splitlines():
                if "refit" in line and "INFO" not in line:
                    print("  " + line)

    # 2/3) 逐 fold 推进
    for fold in folds:
        te_end = pd.Timestamp(fold["test"][1])
        models = ["alstm"] + (["master"] if te_end <= gate_end else [])
        for model in models:
            key = f"{model}_fold{fold['id']}"
            target = cfg["models"][model]["epochs"]
            done = epochs_done(cfg, model, fold["id"])
            cache = os.path.join(wf.PREP_WF, f"fold_{fold['id']}.pkl")

            if done >= target and ic_csv_exists(model, fold["id"]):
                continue
            if dry:
                progress.append(f"{key}: epochs {done}/{target}, cache={'yes' if os.path.exists(cache) else 'NO'}")
                continue

            print(f"== {key} (epochs {done}/{target}) ==", flush=True)
            if not os.path.exists(cache):
                wf.prep_fold(fold, cfg)                      # Alpha158 fetch + 面板
            batch_to = min(target, done + epr)
            tr = wf.train_fold(cfg, model, fold, max_epochs=batch_to)
            wf.mlflow_log(cfg, model, fold, cache, train_res=tr)
            if batch_to >= target:
                ev = wf.eval_fold(cfg, model, fold)
                wf.mlflow_log(cfg, model, fold, cache, eval_res=ev)
                m = ev["metrics"]
                metrics_all.append({"model": model, "fold": fold["id"],
                                    "test": fold["test"], "rankic": m["RankIC"],
                                    "ic": m["IC"], "icir": m["ICIR"],
                                    "regime_ic": ev["regime_ic"]})
                st["folds"][key] = {"status": "evaluated", "rankic": m["RankIC"],
                                    "ic": m["IC"], "evaluated_at": time.strftime("%Y-%m-%d %H:%M")}
            else:
                st["folds"][key] = {"status": "training", "epochs_done": batch_to,
                                    "updated_at": time.strftime("%Y-%m-%d %H:%M")}

    # 4) 上线判定：同模型相邻 fold 样本外 RankIC 对比
    if metrics_all:
        for model in {m["model"] for m in metrics_all}:
            hist = [st["folds"].get(f"{model}_fold{f['id']}") for f in folds]
            vals = [(f["id"], h.get("rankic")) for f, h in zip(folds, hist) if h and h.get("rankic") is not None]
            new = [m for m in metrics_all if m["model"] == model]
            for m in new:
                prev = [v for fid, v in vals if fid < m["fold"]]
                if prev:
                    prev_v = prev[-1]
                    keep = m["rankic"] >= prev_v * 0.8   # 容忍 20% 衰减
                    dec = {"model": model, "fold": m["fold"], "new_rankic": m["rankic"],
                           "prev_rankic": prev_v, "decision": "replace" if keep else "hold_old",
                           "rule": "new >= 0.8 * prev"}
                    st["decisions"].append({**dec, "at": time.strftime("%Y-%m-%d %H:%M")})
                    metrics_all.append({"decision": dec})

    if not dry:
        save_state(st)
        dsr = {}
        for model in ("alstm", "master"):
            r = wf.audit(cfg, model)
            if r:
                dsr[model] = r["deflated_sharpe_ratio"]

    done_cnt = sum(1 for k, v in st["folds"].items() if v.get("status") == "evaluated")
    total_cnt = len(folds) * 2
    summary = (f"本周推进完成：已评估 {done_cnt}/{total_cnt} fold×模型组合；"
               f"本轮新评估 {len(metrics_all)} 项；" +
               ("DSR: " + json.dumps(dsr) if not dry else "dry-run"))
    artifact = {"summary": summary,
                "progress": {k: v for k, v in st["folds"].items()},
                "plan": progress,
                "newMetrics": metrics_all,
                "dsr": dsr if not dry else {},
                "decisions": st["decisions"][-5:],
                "ranAt": time.strftime("%Y-%m-%d %H:%M")}
    print("ARTIFACT_JSON: " + json.dumps(artifact, ensure_ascii=False))
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    run(dry=args.dry)
