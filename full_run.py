# -*- coding: utf-8 -*-
"""全量 walk-forward 推进器：一次运行把尽可能多的 fold×模型 组合推到完成。

与 weekly_retrain.py 的关系：
- 共享 data/wf_state.json 进度状态和 MLflow 实验（同一套 artifact 结构）
- weekly 是"每周每组合补 4 epoch"的常设节奏；本脚本是一次性全量冲刺，
  每个组合直接训到目标 epochs → 评估 → 入 MLflow → 增量 DSR
- 内部有时间预算（默认 33 分钟），到预算就干净退出（每个 epoch 都存检查点，
  下一次运行无缝续跑），供定时任务反复触发直至 16/16 完成

用法：
  python full_run.py                 # 一个推进周期（受预算限制）
  python full_run.py --budget 3300   # 自定义预算秒数
  python full_run.py --dry           # 只看计划
"""
import os, sys, json, time, argparse
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)
import wf_pipeline as wf
from weekly_retrain import load_state, save_state, epochs_done, ic_csv_exists, STATE_PATH


def run(budget=1980, dry=False):
    t_start = time.time()
    cfg = wf.load_config()
    folds = wf.generate_folds(cfg)
    gate_end = pd.Timestamp(cfg["retrain"]["gate_csv_end"])
    st = load_state()
    metrics_all, dsr = [], {}

    # HMM 滚动重估补算（幂等）
    if cfg["retrain"].get("hmm_roll", True) and not dry:
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(WS, "factor_lab", "regime_hmm.py"), "refit"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "refit" in line and "INFO" not in line:
                print("  " + line, flush=True)

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
                print(f"todo {key}: epochs {done}/{target}, cache={'yes' if os.path.exists(cache) else 'NO'}")
                continue
            if time.time() - t_start > budget:
                print(f"budget reached, stop before {key}")
                break

            print(f"== {key} (epochs {done}/{target}) ==", flush=True)
            if not os.path.exists(cache):
                wf.prep_fold(fold, cfg)
            tr = wf.train_fold(cfg, model, fold, max_epochs=target)   # 直接训到目标
            wf.mlflow_log(cfg, model, fold, cache, train_res=tr)
            ev = wf.eval_fold(cfg, model, fold)
            wf.mlflow_log(cfg, model, fold, cache, eval_res=ev)
            m = ev["metrics"]
            metrics_all.append({"model": model, "fold": fold["id"], "test": fold["test"],
                                "rankic": m["RankIC"], "ic": m["IC"], "icir": m["ICIR"],
                                "regime_ic": ev["regime_ic"]})
            st["folds"][key] = {"status": "evaluated", "rankic": m["RankIC"], "ic": m["IC"],
                                "evaluated_at": time.strftime("%Y-%m-%d %H:%M")}
            save_state(st)
        else:
            continue
        break   # 预算耗尽，跳出 fold 循环

    if not dry:
        for model in ("alstm", "master"):
            r = wf.audit(cfg, model)
            if r:
                dsr[model] = r["deflated_sharpe_ratio"]

    done_cnt = sum(1 for v in st["folds"].values() if v.get("status") == "evaluated")
    total_cnt = len(folds) * 2
    summary = (f"全量推进：已评估 {done_cnt}/{total_cnt} fold×模型；"
               f"本轮新完成 {len(metrics_all)} 项；DSR: {json.dumps(dsr) if dsr else '待首个 fold 完成'}")
    artifact = {"summary": summary, "progress": dict(st["folds"]), "plan": [],
                "newMetrics": metrics_all, "dsr": dsr,
                "decisions": st["decisions"][-5:],
                "ranAt": time.strftime("%Y-%m-%d %H:%M")}
    print("ARTIFACT_JSON: " + json.dumps(artifact, ensure_ascii=False))
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1980, help="本轮预算秒数（默认 33 分钟）")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    run(budget=args.budget, dry=args.dry)
