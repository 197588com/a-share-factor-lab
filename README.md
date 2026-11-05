# A 股因子实验室（a-share-factor-lab）

一个面向 A 股的量化选股研究系统：以 walk-forward 时序验证为纪律，围绕「每日 top-10 选股信号 → 模拟账户结算 → 胜率回灌迭代」的闭环运转。

> 研究用途，不构成投资建议。

## 系统结构

```
数据层    tushare（日线/分钟/集合竞价/资金流/龙虎榜）→ qlib 二进制格式本地仓库（data/，不入库）
因子层    日频因子（moneyflow / margin / dragon 等）+ 分钟级 160 维 dragon 因子，增量计算
市场状态  HMM regime 门控（regime_gate.py），识别市场状态以适配选股方向
模型层    ALSTM 时序模型（master-repo 为 MASTER 架构参考实现），walk-forward 分 fold 训练与配对检验
信号层    live_infer.py 每日打分出 top-10 信号
模拟层    sim_v2 模拟账户（T+1 可执行口径：次日开盘价成交、停牌/跌停不可买处理）
调度层    daily_master.py 每日总流水线（数据更新 → 因子 → 推断 → 台账），由 Kimi Work 定时任务驱动
进化闭环  每日结算胜率并回灌，驱动因子与结构迭代（evolve_min.py / factor_lab/）
```

## 目录说明

| 路径 | 内容 |
| --- | --- |
| `daily_master.py` / `daily_chain.py` | 每日总流水线与任务链 |
| `min_factor.py` / `dragon_factor.py` / `moneyflow_factor.py` / `margin_factor.py` | 因子计算 |
| `minute_recorder.py` | 分钟/集合竞价行情本地自录 |
| `fetch_qlib_data.py` / `update_qlib_data.py` / `update_index_data.py` | 行情落库与增量更新 |
| `live_infer.py` / `live_ledger.py` / `live_sim.py` | 信号推断、台账、模拟账户 |
| `regime_gate.py` | HMM 市场状态门控 |
| `ckpt/` | 训练好的模型权重（fold 系列） |
| `master-repo/` | MASTER 模型架构参考代码 |
| `qlib-repo/` | qlib 工具（dump_bin.py 等） |
| `factor_lab/` | 因子质量与迭代实验 |
| `claude-trading-skills/` `qs-skills/` `stock-analyze-skills/` `stock-data/` | 投研技能参考库 |
| `p1_report.md` … `p4_report.md` `MODEL_CARD.md` `framework_design.md` `system_inventory.md` | 阶段报告与工程文档 |

## 运行前提

1. Python 依赖：qlib、torch、tushare、hmmlearn、duckdb、pandas 等。
2. 凭证：在项目根目录 `.env` 写入 `TUSHARE_TOKEN=<你的 token>`（代码经 `token_loader.py` 统一加载，亦支持环境变量）。
3. 行情：需自行通过 tushare 拉取并 dump 为 qlib 格式到 `data/`（见 `fetch_qlib_data.py`）。
4. 定时任务：总流水线由外部调度器（Kimi Work Automation）每日驱动，也可手动 `python daily_master.py`。

## 敏感信息说明

仓库已脱敏：源码中不含任何真实 token；`data/`（155G 行情）与 `.env`（凭证）不入库；`reference_t1overnight/` 为私有配置副本，整体排除。
