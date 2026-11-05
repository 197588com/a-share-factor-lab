# P3 交付报告：walk-forward 深度模型自动再训练 + MLflow

日期：2026-11-03 ｜ 状态：**全部建成，16/16 fold×模型全量训练评估完成，DSR 审计通过**

## 〇、全量结果（2026-11-03 晚，6 轮推进约 3 小时跑完）

测试窗 2015→2022 逐年滚动，每 fold 样本外（一年）：

| fold | 测试年 | ALSTM IC / RankIC | MASTER IC / RankIC |
|---|---|---|---|
| 0 | 2015 | 0.0298 / 0.0345 | **0.0464 / 0.0456** |
| 1 | 2016 | **0.0337 / 0.0322** | 0.0203 / 0.0163 |
| 2 | 2017 | 0.0200 / 0.0173 | **0.0169 / 0.0335** |
| 3 | 2018 | **0.0285 / 0.0299** | 0.0252 / 0.0278 |
| 4 | 2019 | **0.0142 / 0.0161** | 0.0125 / 0.0096 |
| 5 | 2020 | 0.0175 / 0.0221 | **0.0193 / 0.0234** |
| 6 | 2021 | **0.0185 / 0.0100** | -0.0066 / 0.0029 |
| 7 | 2022 | **0.0073 / 0.0086** | 0.0022 / 0.0026 |
| **汇总** | | **meanIC 0.0212 · IR 0.196** | **meanIC 0.0170 · IR 0.120** |

**DSR 过拟合审计（n_trials=16 诚实计数）：ALSTM 1.0000 / MASTER 0.9997，双双通过 0.95 闸门。**

三个值得记住的规律：
1. **alpha 随时间衰减**：两模型 IC 都从 2015 的 0.03-0.05 衰减到 2022 的 0.002-0.009——市场对抗性的直接量化证据，也是 1.4a 因子/模型生命周期监控存在的理由
2. **ALSTM 在后期（2021-2022）明显强于 MASTER**（fold6/7 MASTER 已近失效）；早期（2015、2017）MASTER 反超——没有 universally 最优架构，walk-forward 逐窗对比替换是唯一可靠策略
3. **多训不一定更好**：fold0 ALSTM 最优 epoch（约 15ep）时 RankIC 0.0532，训满 20ep 后降到 0.0345；早停（best-epoch 权重）的价值在样本外直接兑现——管线已把"最新/最优"双检查点分开存盘

## 一、交付组件

| 组件 | 文件 | 说明 |
|---|---|---|
| 滚动窗口配置 | `walkforward.yaml` | 数据窗 2008-2022、训练6年/验证1年/测试1年/年步进 → 自动生成 8 个 fold（**切分由配置生成，禁止手写**，framework 1.3 硬约束） |
| walk-forward 管线 | `wf_pipeline.py` | prep（每 fold Alpha158 处理器只在 fold 训练窗 fit，防归一化泄漏）→ train（断点续训）→ eval（含 **regime 分组 IC**）→ MLflow 记录 → DSR 审计 |
| 实验追踪 | MLflow 3.17.2（sqlite + 本地 artifact 仓库） | 每次运行记录 config 哈希、数据快照 ID、超参、逐 epoch 指标、测试段 IC/ICIR/RankIC、**分 regime IC**、模型权重 |
| 过拟合审计 | `wf_pipeline.py audit`（复用 skill-backtest-overfit 的 DSR 实现，Bailey & López de Prado 2014） | 日度 IC 序列 → 给定试验次数（fold×模型×变体）的 deflated Sharpe；≥0.95 才算通过上线闸门（1.4c） |
| HMM 滚动重估 | `factor_lab/regime_hmm.py refit` | 每年初用此前全部数据重估，当年因果滤波解码；落 `source='hmm_roll'`，与生产标签 `hmm_v2` 并存不覆盖。**已跑 2015-2026 全 12 年**（P2 遗留项清零） |
| 每周编排 | `weekly_retrain.py` | HMM 补算 → 逐 fold 每周补 4 epoch（幂等断点续训）→ fold 完成即评估+MLflow+增量 DSR → 样本外 RankIC 对比给出维持/替换决策（容忍 20% 衰减） |
| 定时任务 | `automation_9a77335d`（**停用**待启用） | 每周五 20:13（Asia/Shanghai）纯 Python 执行，不耗对话额度，单次超时下次自动续 |

## 二、smoke 验证结果（P0 基准切分，ALSTM 仅 2 epochs）

| 指标 | P0 基线（10 epochs） | P3 smoke（2 epochs） |
|---|---|---|
| 测试段 IC / ICIR | 0.0068 / 0.051 | **0.0184 / 0.149** |
| RankIC / RankICIR | 0.0025 / 0.019 | **0.0107 / 0.087** |
| DSR（n_trials=2） | 未做 | **0.996（PASS）** |

分 regime IC（2021-2022 测试段）：quiet 0.0178（404 天）、bull 0.0219、bear 0.0213、shock 0.0117（1 天）——regime 分组评估链路打通，正式结论待 8-fold 全量。

注：smoke 复用 P0 缓存（fit 2015-2019），只验证管线正确性，不参与正式统计；smoke 的 IC 文件已清理。

## 三、与 P0/P2 的关系

- P0 的 `train_models.py` 作为模型/数据集/评估的单一事实源被 `wf_pipeline.py` import 复用，未 fork
- P2 遗留"HMM 滚动重估归 P3"已完成（hmm_roll 2015-2026 全覆盖，2026 实时段已解码）
- 训练框架仍是"线性因子打底 + 深度增强"双轨制（framework 1.3 模型族：ALSTM/MASTER 基线，Mamba 系替换归后续）

## 四、已知限制（诚实清单）

1. **MASTER 市场门控文件只到 2022-12-31**：fold 0-7 不受影响；fold 测试窗 >2022 的部分只训 ALSTM（已按 `gate_csv_end` 自动过滤）。门控文件延长后可放开
2. **2022 年 alpha 已接近噪声**（两模型 RankIC < 0.01）：2023+ 数据已入 qlib 但 Alpha158 面板与标签尚未延伸到 2023+（prep 窗口仍到 2022）——这是 P4 因子日更链路接通后的自然延伸
3. DSR 的 n_trials 默认 = fold×模型数（16），后续每试一个超参变体都应显式 `--trials` 报大数字（防自欺）
4. PBO/CSCV 未接（DSR 已覆盖核心闸门）；需要时 skill-backtest-overfit 的 `pbo_cscv.py` 可直接复用
5. 每周任务与全量推进共享「训练进度」小组件时同一 slot 只能有一个活动绑定，当前绑在全量推进上；若启用每周任务并想让小组件改吃每周数据，说一声我切换绑定

## 五、组件与启用方式

- 全量推进任务 automation_71b83a1d（手动触发）+ 每周再训练 automation_9a77335d（周五 20:13）+ 每日巡检 automation_ae1835c7（每交易日 17:07）：后两者**停用待启用**
- 「训练进度」小组件（widget_1f0f4e32）已挂「看板入门」，实时显示 16 格进度、DSR、样本外指标
- MLflow 查看：`cd 工作区 && mlflow ui`（sqlite 后端 + mlruns/ artifact 仓库）
