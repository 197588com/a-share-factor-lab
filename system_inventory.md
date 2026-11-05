# A 股因子实验室 · 系统工程详单

> 版本：2026-11-05 ｜ 工作区：`/Users/yanglin/Documents/kimi/tasks/2026-11-03/13-01-52-a8ddd0ae`
> 配套文档：`framework_design.md`（P0–P9 研究全记录，本详单的结论出处）

---

## 1. 系统定位与设计哲学

一句话：**以"证伪优先"为纪律的 A 股选股研究系统**——先建立不可过拟合的评估底座，再用三层闸门筛选一切增量，宁可错过不可错信。

三条贯穿性设计决策：

1. **对抗性假设**：市场是非平稳、自适应的对手。任何改进必须假设自己会过拟合，评估协议（跨年 walk-forward + 配对检验 + 预注册判定标准）先于实验设计存在，结果出来后不许改标准。
2. **三层闸门**：单因子诊断 → 模型增量 → 组合构建，一关不过就换层，禁止"换个权重再试试"式的同层反复挖掘（P9 即按此纪律两刀即停）。
3. **研究/运维同构**：所有研究产物（分数、检查点、因子面板）直接复用为线上信号链路的原料，研究与实盘零口径差。

## 2. 架构总览

```
┌─ 运维层 ──────────────────────────────────────────────┐
│ 每日巡检 17:07 → 每日信号 18:23 → 每周重训 周五 20:13     │
│      (quality gate)      (top10 widget)  (增量 epochs)  │
├─ 信号层 ──────────────────────────────────────────────┤
│ live_infer.py 两阶段：bootstrap(全量拟合) / daily(增量)   │
│   → ckpt/wf/alstm/fold_11.pt.best + 处理器 pickle        │
│   → artifact schema → Binding → 看板「看板入门」widget    │
├─ 组合层 ──────────────────────────────────────────────┤
│ 基线：满仓 top50 + TopkDropout(n_drop=5)（P4 定案）      │
│ P7 regime 门控 ❌ ｜ P9 信号融合(静态/自适应) ❌          │
├─ 模型层 ──────────────────────────────────────────────┤
│ ALSTM(hidden=64×2) 主模型，12 fold walk-forward          │
│ MASTER 对照轨（门控数据 2022 截止）｜ HGB 线性对照轨        │
│ 增量评审：increment_test.py + wf_dragon.py 终审器         │
├─ 特征层 ──────────────────────────────────────────────┤
│ Alpha158（qlib，RobustZScore 处理器防泄漏拟合）            │
│ + 分钟微观结构（IVOL/HEADR，P8 资产）                     │
│ + 候选族：margin/dragon/mf/corr/min（观察名单）            │
├─ 数据层 ──────────────────────────────────────────────┤
│ qlib cn_data（Tushare 前复权自建）+ factor_lab.duckdb    │
│ + 分钟线 14,720 块（2015–2026）+ prep_wf 训练面板 99G      │
└─────────────────────────────────────────────────────────┘
```

## 3. 数据资产清单（107G）

| 资产 | 体量 | 说明 |
|---|---|---|
| `data/prep_wf/` | 99G | 12 fold × 模型训练面板缓存（Alpha158 处理后张量，断点续训直接复用） |
| `data/min/` | 2.7G | **320 股 × 46 块 × 5min 线，2015–2026 全覆盖**（Tushare stk_mins，P8 资产，可复用于任何日内研究） |
| `data/prep/` | 2.5G | Alpha158 全跨度特征缓存 |
| `data/factor_decay/` | 795M | 因子衰竭诊断面板 |
| `data/live/` | 499M | 线上信号链路近窗特征 + fit_parts + 各实验 increment/wf 结果 JSON |
| `data/factor_lab.duckdb` | 215M | 因子注册表与分析库，8 表：`factor_registry` / `factor_test_results` / `ic_daily` / `factor_alerts` / `factor_regime_stats` / `model_scores` / `regime_labels` / `regime_probs` |
| `data/csv/` + `data/csv_index/` | 200M | qlib dump 源 CSV（前复权口径） |
| `data/moneyflow/` `dragon/` `margin/` | 246M | P6 候选因子原始数据（龙虎榜/资金流/两融） |
| `data/scores/` | 9.4M | 各 fold 模型逐日逐股预测分数（回测原料，20 文件） |
| `data/wf_ic/` | 664K | 基线与各候选组的逐日 IC 序列（配对检验直接 join） |
| `data/backtest/` | 68K | P7/P9 组合层实验结果 CSV |
| `ckpt/` | 50M | 模型检查点（见 §5） |

**数据流口径**：Tushare 前复权日线 → `fetch_qlib_data.py` 转 qlib CSV → `update_qlib_data.py` 每日增量 + 除权检测 → qlib `cn_data` provider → 全链唯一行情源。分钟线为独立通道（`min_factor.py`，断点续传拉取器）。

## 4. 核心代码模块（28 脚本，按层）

### 数据层
| 脚本 | 职责 |
|---|---|
| `fetch_qlib_data.py` | Tushare 拉 CSI300 日线 → qlib dump CSV（前复权） |
| `update_qlib_data.py` | 每日增量补齐 + adj_factor 除权事件检测 |
| `prep_data.py` | Alpha158 分段缓存（训练面板原料） |
| `fit_parts.py` | bootstrap 计算拆解：分年拉特征合成拟合处理器（每步 <300s） |

### 特征/因子层
| 脚本 | 职责 |
|---|---|
| `factor_decay.py` | Alpha158 因子族 2015–2026 衰竭诊断（P1 注册表口径） |
| `min_factor.py` | 分钟线 5 因子日度聚合（IVOL/HEADR/TAILR/PMD/TCONT） |
| `margin_factor.py` / `dragon_factor.py` / `moneyflow_factor.py` | P6 三单候选因子构造 |
| `corr_enhance.py` | corr 族相关结构增强因子 |

### 模型层
| 脚本 | 职责 |
|---|---|
| `wf_pipeline.py`（19K，核心引擎） | walk-forward 切分/训练/评估/MLflow 记录，fold 由 `walkforward.yaml` 推导 |
| `wf_dragon.py` | **通用增量终审器**：`WF_GROUP=<组名> build|train|eval --folds N`（已泛化，任何新特征组注册即用） |
| `increment_test.py` | 单 fold 快速增量筛选（终审前的廉价预检） |
| `train_models.py` | ALSTM/MASTER 模型定义与训练循环（torch MPS） |
| `gbdt_baseline.py` | 线性/树对照轨（双轨制，防深度学习幻觉） |
| `export_scores.py` | 各 fold 检查点推理出逐日分数（不重训） |

### 组合/回测层
| 脚本 | 职责 |
|---|---|
| `backtest_p4.py` | **主回测引擎**：qlib TopkDropout 逐日撮合，含 T+1/涨跌停/停牌/双边费用（0.0005+0.0015），`run_bt()` 为全项目统一入口 |
| `crosscheck_vbt.py` | vectorbt 快速交叉核对（数量级验证主引擎） |
| `regime_gate.py` | P7 regime 暴露门控七场景实验器 |
| `blend_test.py` | P9 信号融合（静态权重 + 60 日自适应门控，20 场景） |

### 信号/运维层
| 脚本 | 职责 |
|---|---|
| `live_infer.py` | 线上信号：bootstrap（训练窗拟合处理器，防泄漏）/ daily（近 90 日增量，分钟级）两阶段 |
| `verify_live.py` | daily 路径特征 vs fold 测试面板逐值交叉核对（口径一致性守门） |
| `daily_chain.py` | 每日收盘巡检链：交易日判断→行情增量→灌库→质量门禁→regime 刷新→因子监控 |
| `weekly_retrain.py` / `full_run.py` | 每周增量再训练（每 fold 补 4 epochs 断点续训）/ 全量推进器 |

## 5. 模型资产（ckpt/）

| 目录 | 内容 |
|---|---|
| `ckpt/wf/alstm/` | **主模型**：12 fold × (.pt 末轮 + .pt.best 最优)，8.6M |
| `ckpt/wf/master/` | MASTER 对照轨 11M |
| `ckpt/wf/hgb_fold11.pkl` | HGB 线性对照 |
| `ckpt/wf/alstm_{margin,dragon,corrx,min}/` | P6/P8 候选组检查点（**已证伪，仅留档**，不进线上） |

线上信号唯一取 `alstm/fold_11.pt.best`（最新 fold，训练窗 2015–2020，验证 2025，测试 2026）。

## 6. 研究结论资产

`framework_design.md`（26K）为单一事实源，P0–P9 每阶段含判定 + 关键数字 + 产物清单。核心沉淀：

- **基线定案（P4）**：ALSTM + Alpha158 + 满仓 top50，样本外净超额 2023 +6.8% / 2024 +17.3% / 2025 +4.6% / 2026 +1.7%（IR 0.92/2.01/0.56/0.12）。
- **五单证伪（P6/P8）**：两融/龙虎榜/资金流/corr/分钟因子全部未过模型增量闸门（12 年池化配对 p≥0.50）。
- **组合层两刀（P7/P9）**：regime 门控、信号融合（静态+自适应）全部未过跨年稳定性闸门。
- **系统级结论**：公开日频/日内数据维度下基线已饱和；增量只能来自新数据维度（level-2/逐笔/舆情 NLP）或新范式（生成式 regime、RL 执行）。
- **方法论四条**：①单 fold 转正不可信，一律全量 WF + 配对检验；②单因子 IC 稳 ≠ 模型增量；③门控类改进须跨年稳定性验收；④强单因子出口在组合层——P9 后修正为"组合层也走不通，三层闸门闭环"。

## 7. 运维自动化（三件套，均已启用）

| 任务 | cron（Asia/Shanghai） | 职责 |
|---|---|---|
| 每日巡检 `automation_ae1835c7…` | `07 17 * * *` | 数据质量门禁 + regime 刷新 + 因子监控（红线告警） |
| 每日选股信号 `automation_371db7f2…` | `23 18 * * 1-5` | fold11 ALSTM → top10 信号 → 「看板入门」widget |
| 每周重训 `automation_9a77335d…` | `13 20 * * 5` | HMM 重估 → 逐 fold 补 4 epochs → 达标 fold 评估 + DSR 审计 + 维持/替换决策 |

**链路衔接**：巡检(17:07) 先保证数据干净 → 信号(18:23) 消费干净数据 → 重训(周五) 原地更新 fold 检查点 → 下周一巡检验证。widget `widget_644cce7c…`（看板入门 x0 y46）经 `binding_fc191c60…` 收 artifact，schema 契约见 automation 定义。

## 8. 关键配置（walkforward.yaml）

- 切分：每 fold 训 6 年 / 验 1 年 / 测 1 年，年步进，共 12 fold（2008–2026），**切分由配置生成，禁止手写**
- 标签：`Ref($close,-2)/Ref($close,-1)-1`（T+1 可交易口径）
- 处理器：RobustZScoreNorm（clip outlier）+ Fillna，仅在训练窗拟合（防泄漏）
- 重训粒度：`epochs_per_run: 4`，断点续训；MASTER 门控数据 2022 截止，此后仅 ALSTM 可训

## 9. 扩展指南（如何动这个系统）

**新增一个因子族进模型（完整流程）**：
1. 构造因子 → `increment_test.py` 注册新特征组（单 fold 预检，<10 分钟）
2. 预检有戏 → `WF_GROUP=<组名> python3 wf_dragon.py build`（注册进 158 维面板）
3. `WF_GROUP=<组名> python3 wf_dragon.py train --folds 12`（逐 fold 断点续训，~4s/epoch·fold on MPS）
4. `WF_GROUP=<组名> python3 wf_dragon.py eval --folds 12` → 与基线 IC 序列配对检验（判定标准事先写入实验计划）
5. 过闸 → 重跑 `export_scores.py` + `backtest_p4.py` 组合层复核；不过 → 观察名单留档

**新增组合层改进**：先用 `blend_test.py` / `regime_gate.py` 的预注册模式写判定标准，再跑实验。

**线上信号改动后必做**：`verify_live.py` 逐值核对 → 手动 run 一次信号任务看 artifact → 才动调度。

## 10. 已知边界

- 2026 为不完整年（数据至 11-03，约 192 交易日），跨年比较须注意分母。
- 分钟数据仅覆盖 CSI300 的 320 只历史成分（含已调出），非全市场。
- Tushare 免费额度：日线/分钟拉取有频控，重训与巡检已错开峰值；stk_mins 权限（历史分钟+集合竞价）为独立付费项，当前有效。
- 系统对"市场状态突变"的响应以周为单位（重训粒度），日内级 regime 适应不在当前边界内。
