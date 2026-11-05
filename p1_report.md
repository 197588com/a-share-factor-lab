# P1 交付报告：因子注册表 + 批量测试管线 + 数据质量校验

日期：2026-11-03 ｜ 状态：✅ 完成（三大件全部跑通）

## 一、交付物清单

| 组件 | 位置 | 说明 |
|---|---|---|
| 因子注册表 + 测试库 | `factor_lab/pipeline.py` + `data/factor_lab.duckdb` | 4 张表：factor_registry / factor_test_results / regime_labels / ic_daily |
| 批量测试管线 | `pipeline.py run` | qlib 表达式引擎批量计算，config_hash 幂等、断点续跑，约 2-3 秒/因子 |
| 数据质量校验 | `factor_lab/data_quality.py` | 4 项自动检查，非 0 退出即报警，**当前 4/4 PASS** |

数据库已注册 22 个价量因子（状态机：candidate → validated/rejected），并落库规则版市场状态标签（bull/bear/shock/volatile 四类，2008-2022，阈值用 2008-2012 样本标定防偷看）。

## 二、22 因子测试结果（2010-2022，CSI300 动态成分，1 日前向收益）

五道门槛：G1 显著性（|t|>2）｜G2 regime 稳定性（最差状态同号）｜G3 池内相关<0.7（首批自动过）｜G4 IC 衰减检验｜G5 成本后毛利（Q5-Q1 年化 > 2×换手成本拖累）

### ✅ 唯一通过全部门槛：60日波动率（低波异象）

RankIC -0.0173、t=-4.33、日换手仅 1.34%、成本拖累 0.010/年 vs 分层年化 0.027——**低波股未来收益更高，且换手低到交易成本几乎不伤收益**。这与 A 股低波异象文献一致。

### 关键发现

1. **信号强度 ≠ 可盈利**。5日动量 RankIC 全场最高（+0.030，t=9.57），但日换手 15.85%，成本拖累 0.12/年，超过分层毛利 0.18 的一半——G5 拒。这是本批次最有价值的负结果：**短线反转信号真实存在，但被交易成本吃掉，必须以降换手方式改造**（平滑、叠加过滤条件、或延长持仓）。

2. **G5 是主要淘汰原因**：19 个 4/5 因子中绝大多数败在成本后毛利，而非信号本身。后续因子设计应把"预期换手"作为先验约束。

3. ** regime 条件化价值已验证**：Amihud 非流动性、60日波动率等在 shock（高波震荡）状态下 IC 明显衰减或反号——支持 P2 做 regime 条件化权重。

4. 经济学方向全部合理（短期反转正 IC、高波/高量价相关/彩票型负 IC），管线结果可信。

## 三、数据质量校验（本次交付新修并跑通）

数据曾停在 2026-10-30：当日（11-03）已收盘而库未更新，触发 freshness FAIL。本次新增 `update_qlib_data.py` 增量更新链路：

- 850 只历史成分全量补齐至 2026-11-03（含除权检测：窗口内 adj_factor 变化自动全量重抓，保证前复权口径一致）
- CSI300 成分区间表自动延长最新一期至月末（月度快照未出时的标准处理）
- dump_bin 重灌 + 日历/成分同步

当前 4/4 PASS：新鲜度（qlib=11-03=tushare 最近开盘日）、价格正值、成分覆盖 850、日历连续（长假断档识别为正常）。

## 四、使用说明

```bash
# 注册新因子（编辑 REGISTRY 后）
python factor_lab/pipeline.py run --start 22 --count 10      # 增量测试，幂等断点续跑
python factor_lab/pipeline.py run --start 0 --count 22 --force  # 强制重测
python factor_lab/pipeline.py report                          # 结果全表
python factor_lab/data_quality.py                             # 数据质量门禁（入库后必跑）
python update_qlib_data.py 0 850                              # 行情增量更新（每日收盘后）
```

## 五、P1 遗留 → P2 承接

- regime 规则版 v1 较粗糙（4 类规则）→ P2 升级 HMM/SSM 概率化状态识别
- 财务类因子需要 point-in-time 表 → P2 数据底座扩展
- 全市场（约 5400 只）未灌 → `fetch_qlib_data.py` 改 universe 逻辑即可扩
- ic_daily 表已落逐日 IC×regime，P2 直接做因子监控衰减报警
