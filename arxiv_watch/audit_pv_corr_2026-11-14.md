# 自查卡：量价相关构造口径审计（线索 9 第一项）

- 执行日期：2026-11-14
- 来源线索：INDEX 线索 9（因子构造的口径审计）
- 文献锚点：Price-Volume Correlations Are Always Zero（arXiv 2611.08528，digest 2026-11 第二期 P2-8）
- 性质：只读自查，不改代码、不动模型

---

## 论文攻击的形式（判定基准）

在**单个平均区间内**用逐笔成交的 (价格, 量) 序列计算相关：区间聚合价格是 VWAP（量加权），与区间内各笔价格构造性耦合，导致按定义计算的价格-成交量相关**恒等于零**；文献中报告的非零值实为「VWAP 与简单均价之差」的伪影。论文建议改用 corr(price, volume²) 类口径。

**触发条件（两个同时满足才踩中）**：① 区间内逐笔/分钟级序列；② 聚合价格经 VWAP（量加权）进入计算。

## 全项目量价相关构造清单（逐一核对）

| # | 构造 | 位置 | 计算口径 | 逐笔+区间内 VWAP？ | 结论 |
| --- | --- | --- | --- | --- | --- |
| 1 | CORR5/10/20/30/60 | Alpha158 内嵌（factor_decay.py corr 族） | Corr($close, Log($volume+1), N)，跨日 bar Pearson | 否：跨 bar，每点为独立 bar 的 close（最后成交价） | 不受影响 |
| 2 | CORD5/10/20/30/60 | 同上 | Corr(日收益, Log 量比, N)，跨日 bar | 否：同上 | 不受影响 |
| 3 | CC20 / CC20TS / CCSPREAD / RC20 | corr_enhance.py（corrx 族） | 跨日 bar close × log(volume)，含时序中性化与一阶差分变体 | 否：复刻 Alpha158 口径 | 不受影响 |
| 4 | corr_pv_20 / corr_pv_60 | factor_lab/pipeline.py | Corr($close, $volume, N)，跨日 bar，原始量未取 log | 否：跨 bar | 不受影响（但见附注 A） |
| 5 | pv_corr | evolve_min.py（分钟族） | corrcoef(5min bar 收益, Δlog 5min bar 量)，日内跨 bar | 否：跨 bar，无区间内 VWAP 聚合 | 不受影响 |

辅助确认：Alpha158 的 VWAP 组（factor_decay.py price 族，VWAP0 = $vwap/$close）是价格水平特征，不是量价相关构造，不在本论文攻击范围。

## 一行结论

**项目内全部 5 处量价相关构造均为跨 bar 口径（bar close × bar 量），无一落入「区间内逐笔 + VWAP 聚合」的恒零伪影形式；且 corr 族实证 12 年稳定非零 IC（−0.003 ~ −0.029，2026 仍 −0.0205）本身即是反证——构造性恒零下不可能有存活信号。自查通过，无需改动。**

## 附注（自查副产品，非本论文问题，不触发动作）

- **A. 冗余提示**：factor_lab 的 corr_pv_20/60 与 corrx 族 CC20 仅差 volume 是否取 log，近乎重复；且原始 $volume 重尾分布下 Pearson 对极端量敏感，鲁棒性弱于 log 口径。若日后精简因子池可优先考虑去重，属正常因子治理，不在本次审计动作内。
- **B. 新构造纪律**：若未来按论文建议尝试 corr(price, volume²) 类新因子，按线索 1 纪律走 candidate_ledger 准入 + 一次性判决，不属本次自查范围。
