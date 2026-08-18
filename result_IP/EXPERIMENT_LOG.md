# 实验数据登记表（Indian Pines / LongKou 诚实评测）

协议：固定最后一个 checkpoint 的 OA/AA/Kappa；多组均值不含单类组（Indian G6 不计）。
标注：动态融合=开关动态调节；z-score=逐波段标准化；平衡采样=类别加权采样；负样本=unseen 类作负类。

## Indian Pines

| 编号 | 配置 | seeds | 结果文件 | OA | AA | Kappa | 状态 |
|------|------|-------|---------|-----|-----|-------|------|
| 基线(原版) | 融合ON，其余全关 | 42 | dynamic_seed42_20ep/All_Summary.txt | 80.88 | 70.67 | 0.5417 | ✅ |
| 基线3seed | 融合ON，其余全关 | 42,123,456 | ablation_base_3seed_run.txt | 80.69 | 68.65 | 0.5936 | ✅ 重跑成功 |
| 全改动 | 融合ON + 全部开 | 42,123,456 | evolved_multi_seed_run.txt | 81.95 | 63.97 | 0.5787 | ✅ |
| B | 融合ON + 只负样本 | 42 | ablation_B_negonly_run.txt | 77.50 | 58.72 | 0.4832 | ✅ |
| C | 融合ON + zscore + 平衡 | 42,123,456 | ablation_C_multi_seed_run.txt | 86.18 | 70.36 | 0.6553 | ✅ |
| D | 融合关 + zscore + 平衡 | 42,123,456 | ablation_D_multi_seed_run.txt | 88.77 | 75.01 | 0.7253 | ✅ 最佳 |
| E | 融合ON + 只zscore | 42,123,456 | ablation_E_zscore_only_run.txt | 78.66 | 66.95 | 0.5603 | ✅ |
| F | 融合ON + 只平衡 | 42,123,456 | ablation_F_balanced_only_run.txt | 85.09 | 74.02 | 0.6713 | ✅ |

## LongKou

| 编号 | 配置 | seeds | 结果文件 | OA | AA | Kappa | 状态 |
|------|------|-------|---------|-----|-----|-------|------|
| C | zscore开 + 平衡 + 融合ON | 42,123,456 | longkou_C_3seed_run.txt | 36.30 | 46.68 | 0.0815 | ✅(崩盘) |
| G1试点 | 关zscore | 42 | longkou_nozscore_g1_run.txt | 87.10 | 71.86 | 0.7329 | ✅ |
| G2试点 | 关zscore | 42 | longkou_nozscore_g2_run.txt | 26.72 | 62.59 | 0.0975 | ✅ |
| G2试点 | 关zscore + 关融合 | 42 | longkou_nozscore_nofusion_g2_run.txt | 25.13 | 46.15 | 0.0503 | ✅ |
| 完整验证 | 关zscore + 关融合 + 平衡采样 | 42,123,456 | longkou_nozscore_nofusion_3seed_run.txt | 54.35 | 57.02 | 0.3279 | ✅ |

## LongKou 逐组对比（3-seed，固定最后 cp；旧C = zscore开 + 平衡 + 融合ON）

| 组 | 类 | 旧C AA | 新配置 AA | ΔAA | 新配置 OA | 新配置每类准确率 |
|----|----|-------|-------|-----|-------|-------|
| G1 | Corn/Cotton/Water | 46.65 | 76.78 | **+30.1 ✅** | 92.14 | 89.2 / 41.2 / 99.9 |
| G2 | Sesame/Broad-leaf soybean/Rice | 63.54 | 61.00 | −2.5 ⚠️ | 44.09 | 52.9 / 33.9 / 96.3 |
| G3 | Narrow-leaf soybean/Mixed weed/Roads | 29.84 | 33.27 | +3.4 | 26.81 | 88.1 / 3.6 / 8.1 |

- G1 大胜：水类从 ~20% → 99.9%，z-score 破坏 LongKou 的 float32(0–28) 分布，关掉后水类几乎全对。
- G2 种子波动极大（seed42 OA 28.03 / seed123 45.91 / seed456 58.34），均值与旧C基本持平。
- G3 的 Mixed weed(3.6%) 与 Roads(8.1%) 在开/关 z-score 下都是个位数 → 数据本身难分，非本版改动引入。

## 归因结论（Indian，3-seed，固定最后cp；D vs 基线 = +8.08 OA，分解全部对账）

| 改动 | 对比 | OA 差 | AA 差 | 结论 |
|------|------|-------|-------|------|
| 类别平衡采样 | F − 基线 | **+4.40** | +5.37 | 最大单一功臣 ✅ |
| 关掉动态融合 | D − C | **+2.59** | +4.65 | 动态调节是净负，关掉更好 ✅ |
| z-score×平衡采样 协同 | C − (E+F+基线) | **+3.12** | — | 一起开有加成 ✅ |
| z-score 单独 | E − 基线 | −2.03 | −1.70 | 单独用反而小负 ⚠️ |
| unseen 负样本（组合中） | 全改动 − C | −4.23 | −6.39 | 负贡献 ❌ |
| unseen 负样本（单独） | B − 基线(seed42) | −3.38 | −11.95 | 负贡献 ❌ |

对账：+4.40 − 2.03 + 3.12 + 2.59 = +8.08 = D(88.77) − 基线(80.69) ✓

## 备注
- 2026-08-05 18:15 起：发现 3 个实验并行会让 RTX 4060 显存冲突（CUDA: illegal memory access / unspecified launch failure），改为最多 2 个并行。
- 基线3seed 首跑因并行冲突在 G1 seed42 崩溃，19:48 重跑成功。

| ???? | ?zscore+???+?? + ??????? | 42,123,456 | longkou_compact_3seed_run.txt | 46.54 | 53.22 | 0.1832 | ? ??????? |
