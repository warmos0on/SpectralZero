# 长尾分析结论汇总（Phase 1 证据链）

日期：2026-08-18
协议：Indian Pines，train_num=0.2，固定最后 checkpoint，3-seed(42/123/456) 均值。

## 一、长尾确实存在（量化证据）

- 训练集（seen）头尾比：17x（Woods 253 样本占 25% vs Oats/Alfalfa 各 15 样本占 1.5%）
- 全图像素头尾比：122.8x、Gini=0.5093、Shannon 熵=2.3262（均匀分布最大熵 2.7726）
- 结论：seen 类训练样本严重不均衡，模型对齐可能偏向头部类。

## 二、逐类分析（诚实结论：不支持"提升集中在尾部类"）

- 基线 per-AA vs 类样本数 logN：Pearson r=0.191（p=0.48）
- 最佳配置 D per-AA vs logN：r=0.264（p=0.32）
- 提升 Δ vs logN：r=0.166（p=0.54），Spearman r=0.076（p=0.78）
- **解读限制**：unseen 类不参与训练，其像素数只决定测试规模，并不直接度量 seen 长尾偏置；
  因此逐类相关性不能作为"平衡采样治长尾"的充分证据。
- 直觉观察（非统计结论）：Hay-windrowed(+45.4)、Soybean-notill(+11.2)、Corn-notill(+10.5)
  提升大，但 Woods(-9.2)、Stone-Steel-Towers(-8.9)、Oats(-3.3) 下降，方向不一致。

## 三、组层面分析（有正相关信号，但样本极少）

- 以每组 unseen 对应的 seen 训练集 Gini 为横轴、该组 ΔOA 为纵轴（5 组）：r=0.972
- G1（Gini 最高 0.528）收益最大 +21.41，G2（Gini 0.473）仅 +4.31
- **必须标注的限制**：
  1. 只有 5 个数据点，r 值不稳定，不能当强证据；
  2. G1 基线最低（65.13），收益大也可能是"基线低→上升空间大"的混淆，不是机理证明；
  3. 脚本里 head_tail 比值对 G3/G4/G5 显示相同值（32.7x），属脚本口径瑕疵，
     但 Gini 本身已区分组间差异，不影响相关计算。

## 四、对论文故事的建议

- 长尾故事线定位为"动机 + 机制假设"，用平衡采样收益 + 组层面趋势作为**支持性证据**，
  不要写成因果结论。
- 逐类散点图（fig_improv_vs_samples.png）展示真实分布，注明 r/p 值与样本数限制。
- ~~后续若想补强：可做 seen 侧消融（训练时人为制造/消除长尾）直接证明机制~~
  → 已执行，见第五节（2026-08-18）。

## 五、seen 侧长尾消融实验（Plan B，已完成）

协议：在最佳配置 D（关动态融合 + z-score + 平衡采样）基础上，用 `--train_cap` 只对
seen 类训练配额设上限，构造 4 档长尾水平（natural/mild/mid/uniform），
每档跑平衡采样开/关两臂，3-seed(42/123/456) 均值，固定最后 checkpoint。
明细见 `result_IP/longtail_ablation/summary.md` 与 `results.tsv`。

| 组 | level | seen Gini | 头尾比 | 平衡关 OA | 平衡开 OA | Δ(开−关) |
|----|-------|----------|--------|----------|----------|---------|
| G1 | natural | 0.528 | 32.7x | 62.56 | 86.54 | **+23.98** |
| G1 | mild | 0.387 | 10.0x | 65.18 | 84.55 | **+19.36** |
| G1 | mid | 0.297 | 6.0x | 72.90 | 77.71 | **+4.81** |
| G1 | uniform | 0.179 | 3.0x | 77.45 | 69.43 | **-8.02** |
| G5 | natural | 0.478 | 32.7x | 80.59 | 89.03 | **+8.44** |
| G5 | mild | 0.311 | 10.0x | 84.46 | 87.83 | **+3.37** |
| G5 | mid | 0.219 | 6.0x | 84.94 | 87.33 | **+2.39** |
| G5 | uniform | 0.135 | 3.0x | 82.85 | 84.54 | **+1.70** |

结论：
- Δ 与 seen Gini 的 Pearson 相关 r = 0.819（8 个点）；两组内部均随长尾减弱单调收窄，
  **支持"平衡采样收益来自 seen 长尾偏置"的机制假设**（长尾越重收益越大）。
- natural+平衡开 = 历史 D 配置均值（G1=86.54 / G5=89.03），未在本批次重跑，口径一致。
- G1 uniform 的 -8.02 主要由 seed42 异常值 45.51（vs seed123 84.55 / seed456 78.24）拉低，
  其余两个 seed 接近 0，如实标注，不宣称稳定回退。
- cap 同时压低了训练总量，绝对 OA 整体下降属预期代价；Δ 在同水平两臂间抵消总量效应，可比。

## 附：产出文件

- analyze_longtail.py / analyze_perclass.py / analyze_group_longtail.py
- 长尾分析.md（逐类表）、长尾分析_group.md（组表）
- fig_longtail_dist.png / fig_perclass_vs_samples.png / fig_improv_vs_samples.png
- 新增：make_longtail_caps.py / run_longtail_ablation.py / run_longtail_batch.py / analyze_longtail_ablation.py
- 新增：result_IP/longtail_caps/（3 个 cap 配置）、result_IP/longtail_ablation/（results.tsv / summary.md / delta_vs_gini.png）
