# SpectralZero 创新研究大纲与计划总览

## 一、核心故事线（给 Introduction 用）

**问题**：高光谱零样本分类中，seen 类训练样本呈严重长尾分布（Indian Pines 头尾比 17倍，
Woods 占 25% 而 Oats/Alfalfa 仅 1.5%）。模型学到的图像-文本对齐偏向头部类，
导致尾部 seen 类表征不足，进而影响向 unseen 类的迁移能力。

**我们的思路**：从"数据分布偏置"入手，而非单纯改模型结构。
1. 类别平衡采样——直接对抗长尾，让尾部类获得等量训练曝光
2. z-score 标准化——消除波段间量纲差异，与平衡采样协同生效
3. 回退固定权重融合——动态融合实测净负，简化反而更好

**证据链**：
- 长尾量化分析（头尾比 17x）→ 问题确实存在
- 消融实验：平衡采样单独 +4.40 OA（最大功臣），z-score×平衡协同 +3.12
- 逐种子验证：3 个种子全部提升（+7.44 / +11.89 / +4.91），非运气

## 二、已完成的工作（按时间线）

| 阶段 | 改动 | Indian OA 变化 | 结论 |
------|------|---------------|------|
| 基线（复现） | 论文原始代码 | 80.69 | 起点 |
| 动态融合 | 光谱空间动态融合+负样本+z-score+平衡 | 81.95（四样全开） | 融合+负样本拖后腿 |
| 消融C | z-score+平衡采样（关负样本，融合开） | 86.18 | 预处理有效 |
| **消融D（最终）** | **关融合+z-score+平衡采样** | **88.77** | **+8.08，最佳** |
| 归因 | 逐因子隔离验证 | 对账 +8.08 | 平衡采样主功臣 |
| Phase2 试点 | 物理属性词表 v2（替换文本） | **74.85（vs 88.77）** | **-13.92，净负，已回滚终止** |
| Plan B 消融 | seen 侧长尾操控（train_cap，4 档水平×2 臂×3-seed） | Δ vs Gini r=0.819 | 机制闭环：平衡采样收益来自 seen 长尾偏置 |

## 三、待完成的实验计划

### Phase 1: 长尾分析与证据补全（2-3天）
- [x] 长尾分布量化（已完成，analyze_longtail.py）
- [x] 逐类准确率 vs 样本数相关性分析（analyze_perclass.py）
  - 诚实结论：逐类层相关性不显著（基线 r=0.191 p=0.48；D r=0.264 p=0.32；Δ 无显著趋势），
    不支持"提升集中在尾部类"的简单说法；unseen 类样本数只决定测试规模，不是 seen 长尾偏置的直接度量
- [x] 组层面 seen 训练长尾 vs 收益分析（analyze_group_longtail.py）
  - seen 训练 Gini 与 ΔOA 的组间相关 r=0.972（仅 5 组，只能定性参考；高收益组基线低，有上升空间混淆）
- [x] 长尾指标计算（entropy / Gini / effective number of classes，见 长尾分析.md）
  - Gini=0.5093、Shannon 熵=2.3262/最大熵 2.7726、头尾比 122.8x（全图）/ 17x（训练集）
  - 产出物：result_IP/长尾分析.md、长尾分析_group.md、3 张 PNG 图
- [x] **seen 侧长尾消融（Plan B，2026-08-18 已完成）**
  - 用 `--train_cap` 构造 4 档长尾水平（natural/mild/mid/uniform），每档跑平衡采样开/关两臂
  - 结果：Δ 与 seen Gini 相关 r=0.819；两组内 Δ 随长尾减弱单调收窄（G1: +23.98→+19.36→+4.81→-8.02；G5: +8.44→+3.37→+2.39→+1.70）
  - G1 uniform 的 -8.02 系 seed42 异常点（45.51）拉低，已如实标注，不宣称稳定回退
  - 结论：平衡采样收益来自 seen 长尾偏置，机制故事线闭环
  - 产出物：result_IP/longtail_ablation/（results.tsv / summary.md / delta_vs_gini.png）、result_IP/longtail_caps/

### Phase 2: 文本属性创新（3-5天）
- [x] 调研高光谱文本引导的属性词表方向（结合论文 Table II 与领域知识；DeepSeek 方向：物理/地理/光谱特征）
  - 落地为 4 个物理维度：spectral_reflectance / phenology / soil_background / canopy_structure
  - 词表文件：attributes/Indian_class_attributes_physical_v1.json → v2（按 CLIP 原型可分性诊断修正）
- [x] 重写物理属性词表（v2 已激活，原自然属性归档为 Indian_class_attributes_v1_natural.json）
- [x] CLIP 原型可分性快速校验（check_text_embeddings.py）
  - G1/G4/G5 原型比自然属性更可分，G2/G3 略差；token 上限留有余量（max 73/77）
- [x] Indian 3-seed 试点（honest eval 对比 D=88.77，已结束（2026-08-18））
  - 结果：OA 74.85 vs D 88.77（**-13.92**），逐类严重极化（Alfalfa +32.7 / Wheat +43.4 大涨，Oats -86.7 / Grass-pasture -64.4 / Hay-windrowed -63.3 崩塌）
  - **结论：净负贡献，物理词表方向终止**。已回滚激活词表为自然版，详细记录见 `result_IP/Phase2_physical_vocab_result.md`
  - 原型可分性诊断与实测不符（预测 G1/G4/G5 更好却实际崩塌），小成本校验不能替代真实训练试点
- [x] 注意：CLIP 文本空间对遥感词汇可能不敏感，已通过小成本校验 + 试点验证

### Phase 3: 跨数据集验证（2-3天）
- [x] LongKou：已有 3-seed 结果（OA 54.35），G1 大胜/G2 持平/G3 数据级难点，口径与旧基线不同，不宣称提升
- [x] Houston：因缺 Houston.mat（仅 gt，磁盘无备份）本轮跳过，待用户提供数据后补跑——已登记阻塞项
- [x] 整理三数据集对比表：result_IP/cross_dataset_comparison.md（含已否决方向登记）

### Phase 4: 论文素材整理（2天）
- [x] 长尾分布图（fig_longtail_dist.png）
- [x] 逐类准确率提升图（fig_perclass_vs_samples.png / fig_improv_vs_samples.png）
- [ ] 消融对比表（已完成，数据改变对比表.md）
- [x] Introduction 初稿 v1（result_IP/paper_materials/intro_draft.md，含限制与已否决方向）

## 四、风险与诚实原则

- 所有数字来自原始实验输出，不造假
- 文本创新可能无效（LongKou 压缩属性实验已证明改文本有风险）
- **已证实：物理属性词表替代自然词表 = 净负 -13.92（74.85 vs 88.77），方向终止（见 result_IP/Phase2_physical_vocab_result.md）**
- 动态融合已证明是负贡献，论文中需诚实说明或不再作为创新点
- LongKou 提升有限，需诚实说明数据级难点

## 五、当前推荐优先级（Phase 2 结束后更新）

1. **Phase 3 跨数据集验证**——把已定稿的数据级创新（平衡采样 + z-score）铺到 LongKou / Houston，补齐三数据集对比表
2. **Phase 4 论文素材**——长尾分布图、逐类提升图、消融对比表、Introduction 初稿（围绕"数据分布偏置"故事线）
3. ~长尾机理解释（seen 类长尾偏置 → unseen 迁移影响）~ 已完成（Plan B 消融 r=0.819，见 Phase 1）
