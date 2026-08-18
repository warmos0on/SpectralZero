# 组层面：seen 训练长尾 vs 平衡采样收益 (Indian, 3-seed OA 均值)

| 组 | unseen 类 | seen 训练 Gini | seen 训练头尾比 | 基线 OA | D OA | ΔOA |
|----|----|----|----|----|----|----|
| G1 | Corn-notill, Hay-windrowed, Woods | 0.528 | 32.7x | 65.13 | 86.54 | +21.41 |
| G2 | Grass-trees, Buildings-Grass-Trees-Drives, Soybean-mintill | 0.473 | 19.0x | 86.46 | 90.77 | +4.31 |
| G3 | Grass-pasture-mowed, Stone-Steel-Towers, Corn-mintill | 0.465 | 32.7x | 84.76 | 86.86 | +2.10 |
| G4 | Corn, Alfalfa, Grass-pasture | 0.469 | 32.7x | 88.16 | 90.64 | +2.48 |
| G5 | Wheat, Oats, Soybean-notill | 0.478 | 32.7x | 78.92 | 89.03 | +10.11 |

- seen 训练 Gini 与 ΔOA 的 Pearson 相关：r=0.972（5 个组，样本少，仅作定性参考）
- 结论：若 r 为正且大，支持'长尾越严重→平衡采样收益越大'；否则机理不成立或需其他解释。