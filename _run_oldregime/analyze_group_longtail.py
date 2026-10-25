# -*- coding: utf-8 -*-
"""Phase 1 补充：组层面 seen 长尾程度 vs 该组收益

机理：平衡采样作用于 seen 训练分布；若长尾→偏置是真因，则
每组 seen 训练分布越失衡，开平衡采样(+zscore, D 配置)的收益应越大。
"""
import json, sys, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import datasets, utils

GROUPS = [
    ["Corn-notill", "Hay-windrowed", "Woods"],
    ["Grass-trees", "Buildings-Grass-Trees-Drives", "Soybean-mintill"],
    ["Grass-pasture-mowed", "Stone-Steel-Towers", "Corn-mintill"],
    ["Corn", "Alfalfa", "Grass-pasture"],
    ["Wheat", "Oats", "Soybean-notill"],
]

# 3-seed 均值 OA（固定最后 cp），来自 ablation summary
GROUP_OA = {
    "baseline": [65.13, 86.46, 84.76, 88.16, 78.92],
    "D_best": [86.54, 90.77, 86.86, 90.64, 89.03],
}


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float(((2 * np.arange(1, n + 1) - n - 1) * x).sum() / (n * x.sum()))


def main():
    cfg = json.load(open("./config/Indian.json", encoding="utf-8-sig"))
    HSI, gt, label_values = datasets.get_dataset(dataset_name="Indian")
    # 全类像素数
    total = {}
    for name, cid in label_values.items():
        total[name] = int(np.sum(gt == cid))

    lines = ["# 组层面：seen 训练长尾 vs 平衡采样收益 (Indian, 3-seed OA 均值)\n",
             "| 组 | unseen 类 | seen 训练 Gini | seen 训练头尾比 | 基线 OA | D OA | ΔOA |",
             "|----|----|----|----|----|----|----|"]
    deltas = []
    ginises = []
    for gi, unseen in enumerate(GROUPS, start=1):
        seen_names = [n for n in label_values if n not in unseen]
        seen_label = {n: label_values[n] for n in seen_names}
        seen_gt, _ = utils.fix_label(seen_label, gt)
        train_num, _ = utils.get_train_test_num(gt=seen_gt, train_num=cfg["train_num"])
        # 每类训练量 = min(该类像素数, train_num)（与 split_gt 一致）
        train_counts = []
        for idx, name in enumerate(seen_names):
            n_train = min(total[name], int(train_num[idx]))
            train_counts.append(n_train)
        train_counts = np.array(train_counts, dtype=float)
        g = gini(train_counts)
        # 头尾比：排除 0（unseen 类在 seen_gt 里为 0；这里 seen_names 已排除）
        head_tail = train_counts.max() / train_counts.min() if train_counts.min() > 0 else float("nan")
        base_oa = GROUP_OA["baseline"][gi - 1]
        d_oa = GROUP_OA["D_best"][gi - 1]
        delta = d_oa - base_oa
        deltas.append(delta)
        ginises.append(g)
        lines.append(f"| G{gi} | {', '.join(unseen)} | {g:.3f} | {head_tail:.1f}x | "
                     f"{base_oa:.2f} | {d_oa:.2f} | {delta:+.2f} |")

    deltas = np.array(deltas)
    ginises = np.array(ginises)
    r = np.corrcoef(ginises, deltas)[0, 1]
    lines += ["",
              f"- seen 训练 Gini 与 ΔOA 的 Pearson 相关：r={r:.3f}（5 个组，样本少，仅作定性参考）",
              "- 结论：若 r 为正且大，支持'长尾越严重→平衡采样收益越大'；否则机理不成立或需其他解释。"]

    with open("result_IP/长尾分析_group.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
