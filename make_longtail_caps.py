"""生成长尾消融实验的每类训练配额 cap 文件（研究用，不改动训练逻辑）。

用法:
    python make_longtail_caps.py

原理:
    main.py 在 get_train_test_num() 之后读取 --train_cap 指定的 JSON，
    对每个 seen 类的训练配额施加上限（cap 只作用于 seen 类，
    unseen 类配额不受影响）。本脚本按与主流程完全相同的逻辑复算
    各类的自然训练数，并生成"自然 / 轻微压制 / 强压制"三种长尾
    水平的 cap 文件。

注意:
    - 类别名 -> 原始标签的映射必须与 datasets.py 的 label_values 一致
      （Indian 的 label 不是按类别表顺序连续的 1..16）。
    - 分组定义与 run_honest_eval.py 的 GROUPS["Indian"] 保持一致。
"""
import json
import math
import os

import numpy as np

import utils
from run_honest_eval import GROUPS


# Indian label_values（与 datasets.py 完全一致）：类别名 -> 原始标签
INDIAN_LABEL_VALUES = {
    "Grass-trees": 6,
    "Buildings-Grass-Trees-Drives": 15,
    "Corn-notill": 2,
    "Grass-pasture-mowed": 7,
    "Corn": 4,
    "Woods": 14,
    "Soybean-mintill": 11,
    "Soybean-clean": 12,
    "Stone-Steel-Towers": 16,
    "Hay-windrowed": 8,
    "Wheat": 13,
    "Alfalfa": 1,
    "Oats": 9,
    "Soybean-notill": 10,
    "Corn-mintill": 3,
    "Grass-pasture": 5,
}

# 三个长尾水平：cap 上限（natural = 不设上限）
LEVEL_DEFS = {
    "mild": 150,      # 头部类压到 150 -> 头尾比约 10x
    "mid": 90,        # 头部类压到 90  -> 头尾比约 6x
    "uniform": 45,    # 头部类压到 45  -> 头尾比约 3x（贴近尾部下限）
}


def natural_train_counts(gt):
    """复现 utils.get_train_test_num(gt, 0.2) 的每类训练数（Indian 全 16 类）。"""
    class_num = int(np.max(gt))
    counts = {}
    for label in range(1, class_num + 1):
        per_class_num = int(np.sum(gt == label))
        theo_num = int(per_class_num * 0.2)
        real_num = theo_num if per_class_num > theo_num + 50 else 15
        counts[label] = real_num
    return counts


def gini(arr):
    arr = np.sort(np.asarray(arr, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return float("nan")
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def head_tail_ratio(counts):
    arr = np.asarray([c for c in counts.values() if c > 0], dtype=float)
    if len(arr) < 2:
        return float("nan")
    return float(arr.max() / arr.min())


def effective_counts(natural_by_name, cap):
    return {name: min(n, cap[name]) for name, n in natural_by_name.items()}


def main():
    if not os.path.exists("./data/Indian.mat") or not os.path.exists("./data/Indian_gt.mat"):
        raise FileNotFoundError("缺少 data/Indian.mat 或 data/Indian_gt.mat")
    gt = utils.read_mat("./data/Indian_gt.mat")
    nat_label = natural_train_counts(gt)  # 原始标签 -> 自然训练配额

    # 类别名 -> 自然训练配额（用准确的 label_values 映射）
    natural_by_name = {name: int(nat_label[lab]) for name, lab in INDIAN_LABEL_VALUES.items()}

    out_dir = os.path.join("result_IP", "longtail_caps")
    os.makedirs(out_dir, exist_ok=True)

    # 生成 cap 文件（key = 类别名；main.py 按 seen 类名匹配）
    for level, cap_val in LEVEL_DEFS.items():
        cap = {name: min(n, cap_val) for name, n in natural_by_name.items()}
        path = os.path.join(out_dir, f"level_{level}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cap, f, ensure_ascii=False, indent=2)
        print(f"已生成 {path}")

    # 校验 + 输出三种水平在"全 16 类"口径下的长尾指标
    levels = {"natural": None}
    for level in LEVEL_DEFS:
        levels[level] = os.path.join(out_dir, f"level_{level}.json")

    caps_by_level = {}
    for level, path in levels.items():
        caps_by_level[level] = (
            {name: float("inf") for name in natural_by_name}
            if path is None
            else json.load(open(path, encoding="utf-8"))
        )

    print("\n=== 每类训练配额（train_num=0.2 复算） ===")
    print(f"{'类别':34s} {'natural':>8s} {'mild(150)':>10s} {'uniform(45)':>12s}")
    for name in natural_by_name:
        nat = natural_by_name[name]
        print(f"{name:34s} {nat:8d} {min(nat, LEVEL_DEFS['mild']):10d} "
              f"{min(nat, LEVEL_DEFS['uniform']):12d}")

    print("\n=== 全 16 类口径长尾指标 ===")
    for level in levels:
        eff = effective_counts(natural_by_name, caps_by_level[level])
        print(f"  {level:8s}: head/tail={head_tail_ratio(eff):6.2f}x  "
              f"Gini={gini(list(eff.values())):.4f}  total={sum(eff.values())}  "
              f"({sum(eff.values()) / sum(natural_by_name.values()) * 100:.1f}% of natural)")

    print("\n=== 各 unseen 组对应的 seen 集合长尾程度（实际生效口径） ===")
    for gi, unseen in enumerate(GROUPS["Indian"], start=1):
        if len(unseen) != 3:
            continue  # 只列 G1-G5
        seen = [n for n in natural_by_name if n not in unseen]
        print(f"  G{gi} unseen={unseen}")
        for level in levels:
            caps = caps_by_level[level]
            eff = {name: min(natural_by_name[name], caps[name]) for name in seen}
            nat = {name: natural_by_name[name] for name in seen}
            print(f"    {level:8s}: head/tail={head_tail_ratio(eff):6.2f}x "
                  f"(natural {head_tail_ratio(nat):6.2f}x)  "
                  f"Gini={gini(list(eff.values())):.4f} (natural {gini(list(nat.values())):.4f})  "
                  f"total={sum(eff.values())} ({sum(eff.values()) / sum(nat.values()) * 100:.1f}% of natural)")

    # 一致性校验：cap 不允许超过自然配额；cap 键必须覆盖全部 16 类
    for level, caps in caps_by_level.items():
        assert set(caps) == set(natural_by_name), f"{level} cap 键集合不完整"
        for name, n in natural_by_name.items():
            assert caps[name] >= 1, f"{level} {name} cap 过小"
    print("\n校验通过：cap 键完整，且不会放大任何类别的训练配额。")


if __name__ == "__main__":
    main()
