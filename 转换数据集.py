# -*- coding: utf-8 -*-
"""把原始高光谱数据集转换成 SpectralZero 代码所需的 data/*.mat 布局。

用法：
    python 转换数据集.py --src <原始数据目录>             # 自动识别目录里的数据集并转换
    python 转换数据集.py --src <目录> --dataset Indian    # 只转指定的一套
    python 转换数据集.py --check                          # 校验 data/ 里已有的文件

目标布局（datasets.get_dataset + utils.read_mat 的约定）：
    data/Indian.mat     只含一个数组，shape = (H, W, C)   —— 高光谱立方体
    data/Indian_gt.mat  只含一个数组，shape = (H, W)      —— 标签图，0=无标注，1..N=类
    （Houston / LongKou 同理）

注意：utils.read_mat 取「文件里第一个 ndarray」，所以每个 .mat 只写一个变量。
原始数据常见变体（Indian_pines_corrected / Houston2013 / WHU_Hi_LongKou 等）会自动识别；
标签 id 与 datasets.py 里的类表对不上时会告警，可用 --gt-remap 传映射 JSON 修正。
"""
import argparse
import json
import sys

import numpy as np
import scipy.io as sio

try:
    import mat73
except ImportError:
    mat73 = None

# 各数据集的期望属性（来自 datasets.py 的 label_values 与论文设定）
EXPECTED = {
    "Indian": {
        "bands": 200,
        "img_hints": ["indian_pines_corrected", "indian_pines"],
        "gt_hints": ["indian_pines_gt", "indian_gt"],
        "classes": {  # 类名 -> gt 标签（与 datasets.py 一致）
            "Alfalfa": 1, "Corn-notill": 2, "Corn-mintill": 3, "Corn": 4,
            "Grass-pasture": 5, "Grass-trees": 6, "Grass-pasture-mowed": 7,
            "Hay-windrowed": 8, "Oats": 9, "Soybean-notill": 10,
            "Soybean-mintill": 11, "Soybean-clean": 12, "Wheat": 13,
            "Woods": 14, "Buildings-Grass-Trees-Drives": 15, "Stone-Steel-Towers": 16,
        },
    },
    "Houston": {
        "bands": 144,
        "img_hints": ["houston2013", "houston"],
        "gt_hints": ["houston2013_gt", "houston_gt"],
        "classes": {
            "Healthy grass": 1, "Stressed grass": 2, "Synthetic grass": 3, "Trees": 4,
            "Soil": 5, "Water": 6, "Residential": 7, "Commercial": 8, "Road": 9,
            "Highways": 10, "Railways": 11, "Parking Lot 1": 12, "Parking Lot 2": 13,
            "Tennis Court": 14, "Running Track": 15,
        },
    },
    "LongKou": {
        "bands": 270,
        "img_hints": ["longkou", "whu_hi_longkou", "whuhilongkou"],
        "gt_hints": ["longkou_gt", "whu_hi_longkou_gt"],
        "classes": {
            "Corn": 1, "Cotton": 2, "Sesame": 3, "Broad-leaf soybean": 4,
            "Narrow-leaf soybean": 5, "Rice": 6, "Water": 7,
            "Roads and houses": 8, "Mixed weed": 9,
        },
    },
}


def load_arrays(path):
    """读 .mat，返回 {变量名: ndarray}，兼容 v5/v7.3。"""
    try:
        mat = sio.loadmat(path)
        arrays = {k: v for k, v in mat.items()
                  if isinstance(v, np.ndarray) and not k.startswith("__")}
    except NotImplementedError:
        if mat73 is None:
            raise SystemExit(f"{path} 是 v7.3 格式，请先 pip install mat73")
        mat = mat73.loadmat(path)
        arrays = {k: v for k, v in mat.items() if isinstance(v, np.ndarray)}
    if not arrays:
        raise SystemExit(f"{path} 里没找到数组")
    return arrays


def pick_array(arrays, kind, name):
    """从多个变量里挑出图像（3D 最大者）或标签图（2D 整数）。"""
    if kind == "img":
        cands = [(k, v) for k, v in arrays.items() if v.ndim == 3]
        if not cands:
            raise SystemExit(f"{name}: 没有 3D 数组可当高光谱立方体")
        return max(cands, key=lambda kv: kv[1].size)
    else:
        cands = [(k, v) for k, v in arrays.items() if v.ndim == 2]
        if not cands:
            raise SystemExit(f"{name}: 没有 2D 数组可当标签图")
        return max(cands, key=lambda kv: kv[1].size)


def convert(dataset, img_path, gt_path, out_dir, remap=None):
    spec = EXPECTED[dataset]
    img_name, img = pick_array(load_arrays(img_path), "img", img_path)
    gt_name, gt = pick_array(load_arrays(gt_path), "gt", gt_path)

    print(f"\n[{dataset}] 图像变量 {img_name!r} {img.shape} {img.dtype} | "
          f"标签变量 {gt_name!r} {gt.shape} {gt.dtype}")

    # 形状自检
    if img.shape[:2] != gt.shape:
        raise SystemExit(f"{dataset}: 图像 {img.shape[:2]} 与标签 {gt.shape} 尺寸不一致！")
    if img.shape[2] != spec["bands"]:
        print(f"  ⚠ 波段数 {img.shape[2]} ≠ 期望 {spec['bands']}（模型能跑，但与论文设定不同，记录在案）")

    # 标签处理：只保留 1..N，0=无标注
    gt = np.asarray(gt)
    if remap:
        new_gt = np.zeros_like(gt, dtype=np.int32)
        for raw, new in remap.items():
            new_gt[gt == int(raw)] = int(new)
        gt = new_gt
    gt = gt.astype(np.int32)

    # 标签自检：与 datasets.py 的类表对照
    labels, counts = np.unique(gt[gt > 0], return_counts=True)
    expect_ids = set(spec["classes"].values())
    got_ids = set(labels.tolist())
    print(f"  类别数 {len(labels)}，像素总计 {int(counts.sum())}")
    for lab, cnt in zip(labels, counts):
        names = [n for n, i in spec["classes"].items() if i == int(lab)]
        tag = names[0] if names else "???（不在 datasets.py 类表里）"
        print(f"    {int(lab):2d}  {tag:35s} {int(cnt):7d} 像素")
    if got_ids != expect_ids:
        print(f"  ⚠ 标签 id 集合与 datasets.py 不一致：多 {sorted(got_ids - expect_ids)} 缺 {sorted(expect_ids - got_ids)}")
        print(f"    如需重映射，用 --gt-remap 传 JSON，例如 '{{\"1\": 3, \"3\": 1}}'")

    sio.savemat(f"{out_dir}/{dataset}.mat", {"hsi": img})
    sio.savemat(f"{out_dir}/{dataset}_gt.mat", {"gt": gt})
    print(f"  ✓ 已写出 {out_dir}/{dataset}.mat 与 {out_dir}/{dataset}_gt.mat")


def check(out_dir):
    """校验 data/ 里的文件是否符合代码约定。"""
    ok = True
    for dataset, spec in EXPECTED.items():
        paths = (f"{out_dir}/{dataset}.mat", f"{out_dir}/{dataset}_gt.mat")
        try:
            img = pick_array(load_arrays(paths[0]), "img", paths[0])
            gt = pick_array(load_arrays(paths[1]), "gt", paths[1])
        except (SystemExit, FileNotFoundError) as e:
            print(f"[{dataset}] ✗ 读不了：{e}")
            ok = False
            continue
        notes = []
        if img.shape[:2] != gt.shape:
            notes.append("图像/标签尺寸不一致")
        if img.shape[2] != spec["bands"]:
            notes.append(f"波段 {img.shape[2]}≠{spec['bands']}")
        got = set(np.unique(gt[gt > 0]).tolist())
        if got != set(spec["classes"].values()):
            notes.append(f"标签集合不符 多{sorted(got - set(spec['classes'].values()))} 缺{sorted(set(spec['classes'].values()) - got)}")
        status = "✓" if not notes else "⚠ " + "；".join(notes)
        print(f"[{dataset}] {img.shape} / {gt.shape}  {status}")
        ok = ok and not notes
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="转换原始高光谱数据为 SpectralZero 的 data/*.mat 布局")
    ap.add_argument("--src", help="原始数据所在目录")
    ap.add_argument("--dataset", choices=["Indian", "Houston", "LongKou", "all"], default="all")
    ap.add_argument("--out", default="data", help="输出目录（默认 data）")
    ap.add_argument("--gt-remap", help='标签重映射 JSON，如 \'{"1": 3, "3": 1}\'')
    ap.add_argument("--check", action="store_true", help="只校验 data/ 里已有文件")
    args = ap.parse_args()

    if args.check:
        sys.exit(check(args.out))
    if not args.src:
        ap.error("需要 --src 原始数据目录（或用 --check 校验）")

    remap = json.loads(args.gt_remap) if args.gt_remap else None
    import os
    files = os.listdir(args.src)
    lowered = {f.lower(): f for f in files}

    def find(hints, exclude_gt=False):
        for h in hints:
            for low, orig in lowered.items():
                if h in low and low.endswith(".mat"):
                    if exclude_gt and "_gt" in low:
                        continue
                    return os.path.join(args.src, orig)
        return None

    todo = ["Indian", "Houston", "LongKou"] if args.dataset == "all" else [args.dataset]
    for ds in todo:
        spec = EXPECTED[ds]
        img = find(spec["img_hints"], exclude_gt=True)
        gt = find(spec["gt_hints"])
        if not img or not gt:
            print(f"[{ds}] 跳过：目录里没找到匹配的图像/标签 .mat（找过 {spec['img_hints']} / {spec['gt_hints']}）")
            continue
        convert(ds, img, gt, args.out, remap)


if __name__ == "__main__":
    main()
