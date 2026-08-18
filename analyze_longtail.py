# -*- coding: utf-8 -*-
"""统计各 seen 类训练样本数，验证长尾分布假设。"""
import numpy as np
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import datasets, utils

for ds_name in ["Indian", "LongKou"]:
    cfg = json.load(open(f"./config/{ds_name}.json", encoding="utf-8-sig"))
    HSI, gt, label_value = datasets.get_dataset(dataset_name=ds_name)
    unseen = cfg["unseen_classes"]
    label_value, seen_names, unseen_names = utils.reloc_class(label_value, unseen)
    seen_label, _ = utils.get_seen_unseen_class(label_value)
    seen_gt, _ = utils.fix_label(seen_names, gt)
    train_num, test_num = utils.get_train_test_num(gt=seen_gt, train_num=cfg["train_num"])
    for seed in [42, 123, 456]:
        utils.fixed_seed(seed)
        train_gt, test_gt, _ = utils.split_gt(seen_gt, train_num, test_num, seen_label)
        counts = np.bincount(train_gt[train_gt>0].ravel().astype(int) - 1, minlength=len(seen_names))
        total = counts.sum()
        print(f"\n=== {ds_name} seed={seed}  total_train={total} ===")
        name_keys = list(seen_names.keys())
        order = np.argsort(counts)[::-1]
        for idx in order:
            name = name_keys[idx]
            c = int(counts[idx])
            pct = 100.0 * c / total
            bar = '#' * int(pct/2)
            print(f"  {name:40s} {c:6d}  ({pct:5.1f}%)  {bar}")
        if counts.max() > 0:
            ratio = counts.max() / counts.min()
            print(f"  >>> max/min ratio = {ratio:.1f}x")
