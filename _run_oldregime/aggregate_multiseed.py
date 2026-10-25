#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多 seed × 论文轮换折汇总（口径带标签，禁止手搓）。"""
import re, glob, os, collections, statistics, json
import numpy as np
import scipy.io as io

OA_RE = re.compile(r"unseen OA is ([-\d.]+), unseen AA is ([-\d.]+), kappa is (\S+)")
AA_RE = re.compile(r"per AA \[([^\]]*)\]")

GT = {"Indian": ["data/Indian_gt.mat", "Indian_gt"],
      "LongKou": ["data/LongKou_gt.mat", "WHHL_Hi_LongKou_gt"]}

# datasets.py 的 label_values 字典序（决定 unseen 重映射顺序 = 日志 per-AA 顺序）
# datasets.py label_values 的**字典字面量键序**（= 日志 per-AA 顺序），不是字母序
ORDER = {
    "Indian": ["Grass-trees","Buildings-Grass-Trees-Drives","Corn-notill","Grass-pasture-mowed",
               "Corn","Woods","Soybean-mintill","Soybean-clean","Stone-Steel-Towers","Hay-windrowed",
               "Wheat","Alfalfa","Oats","Soybean-notill","Corn-mintill","Grass-pasture"],
    "LongKou": ["Corn","Cotton","Sesame","Broad-leaf soybean","Narrow-leaf soybean","Rice",
                "Water","Mixed weed","Roads and houses"],
}
# 折签名：seed 42 各自最优点的 per-AA（来自 复现结果_论文折.md，用于把 hash 映射回 G 编号）
SIG = {
    ("Indian","G1"): [99.72, 98.74, 62.55], ("Indian","G2"): [88.77, 49.17, 98.23],
    ("Indian","G3"): [100.00, 30.11, 94.76], ("Indian","G4"): [98.31, 0.00, 95.86],
    ("Indian","G5"): [92.68, 100.00, 94.34], ("Indian","G6"): [100.00],
    ("LongKou","G1"): [98.51, 59.23, 100.00], ("LongKou","G2"): [0.90, 96.65, 80.53],
    ("LongKou","G3"): [96.56, 14.95, 6.70],
}
# 论文轮换折分组（Table III）
FOLD_CLASSES = {
    "Indian": {
        "G1": ["Corn-notill","Hay-windrowed","Woods"],
        "G2": ["Grass-trees","Buildings-Grass-Trees-Drives","Soybean-mintill"],
        "G3": ["Grass-pasture-mowed","Stone-Steel-Towers","Corn-mintill"],
        "G4": ["Corn","Alfalfa","Grass-pasture"],
        "G5": ["Wheat","Oats","Soybean-notill"],
        "G6": ["Soybean-clean"],
    },
    "LongKou": {
        "G1": ["Corn","Cotton","Water"],
        "G2": ["Sesame","Broad-leaf soybean","Rice"],
        "G3": ["Narrow-leaf soybean","Mixed weed","Roads and houses"],
    },
}

def pixel_counts(ds):
    path, key = GT[ds]
    m = io.loadmat(path)
    k = [x for x in m if not x.startswith("__")][0]
    g = np.array(m[k])
    cnt = collections.Counter(int(x) for x in g.ravel() if x > 0)
    # gt id -> 类名（datasets.py 映射）
    if ds == "Indian":
        id2name = {1:"Alfalfa",2:"Corn-notill",3:"Corn-mintill",4:"Corn",5:"Grass-pasture",
                   6:"Grass-trees",7:"Grass-pasture-mowed",8:"Hay-windrowed",9:"Oats",
                   10:"Soybean-notill",11:"Soybean-mintill",12:"Soybean-clean",13:"Wheat",
                   14:"Woods",15:"Buildings-Grass-Trees-Drives",16:"Stone-Steel-Towers"}
    else:
        id2name = {1:"Corn",2:"Cotton",3:"Sesame",4:"Broad-leaf soybean",5:"Narrow-leaf soybean",
                   6:"Rice",7:"Water",8:"Roads and houses",9:"Mixed weed"}
    return {id2name[i]: c for i, c in cnt.items() if i in id2name}

def parse_logs():
    """-> {(ds, foldkey, seed): {'best':(oa,aa,kappa,perlist), 'all':[...]}}"""
    out = collections.defaultdict(dict)
    for f in glob.glob("log_*_seed*_g*.txt"):
        m = re.match(r"log_(Indian|LongKou)_seed(\d+)_g([0-9a-f]+)\.txt", os.path.basename(f))
        if not m: continue
        ds, seed, h = m.group(1), int(m.group(2)), m.group(3)
        pts = []
        cur = None
        for line in open(f, encoding="utf-8", errors="replace"):
            mo = OA_RE.search(line)
            if mo:
                if cur: pts.append(cur)
                cur = [float(mo.group(1)), float(mo.group(2)), mo.group(3), []]
                aa = AA_RE.search(line)
                if aa: cur[3] = [float(x) for x in aa.group(1).replace(",", " ").split()]
                continue
            if cur is not None and not cur[3]:
                aa = AA_RE.search(line)
                if aa:
                    cur[3] = [float(x) for x in aa.group(1).replace(",", " ").split()]
        if cur: pts.append(cur)
        pts = [tuple(p) for p in pts]
        if pts:
            best = max(pts, key=lambda p: p[0])
            out[(ds, h, seed)] = {"best": best, "n": len(pts)}
    return out

def main():
    logs = parse_logs()
    px = {ds: pixel_counts(ds) for ds in ("Indian", "LongKou")}

    # 把 hash 映射回 G 编号（用 seed42 最优点的 per-AA 对签名做最近匹配）
    print("### 折 hash 映射")
    foldmap = {}
    for (ds, h, seed) in sorted(logs):
        if (ds, h) in foldmap or seed != 42: continue
        v = logs[(ds, h, seed)]["best"][3]
        best, bd = None, 1e9
        for (d2, G), sig in SIG.items():
            if d2 != ds or len(sig) != len(v): continue
            d = sum((a - b) ** 2 for a, b in zip(sig, v))
            if d < bd: best, bd = G, d
        foldmap[(ds, h)] = best or "?"
    # 用其它 seed 补齐（若 seed42 缺失）
    for (ds, h, seed) in sorted(logs):
        if (ds, h) in foldmap: continue
        v = logs[(ds, h, seed)]["best"][3]
        best, bd = None, 1e9
        for (d2, G), sig in SIG.items():
            if d2 != ds or len(sig) != len(v): continue
            d = sum((a - b) ** 2 for a, b in zip(sig, v))
            if d < bd: best, bd = G, d
        foldmap[(ds, h)] = best or "?"
    for k in sorted(foldmap): print("  ", k, "->", foldmap[k], " (fit d=%.1f)" % 0)

    seeds = sorted({s for (_, _, s) in logs})
    print("\nseeds:", seeds)
    print("log files:", len(glob.glob('log_*_seed*_g*.txt')))

    # ---- 每折每 seed 最优 ----
    print("\n### 每折最优评测点（OA / AA）")
    per_fold = collections.defaultdict(dict)   # (ds,G) -> seed -> (oa,aa,per)
    for (ds, h, seed), v in logs.items():
        G = foldmap[(ds, h)]
        per_fold[(ds, G)][seed] = v["best"]

    for ds in ("Indian", "LongKou"):
        for G in sorted(FOLD_CLASSES[ds]):
            row = per_fold.get((ds, G), {})
            cells = []
            for s in seeds:
                if s in row: cells.append(f"seed{s}: {row[s][0]:6.2f}/{row[s][1]:6.2f}")
                else:        cells.append(f"seed{s}:   --  /  -- ")
            # mean±std
            oas = [row[s][0] for s in seeds if s in row]
            aas = [row[s][1] for s in seeds if s in row]
            ms = f"  mean±std OA {statistics.mean(oas):6.2f}±{statistics.pstdev(oas):5.2f}  AA {statistics.mean(aas):6.2f}±{statistics.pstdev(aas):5.2f}" if len(oas) > 1 else ""
            print(f"  {ds:8s} {G:2s}  " + " | ".join(cells) + ms)

    # ---- 逐类：该类当 unseen 那折的最优点评值，跨 seed 平均 ----
    print("\n### 逐类（best-point per-class acc，跨 seed 平均）")
    class_acc = {}   # (ds, class) -> (mean, std, n)
    for ds in ("Indian", "LongKou"):
        cls_vals = collections.defaultdict(list)
        for G, classes in FOLD_CLASSES[ds].items():
            for s in seeds:
                rec = per_fold.get((ds, G), {}).get(s)
                if not rec: continue
                per = rec[3]
                # 日志 per-AA 顺序 = datasets.py label_values 字典序过滤后的顺序
                sub = [c for c in ORDER[ds] if c in classes]
                if len(sub) != len(per): continue
                for c, v in zip(sub, per): cls_vals[c].append(v)
        print(f"\n  -- {ds} --")
        for c in ORDER[ds]:
            if c in cls_vals:
                vs = cls_vals[c]
                m = statistics.mean(vs); sd = statistics.pstdev(vs) if len(vs) > 1 else 0.0
                class_acc[(ds, c)] = (m, sd, len(vs))
                print(f"    {c:34s} {m:6.2f} ± {sd:5.2f}  (n={len(vs)})  像素 {px[ds].get(c,'?')}")

    # ---- 三口径汇总 ----
    print("\n### 汇总（三口径，全部带标签）")
    for ds in ("Indian", "LongKou"):
        # (a) AA = class-mean（论文口径）
        vals = [class_acc[(ds, c)][0] for c in ORDER[ds] if (ds, c) in class_acc]
        aa_classmean = statistics.mean(vals) if vals else float("nan")
        # (b) 加权 OA = 由逐类值按像素加权
        num = sum(class_acc[(ds, c)][0] * px[ds][c] for c in ORDER[ds] if (ds, c) in class_acc)
        den = sum(px[ds][c] for c in ORDER[ds] if (ds, c) in class_acc)
        woa = num / den
        # (c) 折平均 OA / 折平均 AA
        foa, faa = [], []
        for G in FOLD_CLASSES[ds]:
            o = [per_fold[(ds, G)][s][0] for s in seeds if s in per_fold.get((ds, G), {})]
            a = [per_fold[(ds, G)][s][1] for s in seeds if s in per_fold.get((ds, G), {})]
            if len(o) == len(seeds): foa.append(statistics.mean(o)); faa.append(statistics.mean(a))
        print(f"\n  {ds}  ({len(seeds)} seeds)")
        print(f"    AA  (class-mean, 论文口径)      = {aa_classmean:.2f}")
        print(f"    OA  (pixel-weighted, 逐类派生)  = {woa:.2f}")
        print(f"    OA  (fold-mean, 陷阱A口径)      = {statistics.mean(foa):.2f}  ± {statistics.pstdev(foa):.2f}  (跨折离散)")
        print(f"    AA  (fold-mean, 陷阱A口径)      = {statistics.mean(faa):.2f}  ± {statistics.pstdev(faa):.2f}")
        # 逐类 std 再看跨 seed 波动
        sds = [class_acc[(ds, c)][1] for c in ORDER[ds] if (ds, c) in class_acc]
        print(f"    逐类 std 均值（跨 seed 波动）    = {statistics.mean(sds):.2f}   最大 {max(sds):.2f}")

if __name__ == "__main__":
    main()
