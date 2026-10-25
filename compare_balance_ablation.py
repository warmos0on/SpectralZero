#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""平衡采样对照分析：logs_base/ (A) vs logs_bal/ (B)

判定标准见 实验协议_平衡采样对照.md 第三节（跑前已登记）：
  主指标 = AA（逐类平均）；三 seed 方向一致才算结论；必须同报跨 seed std；逐类全表。
"""
import re, os, sys, glob, statistics, collections
import numpy as np

# Windows 默认 GBK 控制台 print 不了「⚠️」等字符，直接抛 UnicodeEncodeError；
# 统一改走 UTF-8，避免分析在默认终端下崩掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OA_RE = re.compile(r"unseen OA is ([-\d.]+), unseen AA is ([-\d.]+), kappa is (\S+)")
AA_RE = re.compile(r"per AA \[([^\]]*)\]")

GROUPS = {
    "G1": "Corn-notill,Hay-windrowed,Woods",
    "G2": "Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill",
    "G3": "Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill",
    "G4": "Corn,Alfalfa,Grass-pasture",
    "G5": "Wheat,Oats,Soybean-notill",
    "G6": "Soybean-clean",
}
# datasets.py label_values 的字典字面量键序（= 日志 per-AA 顺序）
ORDER = ["Grass-trees","Buildings-Grass-Trees-Drives","Corn-notill","Grass-pasture-mowed",
         "Corn","Woods","Soybean-mintill","Soybean-clean","Stone-Steel-Towers","Hay-windrowed",
         "Wheat","Alfalfa","Oats","Soybean-notill","Corn-mintill","Grass-pasture"]
ID2NAME = {1:"Alfalfa",2:"Corn-notill",3:"Corn-mintill",4:"Corn",5:"Grass-pasture",6:"Grass-trees",
           7:"Grass-pasture-mowed",8:"Hay-windrowed",9:"Oats",10:"Soybean-notill",
           11:"Soybean-mintill",12:"Soybean-clean",13:"Wheat",14:"Woods",
           15:"Buildings-Grass-Trees-Drives",16:"Stone-Steel-Towers"}

def fold_of_hash(h):
    import hashlib
    for g, s in GROUPS.items():
        if hashlib.md5(s.encode()).hexdigest()[:4] == h:
            return g
    return None

def parse_dir(d):
    """-> ({(fold, seed): (oa, aa, per[list])}, [未跑完的日志名])

    每折取所有评测点中的最优（论文协议）。只接受跑完的日志（含 loger 的 CRITICAL
    完成行）：中途崩溃的日志若参与「取最优」，截断轨迹能靠早期尖峰赢过训练完整的折。
    """
    out, partial = {}, []
    for f in glob.glob(os.path.join(d, "log_Indian_seed*_g*.txt")):
        m = re.match(r"log_Indian_seed(\d+)_g([0-9a-f]{4})\.txt", os.path.basename(f))
        if not m: continue
        seed, h = int(m.group(1)), m.group(2)
        g = fold_of_hash(h)
        if g is None: continue
        text = open(f, encoding="utf-8", errors="replace").read()
        if "Current experment have been finished" not in text:
            partial.append(os.path.basename(f)); continue
        pts, cur = [], None
        for line in text.splitlines():
            mo = OA_RE.search(line)
            if mo:
                if cur: pts.append(cur)
                cur = [float(mo.group(1)), float(mo.group(2)), []]
                a = AA_RE.search(line)
                if a: cur[2] = [float(x) for x in a.group(1).replace(",", " ").split()]
                continue
            if cur is not None and not cur[2]:
                a = AA_RE.search(line)
                if a: cur[2] = [float(x) for x in a.group(1).replace(",", " ").split()]
        if cur: pts.append(cur)
        if pts:
            out[(g, seed)] = tuple(max(pts, key=lambda p: p[0]))
    return out, partial

def pixel_counts():
    import scipy.io as io
    m = io.loadmat("data/Indian_gt.mat")
    g = np.array([v for v in m if not k_start(v)][0]) if False else np.array(
        [m[k] for k in m if not k.startswith("__")][0])
    c = collections.Counter(int(x) for x in g.ravel() if x > 0)
    return {ID2NAME[i]: n for i, n in c.items() if i in ID2NAME}

def k_start(k):  # 占位，保持上面写法可读
    return k.startswith("__")

def summarize(tag, data, seeds):
    folds = [g for g in GROUPS if any((g, s) in data for s in seeds)]
    per_fold, cls_vals = {}, collections.defaultdict(list)
    for g in folds:
        oa = [data[(g, s)][0] for s in seeds if (g, s) in data]
        aa = [data[(g, s)][1] for s in seeds if (g, s) in data]
        per_fold[g] = (statistics.mean(oa), statistics.pstdev(oa) if len(oa) > 1 else 0.0,
                       statistics.mean(aa), statistics.pstdev(aa) if len(aa) > 1 else 0.0)
        sub = [c for c in ORDER if c in GROUPS_CLASSES[g]]
        for s in seeds:
            if (g, s) not in data: continue
            per = data[(g, s)][2]
            if len(per) != len(sub): continue
            for c, v in zip(sub, per): cls_vals[c].append(v)
    return per_fold, cls_vals

GROUPS_CLASSES = {g: [c.strip() for c in s.split(",")] for g, s in GROUPS.items()}

def main():
    seeds = [42, 123, 456]
    A, A_part = parse_dir("logs_base")
    B, B_part = parse_dir("logs_bal")
    print(f"解析到：A(base) {len(A)} 条 · B(bal) {len(B)} 条   （期望各 18 条 = 6 折 × 3 seed）\n")
    if A_part or B_part:
        print(f"⚠️ 跳过未跑完的日志 {len(A_part) + len(B_part)} 个（缺完成标记）："
              f"{sorted(A_part + B_part)}\n")
    if not A or not B:
        print("⚠️ 数据不全，先跑完再分析"); return

    # 配对比较要求两侧覆盖同一组 (折, seed)；否则 Δ 是两套折集之差
    common = sorted(set(A) & set(B))
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if only_a or only_b:
        print(f"⚠️ A/B 覆盖不一致：仅在 A {only_a}，仅在 B {only_b}；"
              f"以下只用共同 {len(common)} 条配对\n")
    if not common:
        print("⚠️ 没有共同的 (折, seed)，无法做配对比较"); return
    A = {k: A[k] for k in common}
    B = {k: B[k] for k in common}

    px = pixel_counts()
    pfA, cvA = summarize("A", A, seeds)
    pfB, cvB = summarize("B", B, seeds)

    print("="*104)
    print("### 一、逐折（每折取最优评测点，跨 3 seed 均值±std）\n")
    hdr = f"{'折':6s}{'A:OA':>16s}{'B:OA':>16s}{'ΔOA':>8s}{'A:AA':>16s}{'B:AA':>16s}{'ΔAA':>8s}"
    print(hdr); print("-"*len(hdr))
    for g in GROUPS:
        if g not in pfA or g not in pfB: continue
        a, b = pfA[g], pfB[g]
        print(f"{g:6s}{a[0]:8.2f}±{a[1]:<6.2f}{b[0]:8.2f}±{b[1]:<6.2f}{b[0]-a[0]:+8.2f}"
              f"{a[2]:8.2f}±{a[3]:<6.2f}{b[2]:8.2f}±{b[3]:<6.2f}{b[2]-a[2]:+8.2f}")

    print("\n" + "="*104)
    print("### 二、汇总（三种口径，全部带标签）\n")
    def agg(pf, cv):
        foa = [pf[g][0] for g in pf]; faa = [pf[g][2] for g in pf]
        vals = {c: statistics.mean(v) for c, v in cv.items()}
        aa_cm = statistics.mean(vals.values())
        num = sum(vals[c]*px[c] for c in vals); den = sum(px[c] for c in vals)
        sds = [statistics.pstdev(v) if len(v) > 1 else 0.0 for v in cv.values()]
        return statistics.mean(foa), statistics.mean(faa), aa_cm, num/den, statistics.mean(sds)
    ra, rb = agg(pfA, cvA), agg(pfB, cvB)
    labels = ["OA (fold-mean)", "AA (fold-mean, 陷阱口径)", "AA (class-mean, 论文口径) ★",
              "OA (pixel-weighted)", "逐类跨seed σ 均值"]
    print(f"{'口径':34s}{'A 基线':>12s}{'B 平衡采样':>12s}{'Δ':>10s}")
    print("-"*70)
    for i, lb in enumerate(labels):
        print(f"{lb:34s}{ra[i]:12.2f}{rb[i]:12.2f}{rb[i]-ra[i]:+10.2f}")

    print("\n" + "="*104)
    print("### 三、逐类（该类当 unseen 那折的最优点，跨 3 seed）\n")
    hdr = f"{'类':34s}{'像素':>7s}{'A':>9s}{'B':>9s}{'Δ':>9s}"
    print(hdr); print("-"*len(hdr))
    for c in ORDER:
        if c not in cvA or c not in cvB: continue
        a, b = statistics.mean(cvA[c]), statistics.mean(cvB[c])
        print(f"{c:34s}{px.get(c,0):7d}{a:9.2f}{b:9.2f}{b-a:+9.2f}")

    # 预先登记的判定
    print("\n" + "="*104)
    print("### 四、按预登记标准的判定\n")
    per_seed = {}
    for s in seeds:
        # 与第二节 agg() 用同一组折（含单类折 G6）。G6 两臂都恒 ~100，排除它会让折平均 Δ
        # 被乘上 6/5（实测比值恒为 1.200），且同一份输出里出现两个不同的 ΔOA。
        a = [A[(g, s)][0] for g in GROUPS if (g, s) in A]
        b = [B[(g, s)][0] for g in GROUPS if (g, s) in B]
        if a and b: per_seed[s] = (statistics.mean(a), statistics.mean(b))
    print("  逐 seed（折平均 OA，含单类折 G6；排除 G6 会把 Δ 机械放大 6/5）：")
    up = 0
    for s, (a, b) in per_seed.items():
        d = b - a
        up += (d > 0)
        print(f"    seed{s}: A {a:.2f} → B {b:.2f}   ({d:+.2f})")
    print(f"  方向一致（三 seed 全涨）: {'✅ 是' if up == len(per_seed) == 3 else f'❌ 否（{up}/{len(per_seed)} 涨）'}")
    print(f"  主指标 ΔAA(class-mean) = {rb[2]-ra[2]:+.2f}   （std 变化 {rb[4]-ra[4]:+.2f}）")

if __name__ == "__main__":
    main()
