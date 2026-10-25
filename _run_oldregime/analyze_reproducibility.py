#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可复现性归因：把「复现精度 / 跨 seed 波动」与「类别固有属性」对照。

类别属性（全部从本地原始数据算，不依赖远程日志）：
  n          该类像素数
  purity     邻域纯度 = patch 内同类像素 / patch 内非背景像素（模型实际看到的比例）
  purity_raw 邻域纯度（分母用 patch 面积，含背景）
  maxcos     与「其他类平均谱」的最大余弦（越大越难分；也叫可混性）
  margin     1 − maxcos（光谱可分性）

复现结果（三 seed 42/123/456 的最优评测点，来自远程 loger；远程已关机，此处固化）
"""
import collections
import numpy as np
import scipy.io as io
from scipy.ndimage import uniform_filter

DATASETS = {
    "Indian": dict(mat="data/Indian.mat", gt="data/Indian_gt.mat", patch=15,
                   ids={1:"Alfalfa",2:"Corn-notill",3:"Corn-mintill",4:"Corn",5:"Grass-pasture",
                        6:"Grass-trees",7:"Grass-pasture-mowed",8:"Hay-windrowed",9:"Oats",
                        10:"Soybean-notill",11:"Soybean-mintill",12:"Soybean-clean",13:"Wheat",
                        14:"Woods",15:"Buildings-Grass-Trees-Drives",16:"Stone-Steel-Towers"}),
    "LongKou": dict(mat="data/LongKou.mat", gt="data/LongKou_gt.mat", patch=13,
                    ids={1:"Corn",2:"Cotton",3:"Sesame",4:"Broad-leaf soybean",5:"Narrow-leaf soybean",
                         6:"Rice",7:"Water",8:"Roads and houses",9:"Mixed weed"}),
}

# 三 seed 复现（mean, std）—— 最优评测点逐类精度
REPRO = {
 "Indian": {
  "Alfalfa":(0.00,0.00),"Corn-notill":(99.32,0.34),"Corn-mintill":(93.52,4.16),"Corn":(99.01,0.72),
  "Grass-pasture":(96.07,0.17),"Grass-trees":(92.15,2.64),"Grass-pasture-mowed":(100.00,0.00),
  "Hay-windrowed":(60.74,21.05),"Oats":(66.67,33.99),"Soybean-notill":(96.81,2.31),
  "Soybean-mintill":(98.67,0.43),"Soybean-clean":(100.00,0.00),"Wheat":(71.87,31.52),
  "Woods":(97.68,2.02),"Buildings-Grass-Trees-Drives":(32.50,13.53),"Stone-Steel-Towers":(53.77,24.25)},
 "LongKou": {
  "Corn":(92.97,5.05),"Cotton":(57.65,10.60),"Sesame":(0.81,0.57),"Broad-leaf soybean":(94.62,4.24),
  "Narrow-leaf soybean":(84.03,15.77),"Rice":(80.36,4.95),"Water":(100.00,0.00),
  "Roads and houses":(4.34,2.91),"Mixed weed":(37.66,18.42)},
}
PAPER = {
 "Indian": {
  "Corn-notill":99.84,"Hay-windrowed":94.39,"Woods":67.99,"Grass-trees":99.67,
  "Buildings-Grass-Trees-Drives":61.60,"Soybean-mintill":97.45,"Grass-pasture-mowed":100.00,
  "Stone-Steel-Towers":0.00,"Corn-mintill":97.13,"Corn":100.00,"Alfalfa":0.00,
  "Grass-pasture":93.58,"Wheat":99.66,"Oats":100.00,"Soybean-notill":92.08,"Soybean-clean":99.85},
 "LongKou": {
  "Corn":99.26,"Cotton":85.40,"Water":100.00,"Sesame":93.70,"Broad-leaf soybean":1.47,
  "Rice":89.50,"Narrow-leaf soybean":95.50,"Mixed weed":18.76,"Roads and houses":26.00},
}

def load(cfg):
    m = io.loadmat(cfg["mat"]); g = io.loadmat(cfg["gt"])
    X = [v for v in m.values() if isinstance(v, np.ndarray) and v.ndim == 3][0].astype(np.float32)
    G = [v for v in g.values() if isinstance(v, np.ndarray) and v.ndim == 2][0]
    return X, np.asarray(G).astype(np.int32)

def features(X, G, ids, patch):
    """每类的 n / 纯度 / 平均谱 / 可混性 / 纯光谱可分性。"""
    H, W, B = X.shape
    valid = G > 0
    k = patch
    nvalid = uniform_filter(valid.astype(np.float32), size=k, mode="constant") * (k * k)
    rows = {}
    means = {}
    for cid, name in ids.items():
        mask = (G == cid)
        n = int(mask.sum())
        if n == 0: continue
        same = uniform_filter(mask.astype(np.float32), size=k, mode="constant") * (k * k)
        purity_valid = float(same[mask].sum() / max(nvalid[mask].sum(), 1e-9))
        purity_raw = float(same[mask].mean() / (k * k))
        mu = X[mask].mean(axis=0)
        rows[name] = dict(n=n, purity=purity_valid, purity_raw=purity_raw)
        means[name] = mu
    # 光谱可混性：与其它类平均谱的最大余弦
    names = list(means)
    M = np.stack([means[n] for n in names])
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = Mn @ Mn.T
    np.fill_diagonal(C, -1)
    for i, n in enumerate(names):
        rows[n]["maxcos"] = float(C[i].max())
        rows[n]["margin"] = float(1.0 - C[i].max())
        rows[n]["nearest"] = names[int(C[i].argmax())]
    # 最近质心分类器（NCM）逐类精度 —— 「只用光谱」能分对多少
    cen = Mn                       # 单位化类中心
    flat = X.reshape(-1, B)
    lab = G.ravel()
    keep = lab > 0
    Xn = flat[keep] / np.maximum(np.linalg.norm(flat[keep], axis=1, keepdims=True), 1e-9)
    pred = (Xn @ cen.T).argmax(axis=1)
    true = lab[keep]
    for i, name in enumerate(names):
        sel = true == ids_inv(ids, name)
        rows[name]["ncm"] = float((pred[sel] == i).mean() * 100) if sel.any() else float("nan")
    return rows

def ids_inv(ids, name):
    for k, v in ids.items():
        if v == name: return k
    return -1

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/d) if d else float("nan")

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a**2).sum()*(b**2).sum())
    return float((a*b).sum()/d) if d else float("nan")

def main():
    allrows = {}
    for ds, cfg in DATASETS.items():
        X, G = load(cfg)
        rows = features(X, G, cfg["ids"], cfg["patch"])
        print(f"\n{'='*100}\n### {ds}  ({X.shape[2]} 波段, {X.shape[0]}×{X.shape[1]}, patch={cfg['patch']})\n")
        hdr = f"{'类':34s}{'n':>7s}{'纯度':>8s}{'可混性':>9s}{'纯光谱':>8s}{'复现':>14s}{'波动σ':>8s}{'论文':>8s}{'偏差':>8s}"
        print(hdr); print("-"*len(hdr))
        recs = []
        for name in sorted(rows, key=lambda x: -rows[x]["n"]):
            r = rows[name]
            rm, rs = REPRO[ds].get(name, (float("nan"),)*2)
            pp = PAPER[ds].get(name, float("nan"))
            delta = rm - pp
            print(f"{name:34s}{r['n']:7d}{r['purity']*100:7.1f}%{r['maxcos']:9.4f}{r['ncm']:7.1f}%"
                  f"{rm:8.2f}±{rs:<5.2f}{rs:8.2f}{pp:8.2f}{delta:+8.2f}")
            recs.append((name, r, rm, rs, pp, delta))
        allrows[ds] = recs

        n   = np.array([r[1]["n"] for r in recs], float)
        pur = np.array([r[1]["purity"] for r in recs], float)
        mcx = np.array([r[1]["maxcos"] for r in recs], float)
        ncm = np.array([r[1]["ncm"] for r in recs], float)
        acc = np.array([r[2] for r in recs], float)
        sd  = np.array([r[3] for r in recs], float)

        print(f"\n  ── {ds} 相关性（Spearman ρ / Pearson r）──")
        for label, x in (("像素数 n", n), ("邻域纯度", pur), ("光谱可混性 maxcos", mcx),
                         ("★纯光谱 NCM 精度", ncm)):
            print(f"    {label:20s} vs 复现精度 : ρ={spearman(x,acc):+.3f}  r={pearson(x,acc):+.3f}")
            print(f"    {label:20s} vs 跨seed波动: ρ={spearman(x,sd):+.3f}   r={pearson(x,sd):+.3f}")

        # 分层
        print(f"\n  ── {ds} 分层 ──")
        stable_hi  = [r[0] for r in recs if r[3] < 5 and r[2] >= 90]
        stable_lo  = [r[0] for r in recs if r[3] < 5 and r[2] <  90]
        unstable   = [r[0] for r in recs if r[3] >= 15]
        print(f"    稳定且高 (σ<5, ≥90)   : {stable_hi}")
        print(f"    稳定但差 (σ<5, <90)   : {stable_lo}")
        print(f"    波动大   (σ≥15)       : {unstable}")

        # CV（论文 Fig.9 口径：逐类精度跨类的变异系数 = std/mean）
        cv_paper = {"Indian": 0.29, "LongKou": 0.55}[ds]
        cv_repro = float(acc.std(ddof=0) / max(acc.mean(), 1e-9))
        print(f"\n  ── {ds} 逐类变异系数 CV（论文 Fig.9 口径，粗读值）──")
        print(f"    论文 Ours  ≈ {cv_paper:.2f}   （论文自述为所有方法中最低 = 最稳）")
        print(f"    复现(3seed) = {cv_repro:.2f}    {'✔ 更稳' if cv_repro < cv_paper else '✘ 更不稳'}")
        print(f"    复现平均跨seed σ = {sd.mean():.2f}（逐类）")

if __name__ == "__main__":
    main()
