#!/usr/bin/env python3
"""把 run_paperfolds.sh 的日志抽成对照表。

每个 log_<dataset>_seed<seed>_g<hash>.txt 里，每个评测点写两行：
    [ts] [INFO]: unseen OA is X, unseen AA is Y, kappa is Z
    [ts] [INFO]: unseen per AA [a b c], kappa is Z
本脚本按 g-hash 归属到折，输出每折的「最优评测点」与「末次评测点」，以及逐类。
"""
import re
import glob
import os
import json
import sys

OA_RE = re.compile(r"unseen OA is ([\d.]+), unseen AA is ([\d.]+), kappa is (\S+)")
AA_RE = re.compile(r"unseen per AA \[([^\]]+)\], kappa is (\S+)")

def tofloat(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")   # 单类折（IP G6）的 kappa 是 nan

FOLDS = {
    "Indian": {
        "epochs": 100, "ckpt": 10,
        "G1": "Corn-notill,Hay-windrowed,Woods",
        "G2": "Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill",
        "G3": "Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill",
        "G4": "Corn,Alfalfa,Grass-pasture",
        "G5": "Wheat,Oats,Soybean-notill",
        "G6": "Soybean-clean",
    },
    "LongKou": {
        "epochs": 50, "ckpt": 5,
        "G1": "Corn,Cotton,Water",
        "G2": "Sesame,Broad-leaf soybean,Rice",
        "G3": "Narrow-leaf soybean,Mixed weed,Roads and houses",
    },
}

# runs/<dataset>_<G>.log 最后一行 [start]/[done] 给出跑完的折名单
done_order = []
try:
    with open("runs/driver.log") as f:
        for line in f:
            m = re.match(r"\[(start|done)\s*\] (\S+)\s", line)
            if m:
                done_order.append((m.group(1), m.group(2)))
except FileNotFoundError:
    pass
started = {t for k, t in done_order if k == "start"}
finished = {t for k, t in done_order if k == "done"}

# 读 loger 文件，按文件切分；用 config 的 unseen_classes 反查是哪一折
def read_points(path):
    pts = []
    cur = None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = OA_RE.search(line)
            if m:
                cur = {"oa": float(m.group(1)), "aa": float(m.group(2)), "kappa": tofloat(m.group(3))}
                continue
            m = AA_RE.search(line)
            if m and cur is not None:
                per = [float(x) for x in m.group(1).split()]
                cur["per_aa"] = per
                pts.append(cur)
                cur = None
    return pts

# 建立 hash -> 折 的映射。main.py 的日志后缀 = md5(raw unseen 字符串)[:4]
import hashlib

def fold_hash(unseen_str):
    return hashlib.md5(unseen_str.encode()).hexdigest()[:4]

mapping = {}   # filename -> "Dataset_GN"
for ds, cfg in FOLDS.items():
    for g in [k for k in cfg if k.startswith("G")]:
        fn = f"log_{ds}_seed42_g{fold_hash(cfg[g])}.txt"
        mapping[fn] = f"{ds}_{g}"

# 校验：实际存在的日志文件是否都能对上
existing = set(glob.glob("log_*_seed42_g*.txt"))
unmatched = existing - set(mapping)
if unmatched:
    print("!! 以下日志文件对不上任何折（unseen 字符串可能不同）：")
    for f in sorted(unmatched):
        print("   ", f)
    print()

rows = []
for ds, cfg in FOLDS.items():
    for g in [k for k in cfg if k.startswith("G")]:
        fn = f"log_{ds}_seed42_g{fold_hash(cfg[g])}.txt"
        if not os.path.exists(fn):
            continue
        pts = read_points(fn)
        if not pts:
            continue
        best = max(pts, key=lambda p: p["oa"])
        rows.append({
            "fold": f"{ds}_{g}", "file": fn, "n": len(pts),
            "expected": cfg["epochs"] // cfg["ckpt"],
            "best": best, "last": pts[-1], "all": pts,
            "classes": cfg[g].split(","),
        })

def fmt(p):
    per = " ".join(f"{v:6.2f}" for v in p.get("per_aa", []))
    return f"{p['oa']:6.2f} {p['aa']:6.2f} {p['kappa']:7.4f}  [{per}]"

if not rows:
    print("!! 未能定位折 — 需要 fold_map.json（由 check_folds.py 生成）")
    print("   现有日志文件：")
    for f in sorted(glob.glob("log_*_seed42_g*.txt")):
        print("     ", f, len(read_points(f)), "个评测点")
    sys.exit(0)

print(f"{'折':<12}{'点数':>5}  {'—— 最优评测点 (OA / AA / Kappa / 逐类) ——':<52}")
print("-" * 100)
for r in rows:
    flag = "" if r["n"] == r["expected"] else f"  ⚠️ 应 {r['expected']} 点"
    print(f"{r['fold']:<12}{r['n']:>5}  {fmt(r['best'])}{flag}")
    print(f"{'':<12}{'':>5}  末次: {fmt(r['last'])}")

print()
print("=== 全 checkpoint 明细 ===")
for r in rows:
    print(f"\n{r['fold']}  ({', '.join(r['classes'])})")
    for i, p in enumerate(r["all"], 1):
        print(f"  #{i:<2} {fmt(p)}")

print()
print("=== 按论文协议汇总（各折取最优评测点后按折平均）===")
for ds in FOLDS:
    allsel = [r for r in rows if r["fold"].startswith(ds + "_")]
    sel = [r for r in allsel if r["n"] == r["expected"]]
    skipped = [r["fold"] for r in allsel if r["n"] != r["expected"]]
    if not sel:
        continue
    oa = sum(r["best"]["oa"] for r in sel) / len(sel)
    aa = sum(r["best"]["aa"] for r in sel) / len(sel)
    kps = [r["best"]["kappa"] for r in sel if r["best"]["kappa"] == r["best"]["kappa"]]
    kp = sum(kps) / len(kps) if kps else float("nan")
    print(f"  {ds:<9} {len(sel)} 折  OA {oa:6.2f}   AA {aa:6.2f}   Kappa {kp:.4f}"
          f"   (kappa 基于 {len(kps)} 折)")
    if skipped:
        print(f"            ⏳ 未计入（点数不足）：{', '.join(skipped)}")
