import re, glob, os, hashlib
import numpy as np

OA = re.compile(r"unseen OA is ([\d.]+), unseen AA is ([\d.]+), kappa is (\S+)")
AA = re.compile(r"unseen per AA \[([^\]]+)\], kappa is (\S+)")

def read(f):
    pts, cur = [], None
    for line in open(f, encoding="utf-8", errors="ignore"):
        m = OA.search(line)
        if m:
            cur = {"oa": float(m.group(1)), "aa": float(m.group(2)), "k": m.group(3)}; continue
        m = AA.search(line)
        if m and cur:
            cur["per"] = [float(x) for x in m.group(1).split()]
            pts.append(cur); cur = None
    return pts

IND_ORDER = ["Grass-trees","Buildings-Grass-Trees-Drives","Corn-notill","Grass-pasture-mowed","Corn","Woods",
             "Soybean-mintill","Soybean-clean","Stone-Steel-Towers","Hay-windrowed","Wheat","Alfalfa","Oats",
             "Soybean-notill","Corn-mintill","Grass-pasture"]
LK_ORDER  = ["Corn","Cotton","Sesame","Broad-leaf soybean","Narrow-leaf soybean","Rice","Water","Mixed weed","Roads and houses"]
ORDERS = {"Indian": IND_ORDER, "LongKou": LK_ORDER}

# 折 -> main.py 收到的 unseen 原始字符串（与批跑脚本一致）
FOLD_STR = {
 "Indian": {
  "G1":"Corn-notill,Hay-windrowed,Woods",
  "G2":"Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill",
  "G3":"Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill",
  "G4":"Corn,Alfalfa,Grass-pasture",
  "G5":"Wheat,Oats,Soybean-notill",
  "G6":"Soybean-clean"},
 "LongKou": {
  "G1":"Corn,Cotton,Water",
  "G2":"Sesame,Broad-leaf soybean,Rice",
  "G3":"Narrow-leaf soybean,Mixed weed,Roads and houses"},
}

def h(s): return hashlib.md5(s.encode()).hexdigest()[:4]

def report(arm_dir, ds, label):
    print(f"\n=== {label}  [{arm_dir}] ===")
    allc, oas, details = [], [], {}
    for g, s in FOLD_STR[ds].items():
        f = os.path.join(arm_dir, f"log_{ds}_seed42_g{h(s)}.txt")
        if not os.path.exists(f):
            print(f"  {g}: ✗ 缺文件 {os.path.basename(f)}"); continue
        pts = read(f)
        if not pts:
            print(f"  {g}: ✗ 无评测点"); continue
        best = max(pts, key=lambda p: p["oa"])
        oc = [c for c in ORDERS[ds] if c in s.split(",")]
        if len(oc) != len(best["per"]):
            print(f"  {g}: ✗ 类数不匹配 (数组 {len(best['per'])}, 期望 {len(oc)})"); continue
        d = dict(zip(oc, best["per"]))
        allc += list(d.values()); oas.append(best["oa"]); details[g] = (d, best)
        per = " ".join(f"{c[:14]}={v:5.2f}" for c, v in d.items())
        print(f"  {g}: 点数={len(pts):2d} 最优OA={best['oa']:6.2f} AA={best['aa']:6.2f} κ={best['k']:>8}  {per}")
    if oas:
        print(f"  >> fold-mean OA = {np.mean(oas):6.2f}   class-mean AA = {np.mean(allc):6.2f}  (类数 {len(allc)})")
        return {"oa": float(np.mean(oas)), "aa": float(np.mean(allc)), "details": details}
    return None

print("########## LongKou A/B · seed 42 ##########")
base = report("实验日志_2026-10-22/lk_base_seed42", "LongKou", "LongKou base")
bal  = report("实验日志_2026-10-22/lk_bal_seed42",  "LongKou", "LongKou bal")
if base and bal:
    print(f"\n>>> Δ(bal − base):  fold-mean OA {bal['oa']-base['oa']:+.2f}   class-mean AA {bal['aa']-base['aa']:+.2f}")
    print("\n  逐类 Δ(bal − base):")
    for g in FOLD_STR["LongKou"]:
        if g in base["details"] and g in bal["details"]:
            for c in base["details"][g][0]:
                dv = bal["details"][g][0][c] - base["details"][g][0][c]
                print(f"    {g} {c:<22} base {base['details'][g][0][c]:6.2f} → bal {bal['details'][g][0][c]:6.2f}  {dv:+7.2f}")

print("\n\n########## IP λ 裁决 · seed 42 ##########")
a4 = report("实验日志_2026-10-22/ip_lc0.4_seed42", "Indian", "IP λ=0.4")
a6 = report("实验日志_2026-10-22/ip_lc0.6_seed42", "Indian", "IP λ=0.6")
if a4 and a6:
    print(f"\n>>> Δ(0.4 − 0.6):  fold-mean OA {a4['oa']-a6['oa']:+.2f}   class-mean AA {a4['aa']-a6['aa']:+.2f}")
    print(f">>> 论文 IP class-mean AA = 81.45")
    print(f"    0.4 的偏差 {a4['aa']-81.45:+.2f}   0.6 的偏差 {a6['aa']-81.45:+.2f}")
