import re, os, hashlib
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

ORDERS = {
 "LongKou": ["Corn","Cotton","Sesame","Broad-leaf soybean","Narrow-leaf soybean","Rice","Water","Mixed weed","Roads and houses"],
 "Indian": ["Grass-trees","Buildings-Grass-Trees-Drives","Corn-notill","Grass-pasture-mowed","Corn","Woods",
            "Soybean-mintill","Soybean-clean","Stone-Steel-Towers","Hay-windrowed","Wheat","Alfalfa","Oats",
            "Soybean-notill","Corn-mintill","Grass-pasture"],
}
FOLD_STR = {
 "LongKou": {"G1":"Corn,Cotton,Water","G2":"Sesame,Broad-leaf soybean,Rice","G3":"Narrow-leaf soybean,Mixed weed,Roads and houses"},
 "Indian": {"G1":"Corn-notill,Hay-windrowed,Woods","G2":"Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill",
            "G3":"Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill","G4":"Corn,Alfalfa,Grass-pasture",
            "G5":"Wheat,Oats,Soybean-notill","G6":"Soybean-clean"},
}
h = lambda s: hashlib.md5(s.encode()).hexdigest()[:4]
BASE = "实验日志_2026-10-22"

def arm(dirname, ds, seed):
    oas, per = [], {}
    for g, s in FOLD_STR[ds].items():
        f = os.path.join(BASE, dirname, f"log_{ds}_seed{seed}_g{h(s)}.txt")
        if not os.path.exists(f): return None
        pts = read(f)
        if not pts: return None
        best = max(pts, key=lambda p: p["oa"])
        oc = [c for c in ORDERS[ds] if c in s.split(",")]
        if len(oc) != len(best["per"]): return None
        for c, v in zip(oc, best["per"]): per.setdefault(c, []).append(v)
        oas.append(best["oa"])
    return {"oa": float(np.mean(oas)), "aa": float(np.mean([np.mean(v) for v in per.values()])),
            "per": {c: float(np.mean(v)) for c, v in per.items()}, "nfold": len(oas)}

def show(ds, seed, arms):
    print(f"\n===== {ds} · seed {seed} =====")
    got = {}
    for name, d in arms.items():
        r = arm(d, ds, seed)
        if r is None: print(f"  {name:<10} （不完整）"); continue
        got[name] = r
        print(f"  {name:<10} {r['nfold']} 折   fold-mean OA {r['oa']:6.2f}   class-mean AA {r['aa']:6.2f}")
    if len(got) >= 2:
        names = list(got)
        ref = names[0]
        print(f"  --- 相对 {ref} 的逐类 Δ ---")
        print(f"  {'类':<22}" + "".join(f"{n:>10}" for n in names))
        for c in ORDERS[ds]:
            if c not in got[ref]["per"]: continue
            row = "".join(f"{got[n]['per'].get(c,float('nan')):10.2f}" for n in names)
            print(f"  {c:<22}{row}")
    return got

lk = show("LongKou", 42, {"base": "lk_base_seed42", "mild(0.5)": "lk_mild_seed42", "bal(1.0)": "lk_bal_seed42"})
ip = show("Indian", 42, {"base(0.6)": "ip_lc0.6_seed42", "bal(1.0)": "ip_bal_seed42"})

print("\n\n===== 汇总 =====")
if lk.get("base") and lk.get("mild(0.5)"):
    print(f"LongKou mild − base : OA {lk['mild(0.5)']['oa']-lk['base']['oa']:+.2f}   AA {lk['mild(0.5)']['aa']-lk['base']['aa']:+.2f}")
if lk.get("base") and lk.get("bal(1.0)"):
    print(f"LongKou bal  − base : OA {lk['bal(1.0)']['oa']-lk['base']['oa']:+.2f}   AA {lk['bal(1.0)']['aa']-lk['base']['aa']:+.2f}")
if ip.get("base(0.6)") and ip.get("bal(1.0)"):
    print(f"IP bal − base(0.6)  : OA {ip['bal(1.0)']['oa']-ip['base(0.6)']['oa']:+.2f}   AA {ip['bal(1.0)']['aa']-ip['base(0.6)']['aa']:+.2f}")
