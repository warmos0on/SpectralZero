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

LK_ORDER = ["Corn","Cotton","Sesame","Broad-leaf soybean","Narrow-leaf soybean","Rice","Water","Mixed weed","Roads and houses"]
FOLDS = {"G1":"Corn,Cotton,Water",
         "G2":"Sesame,Broad-leaf soybean,Rice",
         "G3":"Narrow-leaf soybean,Mixed weed,Roads and houses"}
def h(s): return hashlib.md5(s.encode()).hexdigest()[:4]

def analyze(arm, seed):
    oas, per_class = [], {}
    for g, s in FOLDS.items():
        f = f"实验日志_2026-10-22/lk_{arm}_seed{seed}/log_LongKou_seed{seed}_g{h(s)}.txt"
        if not os.path.exists(f):
            return None
        pts = read(f)
        if not pts:
            return None
        best = max(pts, key=lambda p: p["oa"])
        oc = [c for c in LK_ORDER if c in s.split(",")]
        for c, v in zip(oc, best["per"]):
            per_class.setdefault(c, []).append(v)
        oas.append(best["oa"])
    aa = float(np.mean([np.mean(v) for v in per_class.values()]))
    return {"oa": float(np.mean(oas)), "aa": aa,
            "per": {c: float(np.mean(v)) for c, v in per_class.items()}}

print("LongKou 类平衡采样 A/B —— 逐 seed")
print(f"{'seed':<6}{'arm':<8}{'fold-mean OA':>14}{'class-mean AA':>16}")
rows = {}
for seed in (42, 123):
    for arm in ("base", "bal"):
        r = analyze(arm, seed)
        if r is None:
            print(f"{seed:<6}{arm:<8}{'（不完整）':>14}")
            continue
        rows[(seed, arm)] = r
        print(f"{seed:<6}{arm:<8}{r['oa']:>14.2f}{r['aa']:>16.2f}")

print("\nΔ(bal − base) 逐 seed")
for seed in (42, 123):
    if (seed, "base") in rows and (seed, "bal") in rows:
        b, a = rows[(seed, "base")], rows[(seed, "bal")]
        print(f"  seed {seed}:  OA {a['oa']-b['oa']:+7.2f}    AA {a['aa']-b['aa']:+7.2f}")

print("\n两 seed 汇总（mean ± std）")
for arm in ("base", "bal"):
    oas = [rows[(s, arm)]["oa"] for s in (42, 123) if (s, arm) in rows]
    aas = [rows[(s, arm)]["aa"] for s in (42, 123) if (s, arm) in rows]
    if oas:
        print(f"  {arm:<5} OA {np.mean(oas):6.2f} ± {np.std(oas):5.2f}   AA {np.mean(aas):6.2f} ± {np.std(aas):5.2f}")

print("\n逐类（两 seed 平均）")
print(f"{'类':<24}{'base':>8}{'bal':>8}{'Δ':>8}")
for c in LK_ORDER:
    b = np.mean([rows[(s,'base')]['per'][c] for s in (42,123) if (s,'base') in rows])
    a = np.mean([rows[(s,'bal')]['per'][c] for s in (42,123) if (s,'bal') in rows])
    print(f"{c:<24}{b:>8.2f}{a:>8.2f}{a-b:>+8.2f}")
