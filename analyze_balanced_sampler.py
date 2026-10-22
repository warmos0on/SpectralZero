#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""类平衡采样 A/B 综合分析（两套配置各一份）

数据来源：
  A) 旧配置（patch 11 / lr 2e-4 / λ_sim 0.2 / lambda_clip 0.8 / batch 256 / dynamic_fusion ON / 20 轮）
     —— 本地同机复刻，`_run_oldregime/logs_base/`(基线) vs `_run_oldregime/logs/`(平衡)
  B) 论文配置（patch 15 / lr 1e-4 / λ_sim 0.5 / lambda_clip 0.6 / batch 1024 / 四开关全关 / 100 轮）
     —— 基线 3 seed 在 复现结果_论文折.md；平衡组在远端 logs_bal（跨机）

输出：配对 Δ（末点/最优/均值三种口径）、稳健口径汇总、逐类归因、统计检验。
"""
import re, os, glob, statistics as st, hashlib

OA = re.compile(r"unseen OA is ([-\d.]+), unseen AA is ([-\d.]+)")
AA = re.compile(r"per AA \[([^\]]*)\]")
GROUPS = {"G1": "Corn-notill,Hay-windrowed,Woods",
          "G2": "Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill",
          "G3": "Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill",
          "G4": "Corn,Alfalfa,Grass-pasture",
          "G5": "Wheat,Oats,Soybean-notill", "G6": "Soybean-clean"}
H2G = {hashlib.md5(s.encode()).hexdigest()[:4]: g for g, s in GROUPS.items()}
# datasets.py label_values 的字典字面量键序（= 日志 per-AA 顺序）
ORDER = ["Grass-trees", "Buildings-Grass-Trees-Drives", "Corn-notill", "Grass-pasture-mowed",
         "Corn", "Woods", "Soybean-mintill", "Soybean-clean", "Stone-Steel-Towers",
         "Hay-windrowed", "Wheat", "Alfalfa", "Oats", "Soybean-notill", "Corn-mintill", "Grass-pasture"]
PX = {"Grass-trees": 730, "Buildings-Grass-Trees-Drives": 386, "Corn-notill": 1428,
      "Grass-pasture-mowed": 28, "Corn": 237, "Woods": 1265, "Soybean-mintill": 2455,
      "Soybean-clean": 593, "Stone-Steel-Towers": 93, "Hay-windrowed": 478, "Wheat": 205,
      "Alfalfa": 46, "Oats": 20, "Soybean-notill": 972, "Corn-mintill": 830, "Grass-pasture": 483}


def parse_dir(d):
    """-> {(fold,seed): [(oa,aa,per), ...]}  (按评测点顺序，不排序)"""
    out = {}
    if not os.path.isdir(d):
        return out
    for f in glob.glob(os.path.join(d, "log_Indian_seed*_g*.txt")):
        m = re.match(r"log_Indian_seed(\d+)_g([0-9a-f]{4})\.txt", os.path.basename(f))
        if not m:
            continue
        seed, g = int(m.group(1)), H2G.get(m.group(2))
        if g is None:
            continue
        pts, cur = [], None
        for line in open(f, encoding="utf-8", errors="replace"):
            x = OA.search(line)
            if x:
                if cur:
                    pts.append(tuple(cur))
                cur = [float(x.group(1)), float(x.group(2)), []]
                continue
            if cur is not None and not cur[2]:
                a = AA.search(line)
                if a:
                    cur[2] = [float(y) for y in a.group(1).split()]
        if cur:
            pts.append(tuple(cur))
        out[(g, seed)] = pts
    return out


def modes(pts):
    """三种口径 -> (末点, 最优OA点, 均值) 各自的 (oa, aa)"""
    if not pts:
        return None
    last = pts[-1]
    best = max(pts, key=lambda p: p[0])
    m_oa = st.mean(p[0] for p in pts)
    m_aa = st.mean(p[1] for p in pts)
    return last, best, (m_oa, m_aa)


def report(tag, base, bal, seeds):
    print("=" * 100)
    print(f"### {tag}\n")
    rows = []
    for g in GROUPS:
        for s in seeds:
            if (g, s) in base and (g, s) in bal:
                b, n = modes(base[(g, s)]), modes(bal[(g, s)])
                if b and n:
                    rows.append((g, s, b, n))
    if not rows:
        print("  （无共同数据）\n")
        return None
    print(f"{'折':4s}{'seed':>5s}{'ΔOA末点':>10s}{'ΔOA最优':>10s}{'ΔOA均值':>10s}{'ΔAA最优':>10s}")
    print("-" * 56)
    for g, s, b, n in rows:
        dl = n[0][0] - b[0][0]
        db = n[1][0] - b[1][0]
        dm = n[2][0] - b[2][0]
        da = n[1][1] - b[1][1]
        print(f"{g:4s}{s:5d}{dl:+10.2f}{db:+10.2f}{dm:+10.2f}{da:+10.2f}")
    nonG6 = [r for r in rows if r[0] != "G6"]
    print("-" * 56)
    if nonG6:
        for lbl, idx in (("ΔOA 末点", 0), ("ΔOA 最优", 1), ("ΔOA 均值", 2)):
            v = []
            for g, s, b, n in nonG6:
                v.append((n[idx][0] - b[idx][0]) if idx < 2 else (n[2][0] - b[2][0]))
            m = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            se = sd / len(v) ** 0.5
            t = m / se if se else 0
            print(f"  {lbl}: 均值 {m:+.2f}  sd {sd:.2f}  SE {se:.2f}  t {t:+.2f}  "
                  f"{'显著' if abs(t) > 2.78 else '不显著'}(df={len(v)-1})")
    return rows


def per_class(base, bal, seeds):
    """逐类：跨可用的 (折,seed) 对，取各自最优点的逐类值再平均"""
    cb, cn = {}, {}
    for g in GROUPS:
        subs = [c for c in ORDER if c in GROUPS[g].split(",")]
        for s in seeds:
            for src, dst in ((base, cb), (bal, cn)):
                if (g, s) not in src:
                    continue
                pts = src[(g, s)]
                if not pts:
                    continue
                best = max(pts, key=lambda p: p[0])
                if len(best[2]) == len(subs):
                    for c, v in zip(subs, best[2]):
                        dst.setdefault(c, []).append(v)
    print("\n" + "=" * 100)
    print("### 逐类（各自最优点的逐类值，跨折/seed 平均）\n")
    print(f"{'类':34s}{'像素':>6s}{'基线':>8s}{'n':>3s}{'平衡':>8s}{'n':>3s}{'Δ':>9s}")
    print("-" * 72)
    out = {}
    for c in ORDER:
        if c not in cb or c not in cn:
            continue
        b, n = st.mean(cb[c]), st.mean(cn[c])
        out[c] = (b, n, n - b)
        print(f"{c:34s}{PX[c]:6d}{b:8.2f}{len(cb[c]):3d}{n:8.2f}{len(cn[c]):3d}{n-b:+9.2f}")
    if out:
        d_all = st.mean(v[2] for v in out.values())
        d_wo = st.mean(v[2] for c, v in out.items()
                       if c not in ("Oats", "Stone-Steel-Towers", "Hay-windrowed"))
        wb = sum(v[0] * PX[c] for c, v in out.items()) / sum(PX[c] for c in out)
        wn = sum(v[1] * PX[c] for c, v in out.items()) / sum(PX[c] for c in out)
        print("-" * 72)
        print(f"  AA 逐类平均(16 类)       基线 {st.mean(v[0] for v in out.values()):.2f}  平衡 {st.mean(v[1] for v in out.values()):.2f}  Δ {d_all:+.2f}")
        print(f"  AA 剔除 3 个最不稳的类     Δ {d_wo:+.2f}")
        print(f"  加权 OA（按像素）         基线 {wb:.2f}  平衡 {wn:.2f}  Δ {wn-wb:+.2f}")
    return out


if __name__ == "__main__":
    R = os.path.dirname(os.path.abspath(__file__))
    print("读取数据中…\n")
    # A) 旧配置 —— 本地同机
    ob = parse_dir(os.path.join(R, "_run_oldregime", "logs_base"))
    on = parse_dir(os.path.join(R, "_run_oldregime", "logs"))
    seeds = sorted({s for (_, s) in set(ob) | set(on)})
    print(f"检测到种子: {seeds}   （基线 {len(ob)} 折 · 平衡 {len(on)} 折）\n")
    report("A) 旧配置 · 本地同机复刻", ob, on, seeds)
    per_class(ob, on, seeds)
