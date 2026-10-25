#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测点轨迹分析：解析 复现结果_论文折.md §三 的 9 折 × 10 评测点明细。

可回答：
  · 「取最优」到底取在第几个点？最优点是否集中在早期（=训练还没跑完就已经最好）？
  · best-vs-last 差多少？后期是否系统性下滑（=过拟合信号）？
  · 相邻评测点之间抖动多大？（=评测稳定性下限）
  · 在 seed 42 内，波动主要来自「折之间」还是「评测点之间」？（方差分解）
"""
import re, statistics, sys
from pathlib import Path

# Windows 默认 GBK 控制台 print 不了「−」(U+2212) 等字符，直接抛 UnicodeEncodeError；
# 统一改走 UTF-8，避免分析在默认终端下崩掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path(__file__).with_name("复现结果_论文折.md")
FOLD_RE = re.compile(r"^### (IP|WHHL) (G\d)")
ROW_RE = re.compile(r"^\|\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*\*{0,2}([\d.]+)\*{0,2}\s*\|\s*\*{0,2}([\d.]+)")
# 单类折的压缩行，如「| 1–10 | 100.00 | 100.00 | nan | 100.00 |」
ROW_RANGE_RE = re.compile(r"^\|\s*\d+\s*[–-]\s*\d+\s*\|")

def parse():
    """-> (可分析折, 评测点不足被丢弃的折, 压缩行折)

    压缩行折（单类折）没有逐点轨迹，按协议不计入统计；但仍要报出来，
    否则 9 折悄悄变成 8 折而汇总行仍按 8 折给结论。
    """
    folds, cur, compressed = {}, None, []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        m = FOLD_RE.match(line)
        if m:
            cur = f"{m.group(1)} {m.group(2)}"
            folds[cur] = []
            continue
        if cur and line.startswith("|"):
            r = ROW_RE.match(line)
            if r:
                folds[cur].append((int(r.group(1)), float(r.group(2)), float(r.group(3))))
            elif ROW_RANGE_RE.match(line) and not folds[cur]:
                compressed.append(cur)
    kept = {k: v for k, v in folds.items() if len(v) >= 2}
    dropped = [k for k, v in folds.items()
               if len(v) < 2 and k not in kept and k not in compressed]
    return kept, dropped, compressed

def main():
    folds, dropped, compressed = parse()
    if not folds:
        print("解析失败：检查 §三 表格式"); sys.exit(1)

    print("=" * 100)
    print("### 逐折轨迹（seed 42，口径 = 每 10 轮/WHHL 每 5 轮一个评测点）\n")
    if compressed:
        print(f"  ℹ️ 压缩行折（单类，无逐点轨迹，按协议不计入）：{compressed}")
    if dropped:
        print(f"  ⚠️ 评测点不足 2 个、未计入统计的折：{dropped}")
    print()
    hdr = f"{'折':10s}{'点数':>5s}{'最优点#':>8s}{'最优OA':>9s}{'末点OA':>9s}{'best−last':>11s}{'相邻|ΔOA|':>11s}{'后5点斜率':>11s}"
    print(hdr); print("-" * len(hdr))

    gaps, best_idx, jitters, slopes = [], [], [], []
    for name, pts in folds.items():
        oa = [p[1] for p in pts]
        bi = max(range(len(oa)), key=lambda i: oa[i])
        gap = oa[bi] - oa[-1]
        jit = statistics.mean(abs(oa[i+1] - oa[i]) for i in range(len(oa)-1))
        tail = oa[-5:] if len(oa) >= 5 else oa
        # 最小二乘斜率（后 5 点）
        n = len(tail); xs = list(range(n)); mx = sum(xs)/n; my = sum(tail)/n
        den = sum((x-mx)**2 for x in xs)
        slope = sum((xs[i]-mx)*(tail[i]-my) for i in range(n))/den if den else 0.0
        print(f"{name:10s}{len(oa):5d}{bi+1:8d}{oa[bi]:9.2f}{oa[-1]:9.2f}{gap:+11.2f}{jit:11.2f}{slope:+11.3f}")
        gaps.append(gap); best_idx.append(bi+1); jitters.append(jit); slopes.append(slope)

    print(f"\n  汇总：平均最优点在第 {statistics.mean(best_idx):.1f} 个评测点"
          f"（共 10）｜平均 best−last = {statistics.mean(gaps):+.2f} 点｜"
          f"平均相邻抖动 = {statistics.mean(jitters):.2f} 点"
          f"（未取最优时，单点测量的噪声地板）｜后5点平均斜率 = {statistics.mean(slopes):+.3f}")

    early = sum(1 for b in best_idx if b <= 3)
    late  = sum(1 for b in best_idx if b >= 8)
    print(f"  最优落在前 3 个点：{early}/{len(best_idx)} 折；落在最后 3 个点：{late}/{len(best_idx)} 折")

    # ---- 方差分解（seed 42）：总波动里多少来自折之间、多少来自评测点之间 ----
    allvals = [p[1] for pts in folds.values() for p in pts]
    gm = statistics.mean(allvals)
    between_fold = sum(len(pts) * (statistics.mean([p[1] for p in pts]) - gm)**2 for pts in folds.values())
    within = sum((p[1] - statistics.mean([q[1] for q in pts]))**2 for pts in folds.values() for p in pts)
    tot = between_fold + within
    print(f"\n  ── seed 42 方差分解（OA）──")
    print(f"    总平方和 = {tot:,.0f}")
    print(f"    折之间   = {between_fold:,.0f}  （{between_fold/tot*100:.1f}%）")
    print(f"    折内(评测点) = {within:,.0f}  （{within/tot*100:.1f}%）")

    # ---- 过拟合检验：末点是否显著低于该折中位数 ----
    worse = [n for n, pts in folds.items()
             if pts[-1][1] < statistics.median([p[1] for p in pts])]
    print(f"\n  ── 末点低于该折中位数的折：{len(worse)}/{len(folds)} ──")
    print(f"    {worse}")

    # ---- 关键判别：最优点的红利，是不是「纯噪声的极值」？ ----
    # 若 10 个点的波动近似 iid 正态，则 E[(max−mean)/sd] ≈ 1.54
    # 而 E[(max−last)/sd] ≈ 1.54（last 是其中一个样本，均值 0），ratio 应 ≈1
    print(f"\n  ── 判别：最优红利 vs 纯噪声极值 ──")
    print(f"    {'折':10s}{'(max−mean)/sd':>16s}{'(max−last)/sd':>16s}{'(max−2ndmax)/sd':>18s}")
    r1, r2 = [], []
    for name, pts in folds.items():
        oa = sorted([p[1] for p in pts], reverse=True)
        sd = statistics.pstdev(oa)
        if sd == 0 or len(oa) < 3: continue
        a = (oa[0] - statistics.mean(oa)) / sd
        b = (oa[0] - pts[-1][1]) / sd
        c = (oa[0] - oa[1]) / sd
        r1.append(a); r2.append(b)
        print(f"    {name:10s}{a:16.3f}{b:16.3f}{c:18.3f}")
    print(f"    实测均值：  (max−mean)/sd = {statistics.mean(r1):.3f}   (max−last)/sd = {statistics.mean(r2):.3f}")
    print(f"    纯噪声(iid 正态, n=10)理论值： 1.538   和 1.538")
    print(f"    → 实测若接近理论值，说明「最优评测点」拿到的主要是**噪声的极值**，不是训练质量的差异。")

    # 极差/sd：判别「独立抽样」还是「成簇跳变」
    rng = []
    for name, pts in folds.items():
        oa = [p[1] for p in pts]
        sd = statistics.pstdev(oa)
        if sd: rng.append((max(oa)-min(oa))/sd)
    print(f"\n  ── 轨迹形态判别：极差/sd ──")
    print(f"    实测 = {statistics.mean(rng):.2f}   独立抽样(iid 正态 n=10) 理论 ≈ 3.08")
    print(f"    实测 {'>' if statistics.mean(rng) > 3.08 else '<'} 3.08 ⇒ "
          f"{'分布比正态更散（存在低位离群点）' if statistics.mean(rng) > 3.08 else '比正态更集中'}")
    m1, mr = statistics.mean(r1), statistics.mean(rng)
    print(f"    配合上一格 (max−mean)/sd = {m1:.2f} vs 1.54（iid 理论值）")
    if m1 < 1.54 and mr > 3.08:
        print(f"    ⇒ 两个量都不符合「独立噪声」画像。轨迹更像**成簇的水平跳变**：")
        print(f"      模型在一段时间停在一个水平，偶尔整体跳到另一个水平。")
        print(f"      含义：「取最优」选中的是**水平跳变的高位**，属系统性不稳，")
        print(f"            不是重复测量就能平均掉的随机噪声。")
    elif m1 >= 1.54:
        print(f"    ⇒ (max−mean)/sd ≥ iid 理论值：「取最优」的红利与**噪声极值**相当，")
        print(f"      不能据此推断存在系统性水平跳变。")
    else:
        print(f"    ⇒ 极差/sd = {mr:.2f} 未超过 iid 理论值 3.08，轨迹比独立抽样更集中；")
        print(f"      「成簇跳变」的解释不成立，需结合相邻点的相关性另行判断。")

    print(f"\n  解读：末点若普遍低于中位数 → 后期在退化（过拟合/漂移）；")
    print(f"        平均 best−last 越大 → 「取最优」这条协议吃到的红利越多。")

if __name__ == "__main__":
    main()
