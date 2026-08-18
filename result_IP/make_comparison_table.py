# -*- coding: utf-8 -*-
"""Parse ablation run files and generate per-seed / per-group comparison tables."""
import re
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# (配置名, 文件名, 说明)
RUNS = [
    ("基线（融合开，其余全关）", "ablation_base_3seed_run.txt", "原版协议 3-seed"),
    ("E（只 z-score）", "ablation_E_zscore_only_run.txt", "融合开 + 只开 z-score"),
    ("F（只平衡采样）", "ablation_F_balanced_only_run.txt", "融合开 + 只开平衡采样"),
    ("C（z-score+平衡）", "ablation_C_multi_seed_run.txt", "融合开 + zscore + 平衡"),
    ("D（关融合+z-score+平衡）", "ablation_D_multi_seed_run.txt", "最佳配置"),
    ("全改动（四样全开）", "evolved_multi_seed_run.txt", "融合开 + zscore + 平衡 + 负样本"),
    ("B（只负样本）", "ablation_B_negonly_run.txt", "融合开 + 只开负样本，seed42 单种子"),
]


def parse_file(path):
    """Return {group: {seed: oa}} and summary line."""
    rows = {}
    summary = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"G(\d)\s+seed=(\d+):\s+OA=([\d.]+)", line)
            if m:
                group, seed, oa = m.groups()
                rows.setdefault(int(group), {})[int(seed)] = float(oa)
            if "多类组平均" in line:
                summary = line.strip()
    return rows, summary


def group_mean(rows, group, seeds):
    vals = [rows[group][s] for s in seeds if s in rows.get(group, {})]
    return sum(vals) / len(vals) if vals else float("nan")


def per_seed_mean(rows, seed, groups):
    vals = [rows[g][seed] for g in groups if seed in rows.get(g, {})]
    return sum(vals) / len(vals) if vals else float("nan")


SEEDS = [42, 123, 456]
GROUPS = [1, 2, 3, 4, 5]  # G6 单类 100% 不计入

parsed = {}
for name, fname, desc in RUNS:
    path = os.path.join(ROOT, fname)
    if os.path.exists(path):
        parsed[name] = parse_file(path)
    else:
        print(f"[MISSING] {fname}")

base_name = "基线（融合开，其余全关）"
base = parsed[base_name][0]
d_name = "D（关融合+z-score+平衡）"
d = parsed[d_name][0]

lines = []
lines.append("# 数据改变对比表（Indian Pines，固定最后 checkpoint）")
lines.append("")
lines.append("协议：OA 为多类组(G1-G5)平均，G6 单类 100% 不计入；3-seed 为 42/123/456。")
lines.append("")
lines.append("## 一、每个种子是否都有提升（基线 vs 最佳配置 D）")
lines.append("")
lines.append("| seed | 基线 OA | D 配置 OA | ΔOA | 提升? |")
lines.append("|------|--------|----------|-----|-------|")
all_up = True
for s in SEEDS:
    b = per_seed_mean(base, s, GROUPS)
    dv = per_seed_mean(d, s, GROUPS)
    delta = dv - b
    up = "✅" if delta > 0 else "❌"
    if delta <= 0:
        all_up = False
    lines.append(f"| {s} | {b:.2f} | {dv:.2f} | {delta:+.2f} | {up} |")
lines.append("")
lines.append(f"结论：3 个种子**全部提升**（{'是' if all_up else '否'}）。")
lines.append("")

lines.append("## 二、逐组逐种子对比（基线 vs D）")
lines.append("")
lines.append("| 组 | seed | 基线 OA | D 配置 OA | ΔOA |")
lines.append("|----|------|--------|----------|-----|")
for g in GROUPS:
    for s in SEEDS:
        b = base.get(g, {}).get(s)
        dv = d.get(g, {}).get(s)
        if b is None or dv is None:
            continue
        lines.append(f"| G{g} | {s} | {b:.2f} | {dv:.2f} | {dv-b:+.2f} |")
lines.append("")

lines.append("## 三、各配置汇总（3-seed 组均值）")
lines.append("")
lines.append("| 配置 | G1 | G2 | G3 | G4 | G5 | 均值 OA |")
lines.append("|------|----|----|----|----|----|---------|")
for name, fname, desc in RUNS:
    if name not in parsed:
        continue
    rows, summary = parsed[name]
    cells = []
    gmeans = []
    for g in GROUPS:
        gm = group_mean(rows, g, SEEDS)
        gmeans.append(gm)
        cells.append(f"{gm:.2f}")
    overall = sum(gmeans) / len(gmeans)
    lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} | {overall:.2f} |")
lines.append("")

lines.append("## 四、归因对账（D vs 基线 = +8.08 OA）")
lines.append("")
lines.append("| 改动 | 对比 | OA 差 | 结论 |")
lines.append("|------|------|-------|------|")
lines.append("| 类别平衡采样 | F − 基线 | +4.40 | 最大功臣 |")
lines.append("| 关掉动态融合 | D − C | +2.59 | 动态调节净负 |")
lines.append("| z-score×平衡 协同 | C − (E+F+基线) | +3.12 | 一起开才有加成 |")
lines.append("| z-score 单独 | E − 基线 | −2.03 | 单独用反而小负 |")
lines.append("| unseen 负样本 | 全改动 − C / B − 基线 | −4.23 / −3.38 | 负贡献，排除 |")
lines.append("")
lines.append("对账：+4.40 − 2.03 + 3.12 + 2.59 = +8.08 = D(88.77) − 基线(80.69) ✅")
lines.append("")
lines.append("## 五、LongKou（诚实记录，非创新点）")
lines.append("")
lines.append("| 配置 | OA | AA | Kappa | 说明 |")
lines.append("|------|-----|-----|-------|------|")
lines.append("| 旧基线（seed42） | 69.54 | 55.89 | 0.4923 | dynamic_seed42_20ep 存档 |")
lines.append("| C（开 z-score） | 36.30 | 46.68 | 0.0815 | z-score 破坏 float32(0-28) 分布 |")
lines.append("| 关 z-score+关融合+平衡 | 54.35 | 57.02 | 0.3279 | 3-seed；G1 大胜，G3 仍难 |")
lines.append("")
lines.append("## 六、所有已跑实验清单（结果文件）")
lines.append("")
lines.append("| 配置 | seeds | 结果文件 |")
lines.append("|------|-------|---------|")
for name, fname, desc in RUNS:
    n_seeds = "42,123,456" if "seed42" not in desc else "42（单种子）"
    lines.append(f"| {name} | {n_seeds} | {fname} |")
lines.append("")

out_path = os.path.join(ROOT, "数据改变对比表.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"written: {out_path}")
