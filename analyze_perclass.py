# -*- coding: utf-8 -*-
"""Phase 1: 逐类准确率 vs 样本数 长尾分析

数据来源：result_IP/ablation_*_run.txt 的 per-AA（3-seed 均值，固定最后 cp）。
类样本数：Indian_gt.mat 中每类像素总数（对 unseen 类即测试集大小）。
输出：result_IP/长尾分析.md + 三张图。
"""
import json, re, sys, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import datasets

GROUPS = {
    "Indian": [
        ["Corn-notill", "Hay-windrowed", "Woods"],
        ["Grass-trees", "Buildings-Grass-Trees-Drives", "Soybean-mintill"],
        ["Grass-pasture-mowed", "Stone-Steel-Towers", "Corn-mintill"],
        ["Corn", "Alfalfa", "Grass-pasture"],
        ["Wheat", "Oats", "Soybean-notill"],
        ["Soybean-clean"],
    ]
}

CONFIGS = {
    "baseline": "result_IP/ablation_base_3seed_run.txt",
    "E_zscore": "result_IP/ablation_E_zscore_only_run.txt",
    "F_balanced": "result_IP/ablation_F_balanced_only_run.txt",
    "C_zs_bal": "result_IP/ablation_C_multi_seed_run.txt",
    "D_best": "result_IP/ablation_D_multi_seed_run.txt",
}

PER_RE = re.compile(r"per-AA\s+([\d. ]+)")


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def parse_per_aa(text):
    """返回 dict {group_idx: [per-AA floats]}，取 summary 区（最后一个 per-AA 每个组只出现一次）。"""
    out = {}
    for m in PER_RE.finditer(text):
        vals = [float(x) for x in m.group(1).split()]
        # summary 行形如  G1 ... | per-AA ...
        line_start = text.rfind("\n", 0, m.start()) + 1
        head = text[line_start:m.start()]
        gm = re.match(r"G(\d)\s", head)
        if gm:
            out[int(gm.group(1))] = vals
    return out


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float(((2 * np.arange(1, n + 1) - n - 1) * x).sum() / (n * x.sum()))


def entropy(x):
    x = np.asarray(x, dtype=float)
    p = x / x.sum()
    return float(-(p[p > 0] * np.log(p[p > 0])).sum())


def effective_num(x, beta=None):
    x = np.asarray(x, dtype=float)
    n = float(x.sum())
    if beta is None:
        beta = (n - 1) / n if n > 1 else 1.0
    r = x * (1 - beta) / (1 - np.power(beta, x))
    r[x == 0] = 0
    return float(r.sum())


def main():
    ds = "Indian"
    HSI, gt, label_values = datasets.get_dataset(dataset_name=ds)
    inv = {v: k for k, v in label_values.items()}
    counts = {}
    for cid in range(1, int(np.max(gt)) + 1):
        name = inv.get(cid, f"class{cid}")
        counts[name] = int(np.sum(gt == cid))

    parsed = {}
    for cfg, path in CONFIGS.items():
        parsed[cfg] = parse_per_aa(read_bytes(path))

    rows = []
    for gi, unseen in enumerate(GROUPS[ds], start=1):
        for j, cls in enumerate(unseen):
            d = {"group": gi, "class": cls, "N": counts[cls]}
            for cfg in CONFIGS:
                vals = parsed.get(cfg, {}).get(gi)
                d[cfg] = vals[j] if vals else float("nan")
            rows.append(d)

    n_arr = np.array([r["N"] for r in rows], dtype=float)
    base = np.array([r["baseline"] for r in rows], dtype=float)
    best = np.array([r["D_best"] for r in rows], dtype=float)
    delta = best - base
    logn = np.log(n_arr)

    r_base_logn = stats.pearsonr(logn, base)
    r_best_logn = stats.pearsonr(logn, best)
    r_delta_logn = stats.pearsonr(logn, delta)
    sp_base = stats.spearmanr(logn, base)
    sp_best = stats.spearmanr(logn, best)
    sp_delta = stats.spearmanr(logn, delta)

    all_counts = np.array(list(counts.values()), dtype=float)
    tail_ratio = all_counts.max() / all_counts[all_counts > 0].min()
    metrics = {
        "总类数": len(all_counts),
        "总像素": int(all_counts.sum()),
        "头尾比 max/min": round(tail_ratio, 1),
        "Gini 系数": round(gini(all_counts), 4),
        "Shannon 熵": round(entropy(all_counts), 4),
        "均匀分布最大熵": round(np.log(len(all_counts)), 4),
        "有效类别数 (Cui et al.)": round(effective_num(all_counts), 1),
        "消融基线 per-AA vs logN Pearson r": round(r_base_logn.statistic, 3),
        "基线 p 值": round(r_base_logn.pvalue, 4),
        "最佳 D per-AA vs logN Pearson r": round(r_best_logn.statistic, 3),
        "D p 值": round(r_best_logn.pvalue, 4),
        "提升 Δ vs logN Pearson r": round(r_delta_logn.statistic, 3),
        "Δ p 值": round(r_delta_logn.pvalue, 4),
        "提升 Δ vs logN Spearman r": round(sp_delta.statistic, 3),
        "Δ Spearman p 值": round(sp_delta.pvalue, 4),
    }

    lines = ["# Indian Pines 长尾分布与逐类准确率分析\n",
             "数据：固定最后 checkpoint、3-seed(42/123/456) 均值；N=该类全图像素数（unseen 测试规模）。\n",
             "## 长尾指标\n"]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 逐类表（按样本数升序）\n",
              "| 组 | 类 | N | 基线 per-AA | D per-AA | Δ |",
              "|----|----|----|----|----|----|"]
    order = np.argsort(n_arr)
    for idx in order:
        r = rows[idx]
        lines.append(f"| G{r['group']} | {r['class']} | {r['N']} | {r['baseline']:.1f} | "
                     f"{r['D_best']:.1f} | {delta[idx]:+.1f} |")
    lines += ["", "## 结论要点\n",
              "- 若基线 per-AA 与 logN 显著负相关（小样本类更差），且 D 配置相关性减弱/提升集中在尾部类，则验证：长尾 seen 分布偏置 seen 训练 → 类别平衡采样缓解 → unseen 迁移提升。"]

    md_path = "result_IP/长尾分析.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    # 图1：类样本分布（log 柱状）
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    names = [r["class"] for r in rows]
    order = np.argsort(n_arr)[::-1]
    xs = np.arange(len(rows))
    colors = ["#3a86ff" if r["N"] > np.median(n_arr) else "#ff9e00" for r in rows]
    ax.bar(xs, n_arr[order], color=[colors[i] for i in order])
    ax.set_xticks(xs)
    ax.set_xticklabels([names[i] for i in order], rotation=60, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("pixels (log)")
    ax.set_title("Indian Pines class sample distribution (long-tail)")
    ax.annotate(f"max/min = {tail_ratio:.0f}x", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="#999"))
    fig.tight_layout()
    fig.savefig("result_IP/fig_longtail_dist.png")
    plt.close(fig)

    # 图2：per-AA vs N（基线 vs D + 回归线）
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    ax.scatter(logn, base, s=60, color="#d62828", label="Baseline (fixed fusion)", zorder=3)
    ax.scatter(logn, best, s=60, color="#1d3557", marker="^", label="Best (balanced+zscore)", zorder=3)
    xs_line = np.linspace(logn.min(), logn.max(), 100)
    z1 = np.polyfit(logn, base, 1)
    z2 = np.polyfit(logn, best, 1)
    ax.plot(xs_line, np.polyval(z1, xs_line), "--", color="#d62828", alpha=0.6)
    ax.plot(xs_line, np.polyval(z2, xs_line), "--", color="#1d3557", alpha=0.6)
    ax.set_xlabel("log(class pixels)")
    ax.set_ylabel("per-class AA (%)")
    ax.set_title("Per-class accuracy vs sample size (3-seed mean)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.text(0.03, 0.97, f"base r={r_base_logn.statistic:.2f} (p={r_base_logn.pvalue:.3f})",
            transform=ax.transAxes, va="top", fontsize=9, color="#d62828")
    ax.text(0.03, 0.90, f"D r={r_best_logn.statistic:.2f} (p={r_best_logn.pvalue:.3f})",
            transform=ax.transAxes, va="top", fontsize=9, color="#1d3557")
    fig.tight_layout()
    fig.savefig("result_IP/fig_perclass_vs_samples.png")
    plt.close(fig)

    # 图3：提升 Δ vs N
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    ax.axhline(0, color="#999", lw=1)
    ax.scatter(logn, delta, s=70, color="#e07a00", zorder=3)
    z3 = np.polyfit(logn, delta, 1)
    ax.plot(xs_line, np.polyval(z3, xs_line), "--", color="#e07a00", alpha=0.7)
    for idx in order:
        ax.annotate(f"{rows[idx]['class']}", (logn[idx], delta[idx]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("log(class pixels)")
    ax.set_ylabel("Δ per-class AA (D - baseline)")
    ax.set_title("Improvement concentrated on tail classes?")
    ax.grid(alpha=0.3)
    ax.text(0.03, 0.95, f"Δ vs logN: r={r_delta_logn.statistic:.2f} (p={r_delta_logn.pvalue:.3f})",
            transform=ax.transAxes, va="top", fontsize=10)
    fig.tight_layout()
    fig.savefig("result_IP/fig_improv_vs_samples.png")
    plt.close(fig)

    print("\n已保存: result_IP/长尾分析.md, fig_longtail_dist.png, fig_perclass_vs_samples.png, fig_improv_vs_samples.png")


if __name__ == "__main__":
    main()
