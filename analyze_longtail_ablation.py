"""汇总长尾消融实验结果：每个 (group, level) 的平衡采样收益 Δ 与长尾程度的关系。

用法:
    python analyze_longtail_ablation.py

输出:
    - result_IP/longtail_ablation/summary.md（含每 seed 明细 + 均值 + 趋势统计）
    - result_IP/longtail_ablation/delta_vs_gini.png（Δ 随 seen 训练 Gini 变化的散点）
"""
import json
import os

import numpy as np

from run_honest_eval import GROUPS
from make_longtail_caps import (
    INDIAN_LABEL_VALUES,
    LEVEL_DEFS,
    effective_counts,
    gini,
    head_tail_ratio,
    natural_train_counts,
    utils,
)


OUT_DIR = os.path.join("result_IP", "longtail_ablation")
TSV = os.path.join(OUT_DIR, "results.tsv")
CAP_DIR = os.path.join("result_IP", "longtail_caps")


def load_rows():
    rows = []
    with open(TSV, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 7:
                rows.append({
                    "group": int(parts[0]),
                    "seed": int(parts[1]),
                    "level": parts[2],
                    "bal": int(parts[3]),
                    "oa": float(parts[4]),
                    "aa": float(parts[5]),
                    "kappa": float(parts[6]),
                })
    return rows


def load_caps():
    caps = {"natural": {name: float("inf") for name in INDIAN_LABEL_VALUES}}
    for level in LEVEL_DEFS:
        path = os.path.join(CAP_DIR, f"level_{level}.json")
        with open(path, encoding="utf-8") as f:
            caps[level] = json.load(f)
    return caps


def main():
    gt = utils.read_mat("./data/Indian_gt.mat")
    nat = natural_train_counts(gt)
    natural_by_name = {name: int(nat[lab]) for name, lab in INDIAN_LABEL_VALUES.items()}
    caps = load_caps()
    rows = load_rows()

    # 已知的自然 bal=1 结果（D 配置，3-seed 组均值，来自历史记录）
    known_natural_bal1 = {
        1: 86.54, 2: 90.77, 3: 86.86, 4: 90.64, 5: 89.03,
    }

    lines = []
    lines.append("# 长尾消融实验汇总（seen 侧训练池长尾程度操控）")
    lines.append("")
    lines.append("协议：Indian Pines，train_num=0.2，固定最后一个 checkpoint，3-seed(42/123/456) 均值。")
    lines.append("说明：在最佳配置 D（关动态融合 + z-score + 类别平衡采样）基础上，"
                 "只对 seen 类训练配额施加不同上限（cap），构造不同长尾程度。")
    lines.append("bal=1 为平衡采样开，bal=0 为平衡采样关（其余配置与 D 完全一致）。")
    lines.append("cap 只压缩头部类样本数，总量随之下降；absolute OA 的下降是预期内的代价。")
    lines.append("")

    lines.append("## 一、训练池长尾程度的操控（全 16 类口径）")
    lines.append("")
    lines.append("| level | cap 上限 | 头尾比 | Gini | 训练像素总数 | 相对自然总量 |")
    lines.append("|-------|---------|--------|------|------------|------------|")
    for level in ["natural", "mild", "mid", "uniform"]:
        eff = effective_counts(natural_by_name, caps[level])
        cap_txt = "-" if level == "natural" else str(LEVEL_DEFS[level])
        lines.append(
            f"| {level:8s} | {cap_txt:7s} | {head_tail_ratio(eff):6.2f}x | "
            f"{gini(list(eff.values())):.4f} | {sum(eff.values())} | "
            f"{sum(eff.values()) / sum(natural_by_name.values()) * 100:.1f}% |")
    lines.append("")

    lines.append("## 二、逐任务明细（每 seed）")
    lines.append("")
    lines.append("| 组 | seed | level | 平衡采样 | OA | AA | Kappa |")
    lines.append("|----|------|-------|---------|----|----|-------|")
    for r in sorted(rows, key=lambda x: (x["group"], x["level"], -x["bal"], x["seed"])):
        lines.append(f"| G{r['group']} | {r['seed']} | {r['level']} | "
                     f"{'开' if r['bal'] else '关'} | {r['oa']:.2f} | {r['aa']:.2f} | {r['kappa']:.4f} |")
    lines.append("")

    lines.append("## 三、每 (组, 水平) 的平衡采样收益 Δ（3-seed 均值）")
    lines.append("")
    lines.append("| 组 | level | seen 训练 Gini | 头尾比 | 训练总量 | bal 关 OA | bal 开 OA | Δ(bal开−关) |")
    lines.append("|----|-------|---------------|--------|---------|---------|---------|-----------|")

    records = []
    for group in [1, 5]:
        unseen = GROUPS["Indian"][group - 1]
        seen = [n for n in natural_by_name if n not in unseen]
        nat_seen = {n: natural_by_name[n] for n in seen}
        for level in ["natural", "mild", "mid", "uniform"]:
            eff_seen = {n: min(natural_by_name[n], caps[level][n]) for n in seen}
            off = [r["oa"] for r in rows if r["group"] == group and r["level"] == level and r["bal"] == 0]
            on = [r["oa"] for r in rows if r["group"] == group and r["level"] == level and r["bal"] == 1]
            if level == "natural" and not on:
                # 自然+bal开 = D 配置，来自历史记录（3-seed 均值，无 seed 明细）
                on = [known_natural_bal1[group]]
            if not off or not on:
                lines.append(f"| G{group} | {level} | {gini(list(eff_seen.values())):.4f} | "
                             f"{head_tail_ratio(eff_seen):.2f}x | {sum(eff_seen.values())} | "
                             f"- | - | 数据不全 |")
                continue
            off_m, on_m = np.mean(off), np.mean(on)
            delta = on_m - off_m
            records.append((group, level, gini(list(eff_seen.values())),
                            head_tail_ratio(eff_seen), sum(eff_seen.values()), off_m, on_m, delta))
            lines.append(
                f"| G{group} | {level} | {records[-1][2]:.4f} | {records[-1][3]:.2f}x | "
                f"{records[-1][4]} | {off_m:.2f} | {on_m:.2f} | **{delta:+.2f}** |")
    lines.append("")

    lines.append("## 四、趋势统计")
    lines.append("")
    if len(records) >= 4:
        gini_vals = np.array([r[2] for r in records])
        delta_vals = np.array([r[7] for r in records])
        r_pearson = np.corrcoef(gini_vals, delta_vals)[0, 1]
        lines.append(f"- 组内跨水平：Δ 与 seen 训练 Gini 的 Pearson 相关 r = {r_pearson:.3f} "
                     f"（{len(records)} 个点）")
        lines.append(f"- 全部 {len(records)} 个点：Δ 均值 {delta_vals.mean():+.2f}，"
                     f"Gini 范围 {gini_vals.min():.3f}~{gini_vals.max():.3f}")
    lines.append("")
    lines.append("### 解读（诚实标注）")
    lines.append("- 若 Δ 随 Gini 下降而单调收窄 → 支持'平衡采样收益来自 seen 长尾偏置'。")
    lines.append("- 若 Δ 在压头后仍保持 → 收益不纯粹来自头尾比，可能来自尾部类过采样/多样性。")
    lines.append("- 注意：cap 同时降低训练总量，绝对 OA 整体下降；Δ 的变化需对照同水平内两个臂，"
                 "总量效应在臂间被抵消，但极端水平下噪声变大。")
    lines.append("")

    out_txt = os.path.join(OUT_DIR, "summary.md")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已生成 {out_txt}")

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        for font_path in (
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyh.ttf",
            r"C:\Windows\Fonts\simhei.ttf",
        ):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                continue
        plt.rcParams["font.family"] = "Microsoft YaHei, SimHei, sans-serif"
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for group in [1, 5]:
            grp = [r for r in records if r[0] == group]
            axes[0].plot([r[2] for r in grp], [r[7] for r in grp], "o-",
                         label=f"G{group}")
        if records:
            gv = [r[2] for r in records]
            dv = [r[7] for r in records]
            axes[0].plot(gv, dv, "o", alpha=0.5, label="全部")
        axes[0].axhline(0, color="gray", ls="--", lw=0.8)
        axes[0].set_xlabel("seen 训练 Gini（越高越长尾）")
        axes[0].set_ylabel("ΔOA = bal开 − bal关")
        axes[0].set_title("平衡采样收益 vs 训练池长尾程度")
        axes[0].legend()

        for group in [1, 5]:
            grp = [r for r in records if r[0] == group]
            axes[1].plot([r[4] for r in grp], [r[6] for r in grp], "o-", label=f"G{group} bal关")
            axes[1].plot([r[4] for r in grp], [r[7] + r[6] for r in grp], "s--",
                         label=f"G{group} bal开", alpha=0.8)
        axes[1].set_xlabel("训练像素总数（cap 越小越少）")
        axes[1].set_ylabel("OA %")
        axes[1].set_title("绝对 OA（同臂内总量下降是预期代价）")
        axes[1].legend()
        fig.tight_layout()
        fig_path = os.path.join(OUT_DIR, "delta_vs_gini.png")
        fig.savefig(fig_path, dpi=150)
        print(f"已生成 {fig_path}")
    except Exception as e:  # pragma: no cover
        print(f"画图失败（不影响表格）：{e}")


if __name__ == "__main__":
    main()
