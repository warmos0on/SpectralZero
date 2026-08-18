"""Generate 2 condensed LaTeX tables from Indian Pines results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "c:/Users/39448/Desktop/SpectralZero/tables"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Data
# ============================================================

# Per-class: (class, group, paper_aa, our_aa, best_seed)
per_class = [
    ("Corn-notill",                 "G1", 99.84, 79.82, 42),
    ("Hay-windrowed",               "G1", 94.39, 90.07, 42),
    ("Woods",                       "G1", 67.99, 48.62, 123),
    ("Grass-trees",                 "G2", 99.67, 91.15, 789),
    ("Buildings-Grass-Trees-Drives","G2", 61.60, 46.63, 123),
    ("Soybean-mintill",             "G2", 97.45, 91.55, 1024),
    ("Grass-pasture-mowed",         "G3", 100.00, 96.43, 1024),
    ("Stone-Steel-Towers",          "G3", 0.00, 90.75, 123),
    ("Corn-mintill",                "G3", 97.13, 87.88, 123),
    ("Corn",                        "G4", 100.00, 97.97, 123),
    ("Alfalfa",                     "G4", 0.00, 4.35, 42),
    ("Grass-pasture",               "G4", 93.58, 95.44, 789),
    ("Wheat",                       "G5", 99.66, 96.98, 42),
    ("Oats",                        "G5", 100.00, 98.00, 123),
    ("Soybean-notill",              "G5", 92.08, 92.57, 42),
    ("Soybean-clean",               "G6", 99.85, 100.00, 42),
]

# Group summary
groups = [
    ("G1", "Corn-notill,\nHay-windrowed,\nWoods",
     87.41, 76.68, 42, 0.6651,
     "42:76.68 123:73.52 456:72.88 789:72.49 1024:68.61"),
    ("G2", "Grass-trees,\nBuildings-Grass-Trees-Drives,\nSoybean-mintill",
     86.24, 85.68, 123, 0.7603,
     "42:73.46 123:85.68 456:66.29 789:79.57 1024:77.21"),
    ("G3", "Grass-pasture-mowed,\nStone-Steel-Towers,\nCorn-mintill",
     65.71, 96.58, 123, 0.8851,
     "42:93.00 123:96.58 456:89.65 789:87.62 1024:91.59"),
    ("G4", "Corn,\nAlfalfa,\nGrass-pasture",
     64.53, 67.00, 789, 0.8042,
     "42:66.44 123:65.22 456:65.79 789:67.00 1024:65.17"),
    ("G5", "Wheat,\nOats,\nSoybean-notill",
     97.25, 98.42, 42, 0.8855,
     "42:98.42 123:95.45 456:97.43 789:95.42 1024:92.53"),
    ("G6", "Soybean-clean",
     99.85, 100.00, 42, None,
     "All seeds: 100.00"),
]

# Overall metrics
overall_metrics = [
    ("Overall Accuracy (OA)", "97.34%", "--"),
    ("Average Accuracy (AA)", "81.45%", "81.76%"),
    ("Kappa ($\\kappa$)", "0.8778", "--"),
]

# ============================================================
# LaTeX .tex files
# ============================================================

def table1_tex():
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Indian Pines: Group-Level Zero-Shot Classification Results}",
        r"\label{tab:ip_group}",
        r"\begin{tabular}{c p{3.4cm} c c c c}",
        r"\toprule",
        r"Group & Unseen Classes & Paper AA\% & Our Best AA\% & Best Seed & $\kappa$ \\",
        r"\midrule",
    ]
    for g, unseen, paper, ours, seed, kappa, _ in groups:
        k = f"{kappa:.4f}" if kappa else "--"
        lines.append(
            f"  {g} & {unseen} & {paper:.2f}\% & {ours:.2f}\% & {seed} & {k} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table2_tex():
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Indian Pines: Per-Class Accuracy (AA\%) -- Paper vs. Ours}",
        r"\label{tab:ip_perclass}",
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        r"Class & Group & Paper AA\% & Our AA\% & $\Delta$ & Best Seed \\",
        r"\midrule",
    ]
    for name, grp, paper, ours, seed in per_class:
        diff = ours - paper
        sign = "+" if diff >= 0 else ""
        lines.append(
            f"  {name.replace('_', r'\_')} & {grp} & {paper:.2f}\% & {ours:.2f}\% & ${sign}{diff:.2f}$ & {seed} \\\\"
        )
    paper_mean = np.mean([p for _, _, p, _, _ in per_class])
    our_mean  = np.mean([o for _, _, _, o, _ in per_class])
    diff_mean = our_mean - paper_mean
    sign = "+" if diff_mean >= 0 else ""
    lines += [
        r"\midrule",
        fr"  \textbf{{Mean (16 classes)}} & -- & \textbf{{{paper_mean:.2f}\%}} & \textbf{{{our_mean:.2f}\%}} & $\mathbf{{{sign}{diff_mean:.2f}}}$ & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


for fname, fn in [("table_group_summary.tex", table1_tex),
                   ("table_per_class.tex", table2_tex)]:
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(fn())
    print(f"  Saved .tex: {path}")

# ============================================================
# Render tables as images
# ============================================================

plt.rcParams.update({"font.family": "serif", "font.size": 8})

def render_table(headers, rows, title, filename, col_widths=None):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    fig, ax = plt.subplots(figsize=(n_cols * 2.1, n_rows * 0.40 + 0.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        edges="horizontal",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.55)

    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    for i in range(1, n_rows):
        for j in range(n_cols):
            cell = tbl[i, j]
            # highlight the last row (mean)
            if i == n_rows - 1:
                cell.set_facecolor("#d4e6f1")
                cell.set_text_props(fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#ecf0f1")
            else:
                cell.set_facecolor("white")

    ax.set_title(title, fontweight="bold", fontsize=10, pad=12)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Rendered: {path}")


# Table 1: Group summary + multi-seed
h1 = ["Group", "Unseen Classes", "Paper\nAA%", "Our Best\nAA%", "Best\nSeed", "Kappa"]
r1 = []
for g, unseen, paper, ours, seed, kappa, seeds_info in groups:
    k = f"{kappa:.4f}" if kappa else "--"
    u = unseen.replace("\n", " ")
    r1.append([g, u, f"{paper:.2f}", f"{ours:.2f}", str(seed), k])
render_table(h1, r1,
    "Table 1: Indian Pines — Group-Level Zero-Shot Results", "table_group_summary.png")

# Table 2: Per-class breakdown
h2 = ["Class", "Group", "Paper AA%", "Our AA%", "Δ", "Best Seed"]
r2 = []
for name, grp, paper, ours, seed in per_class:
    diff = ours - paper
    r2.append([name.replace("_", " "), grp, f"{paper:.2f}", f"{ours:.2f}", f"{diff:+.2f}", str(seed)])
p_mean = np.mean([p for _, _, p, _, _ in per_class])
o_mean = np.mean([o for _, _, _, o, _ in per_class])
d_mean = o_mean - p_mean
r2.append(["Mean (16 classes)", "--", f"{p_mean:.2f}", f"{o_mean:.2f}", f"{d_mean:+.2f}", "--"])
render_table(h2, r2,
    "Table 2: Indian Pines — Per-Class Accuracy Comparison", "table_per_class.png")

# Extra: Overall metrics summary as small table
h3 = ["Metric", "Paper", "Ours"]
r3 = []
for m in overall_metrics:
    r3.append(list(m))
for g, unseen, paper, ours, _, kappa, _ in groups:
    k = f"{kappa:.4f}" if kappa else "--"
    r3.append([f"{g} AA", f"{paper:.2f}%", f"{ours:.2f}%"])
render_table(h3, r3,
    "Table 3: Indian Pines — Overall Metrics", "table_overall.png")

print("\nDone! 3 tables generated (2 main + 1 supplemental).")
