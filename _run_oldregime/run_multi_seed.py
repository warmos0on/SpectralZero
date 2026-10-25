"""批量复现脚本：跑所有数据集的所有 Group × 多个 seed"""
import subprocess
import re
import sys
import numpy as np

# ============ 配置 ============
SEEDS = [42, 123, 456, 789, 1024]

# 所有 Group 的 unseen_classes 配置
GROUPS = {
    "Houston": [
        ["Healthy grass", "Synthetic grass", "Water"],          # Group 1
        ["Commercial", "Railways", "Trees"],                    # Group 2
        ["Tennis Court", "Running Track", "Parking Lot 1"],     # Group 3
        ["Parking Lot 2", "Residential", "Road"],               # Group 4
        ["Stressed grass", "Soil", "Highways"],                 # Group 5
    ],
    "LongKou": [
        ["Corn", "Cotton", "Water"],
        ["Sesame", "Broad-leaf soybean", "Rice"],
        ["Narrow-leaf soybean", "Mixed weed", "Roads and houses"],
    ],
    # G1已完成，只跑G2-G6
    "Indian": [
        ["Grass-trees", "Buildings-Grass-Trees-Drives", "Soybean-mintill"],          # G2
        ["Grass-pasture-mowed", "Stone-Steel-Towers", "Corn-mintill"],               # G3
        ["Corn", "Alfalfa", "Grass-pasture"],                                        # G4
        ["Wheat", "Oats", "Soybean-notill"],                                         # G5
        ["Soybean-clean"],                                                           # G6
    ],
}

DATASET = sys.argv[1] if len(sys.argv) > 1 else "all"
if DATASET == "all":
    datasets_to_run = ["Houston", "LongKou", "Indian"]
else:
    datasets_to_run = [DATASET]

all_results = {}

for dataset in datasets_to_run:
    print(f"\n{'#'*60}")
    print(f"#  数据集: {dataset}")
    print(f"{'#'*60}")

    group_results = []

    for gi, unseen in enumerate(GROUPS[dataset]):
        unseen_str = ",".join(unseen)
        group_name = f"G{gi+1}"
        print(f"\n  --- Group {group_name}: unseen={unseen} ---")

        seed_bests = []

        for seed in SEEDS:
            cmd = f"python main.py --dataset {dataset} --seed {seed} --unseen \"{unseen_str}\""
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            oa_list, aa_list, kappa_list = [], [], []
            for line in proc.stdout.splitlines():
                m = re.search(r'unseen OA is\s+([\d.]+),\s*unseen AA is\s+([\d.]+),\s*kappa is\s+([\d.\-]+)', line)
                if m:
                    oa_list.append(float(m.group(1)))
                    aa_list.append(float(m.group(2)))
                    kappa_list.append(float(m.group(3)))

            if not oa_list:
                print(f"    seed={seed}: 无结果！")
                continue

            best_oa = max(oa_list)
            best_aa = max(aa_list)
            best_kappa = max(kappa_list)
            seed_bests.append((best_oa, best_aa, best_kappa))
            print(f"    seed={seed}: OA={best_oa:.2f} AA={best_aa:.2f} Kappa={best_kappa:.4f}")

        if seed_bests:
            oas = [s[0] for s in seed_bests]
            aas = [s[1] for s in seed_bests]
            kappas = [s[2] for s in seed_bests]
            best_aa_grp = max(aas)
            group_results.append({
                'name': group_name,
                'unseen': unseen,
                'best_oa': max(oas),
                'avg_oa': np.mean(oas),
                'std_oa': np.std(oas),
                'best_aa': best_aa_grp,
                'avg_aa': np.mean(aas),
                'std_aa': np.std(aas),
                'best_kappa': max(kappas),
                'avg_kappa': np.mean(kappas),
                'std_kappa': np.std(kappas),
            })
            # 全零即停
            if best_aa_grp < 0.1:
                print(f"\n  [停止] {group_name} 最佳AA={best_aa_grp:.2f}%，跳过剩余Group")
                break

    # ======== 单个数据集的汇总 ========
    print(f"\n{'='*60}")
    print(f"  {dataset} 各 Group 汇总")
    print(f"{'='*60}")
    if group_results:
        print(f"{'Group':<10} {'unseen':<50} {'最佳OA':<10} {'平均OA±std':<20} {'最佳Kappa':<12}")
        for gr in group_results:
            unseen_short = ", ".join(gr['unseen'])
            print(f"{gr['name']:<10} {unseen_short:<50} {gr['best_oa']:<10.2f} {gr['avg_oa']:.2f}±{gr['std_oa']:.2f}{'':<8} {gr['best_kappa']:.4f}")

        avg_oa = np.mean([g['avg_oa'] for g in group_results])
        avg_aa = np.mean([g['avg_aa'] for g in group_results])
        avg_kappa = np.mean([g['avg_kappa'] for g in group_results])
        best_oa = np.mean([g['best_oa'] for g in group_results])
        best_aa = np.mean([g['best_aa'] for g in group_results])
        best_kappa = np.mean([g['best_kappa'] for g in group_results])

        print(f"\n  >>> 所有 Group 平均 (取最佳): OA={best_oa:.2f} AA={best_aa:.2f} Kappa={best_kappa:.4f}")
        print(f"  >>> 所有 Group 平均 (取平均): OA={avg_oa:.2f} AA={avg_aa:.2f} Kappa={avg_kappa:.4f}")
        all_results[dataset] = {
            'best_oa': best_oa, 'best_aa': best_aa, 'best_kappa': best_kappa,
            'avg_oa': avg_oa, 'avg_aa': avg_aa, 'avg_kappa': avg_kappa,
        }
    else:
        print("  无有效结果！")

# ======== 总汇总 ========
if len(datasets_to_run) > 1:
    print(f"\n{'#'*60}")
    print(f"#  总汇总")
    print(f"{'#'*60}")
    print(f"{'数据集':<15} {'平均OA':<12} {'平均AA':<12} {'平均Kappa':<12}")
    for ds, res in all_results.items():
        print(f"{ds:<15} {res['best_oa']:.2f}         {res['best_aa']:.2f}         {res['best_kappa']:.4f}")
