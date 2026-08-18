"""诚实评测协议：固定最后一个 checkpoint、多 seed 统计 mean±std、带 per-class AA。

用法:
    python run_honest_eval.py --dataset Indian --seeds 42,123,456 --epochs 20 --checkpoints 5
    python run_honest_eval.py --dataset Indian --seeds 42,123,456 --epochs 20 --checkpoints 5 --groups 3
"""
import subprocess
import re
import sys
import time
import numpy as np

PYTHON = sys.executable

GROUPS = {
    "Indian": [
        ["Corn-notill", "Hay-windrowed", "Woods"],                          # G1
        ["Grass-trees", "Buildings-Grass-Trees-Drives", "Soybean-mintill"], # G2
        ["Grass-pasture-mowed", "Stone-Steel-Towers", "Corn-mintill"],      # G3
        ["Corn", "Alfalfa", "Grass-pasture"],                               # G4
        ["Wheat", "Oats", "Soybean-notill"],                                # G5
        ["Soybean-clean"],                                                  # G6 单类，仅参考
    ],
    "LongKou": [
        ["Corn", "Cotton", "Water"],                                        # G1
        ["Sesame", "Broad-leaf soybean", "Rice"],                           # G2
        ["Narrow-leaf soybean", "Mixed weed", "Roads and houses"],          # G3
    ],
}

OA_RE = re.compile(r"unseen OA is\s+([\d.]+),\s*unseen AA is\s+([\d.]+),\s*kappa is\s+([\d.\-]+|nan)")
PER_AA_RE = re.compile(r"unseen per AA\s+\[([^\]]*)\]")


def run_one_group(dataset, unseen, seed, epochs, checkpoints, ablation=None):
    unseen_str = ",".join(unseen)
    cmd = (
        f"\"{PYTHON}\" main.py --dataset {dataset} --seed {seed} "
        f"--unseen \"{unseen_str}\" --epochs {epochs} --checkpoints {checkpoints}"
    )
    for item in (ablation or []):
        cmd += f" --{item}"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    stdout = proc.stdout
    oa_matches = OA_RE.findall(stdout)
    if not oa_matches:
        tail = stdout[-2000:] if stdout else ""
        err_tail = proc.stderr[-2000:] if proc.stderr else ""
        raise RuntimeError(f"seed={seed} 无有效评测输出\n[stdout]{tail}\n[stderr]{err_tail}")
    # 固定取最后一个 checkpoint
    last_oa, last_aa, last_kappa = oa_matches[-1]
    per_aa = None
    all_per = PER_AA_RE.findall(stdout)
    if all_per:
        per_aa = np.array([float(x) for x in all_per[-1].split()])
    kappa_val = float(last_kappa) if last_kappa != "nan" else float("nan")
    return float(last_oa), float(last_aa), kappa_val, per_aa


def parse_args(argv):
    opts = {}
    for i in range(0, len(argv), 2):
        if argv[i].startswith("--"):
            opts[argv[i][2:]] = argv[i + 1]
    return opts


def main():
    opts = parse_args(sys.argv[1:])
    dataset = opts.get("dataset", "Indian")
    seeds = [int(x) for x in opts.get("seeds", "42,123,456").split(",")]
    epochs = int(opts.get("epochs", "20"))
    checkpoints = int(opts.get("checkpoints", "5"))
    groups_sel = opts.get("groups", None)
    ablation = [x.strip() for x in opts.get("ablation", "").split(",") if x.strip()]

    groups = GROUPS[dataset]
    if groups_sel is not None:
        sel = {int(x) for x in groups_sel.split(",")}
        groups = [(i + 1, g) for i, g in enumerate(groups) if (i + 1) in sel]
    else:
        groups = [(i + 1, g) for i, g in enumerate(groups)]

    print(f"=== {dataset} 诚实评测（固定最后 checkpoint, seeds={seeds}, epochs={epochs}）===")
    results = {}
    total_t0 = time.time()
    for gi, unseen in groups:
        t0 = time.time()
        oas, aas, kappas, per_list = [], [], [], []
        for seed in seeds:
            oa, aa, kappa, per = run_one_group(dataset, unseen, seed, epochs, checkpoints, ablation)
            oas.append(oa)
            aas.append(aa)
            kappas.append(kappa)
            per_list.append(per)
            print(f"  G{gi} seed={seed}: OA={oa:.2f} AA={aa:.2f} Kappa={kappa:.4f}")
        per_mean = np.mean(per_list, axis=0) if per_list[0] is not None else None
        results[gi] = {
            "unseen": unseen,
            "oa": np.array(oas), "aa": np.array(aas), "kappa": np.array(kappas),
            "per_aa_mean": per_mean,
        }
        print(f"  G{gi} 用时 {time.time() - t0:.0f}s")
    total_time = time.time() - total_t0

    print("\n" + "=" * 100)
    print(f"{'Group':<6} {'Unseen':<52} {'Last-cp OA (mean±std)':<24} {'AA':<18} {'Kappa':<12}")
    for gi, unseen in groups:
        r = results[gi]
        unseen_short = ", ".join(unseen)
        per_str = ""
        if r["per_aa_mean"] is not None:
            per_str = " | per-AA " + " ".join(f"{x:.1f}" for x in r["per_aa_mean"])
        print(f"G{gi:<5} {unseen_short:<52} "
              f"{r['oa'].mean():.2f}±{r['oa'].std():.2f}          "
              f"{r['aa'].mean():.2f}±{r['aa'].std():.2f}      "
              f"{r['kappa'].mean():.4f}±{r['kappa'].std():.4f}{per_str}")

    # 总平均（排除单类 G6）
    multi = {gi: r for gi, r in results.items() if len(r["unseen"]) > 1}
    if multi:
        mean_oa = np.mean([r["oa"].mean() for r in multi.values()])
        std_oa = np.mean([r["oa"].std() for r in multi.values()])
        mean_aa = np.mean([r["aa"].mean() for r in multi.values()])
        mean_kappa = np.mean([r["kappa"].mean() for r in multi.values()])
        print(f"\n>>> 多类组平均（固定最后 cp）: OA={mean_oa:.2f}（组内跨seed std均值 {std_oa:.2f}） "
              f"AA={mean_aa:.2f} Kappa={mean_kappa:.4f}")
    print(f"\n总耗时 {total_time:.0f}s")


if __name__ == "__main__":
    main()
