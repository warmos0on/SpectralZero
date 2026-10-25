"""长尾消融单任务驱动：跑一个 (group, seed, level, balanced on/off) 组合。

用法:
    python run_longtail_ablation.py --group 1 --seed 42 --level natural --bal 0
    python run_longtail_ablation.py --group 2 --seed 123 --level mild --bal 1

协议与 run_honest_eval.py 一致：固定最后一个 checkpoint 的 OA/AA/kappa。
基线配置 D = 关动态融合 + z-score + 类别平衡采样 + 无负样本。
bal=0 时把平衡采样关掉，保持其余不变，用于在同一 cap 水平下测平衡采样的净收益。
结果以 TSV 行追加到 result_IP/longtail_ablation/results.tsv。
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

from run_honest_eval import GROUPS
from make_longtail_caps import LEVEL_DEFS


PYTHON = sys.executable
OA_RE = re.compile(r"unseen OA is\s+([\d.]+),\s*unseen AA is\s+([\d.]+),\s*kappa is\s+([\d.\-]+|nan)")
PER_AA_RE = re.compile(r"unseen per AA\s+\[([^\]]*)\]")
CAP_DIR = os.path.join("result_IP", "longtail_caps")
OUT_DIR = os.path.join("result_IP", "longtail_ablation")


def run_one(group_idx, seed, level, bal_on):
    unseen = GROUPS["Indian"][group_idx - 1]
    unseen_str = ",".join(unseen)
    cmd = [
        PYTHON, "main.py", "--dataset", "Indian", "--seed", str(seed),
        "--unseen", unseen_str, "--epochs", "20", "--checkpoints", "5",
        "--use_zscore", "1", "--use_unseen_negatives", "0", "--dynamic_fusion", "0",
        "--use_balanced_sampler", "1" if bal_on else "0",
    ]
    if level != "natural":
        cap_path = os.path.join(CAP_DIR, f"level_{level}.json")
        if not os.path.exists(cap_path):
            raise FileNotFoundError(cap_path)
        cmd += ["--train_cap", cap_path]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    stdout = proc.stdout
    matches = OA_RE.findall(stdout)
    if not matches:
        tail = stdout[-3000:] if stdout else ""
        err = proc.stderr[-3000:] if proc.stderr else ""
        raise RuntimeError(
            f"G{group_idx} seed={seed} {level} bal={int(bal_on)} 无有效评测输出\n"
            f"[stdout]{tail}\n[stderr]{err}")
    oa, aa, kappa = matches[-1]
    per_aa = None
    all_per = PER_AA_RE.findall(stdout)
    if all_per:
        per_aa = np.array([float(x) for x in all_per[-1].split()])
    return float(oa), float(aa), float(kappa), per_aa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--level", choices=["natural"] + list(LEVEL_DEFS), required=True)
    ap.add_argument("--bal", type=int, choices=[0, 1], required=True)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    oa, aa, kappa, per_aa = run_one(args.group, args.seed, args.level, bool(args.bal))
    per_str = " ".join(f"{x:.1f}" for x in per_aa) if per_aa is not None else "NA"
    row = f"{args.group}\t{args.seed}\t{args.level}\t{int(args.bal)}\t{oa:.2f}\t{aa:.2f}\t{kappa:.4f}\t{per_str}\n"
    tsv = os.path.join(OUT_DIR, "results.tsv")
    with open(tsv, "a", encoding="utf-8") as f:
        f.write(row)
    print(
        f"OK G{args.group} seed={args.seed} level={args.level} bal={int(args.bal)} "
        f"OA={oa:.2f} AA={aa:.2f} Kappa={kappa:.4f} per-AA={per_str}",
        flush=True)


if __name__ == "__main__":
    main()
