"""长尾消融批量调度：串行执行全部 (group, seed, level, bal) 组合，支持断点续跑。

用法:
    python run_longtail_batch.py                # 默认 G1,G5 x 4 水平 x 2 臂 x 3 seed
    python run_longtail_batch.py --groups 1 5 --seeds 42,123,456

规则:
    - 已有 TSV 记录的任务自动跳过（断点续跑）。
    - 每个任务失败重试 1 次，仍失败则记录到 errors 列表并继续。
    - 串行执行：笔记本 GPU 并行训练会严重降频，比串行更慢。
"""
import argparse
import os
import subprocess
import sys
import time

from run_honest_eval import GROUPS
from make_longtail_caps import LEVEL_DEFS


PYTHON = sys.executable
OUT_DIR = os.path.join("result_IP", "longtail_ablation")
TSV = os.path.join(OUT_DIR, "results.tsv")


def done_tasks():
    done = set()
    if not os.path.exists(TSV):
        return done
    with open(TSV, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                done.add((int(parts[0]), int(parts[1]), parts[2], int(parts[3])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--seeds", default="42,123,456")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    levels = ["natural"] + list(LEVEL_DEFS)
    os.makedirs(OUT_DIR, exist_ok=True)
    done = done_tasks()

    tasks = []
    for group in args.groups:
        unseen = GROUPS["Indian"][group - 1]
        for level in levels:
            for bal in (1, 0):
                for seed in seeds:
                    # 自然水平 + 平衡开 = 已知的 D 配置（结果已在既有记录中），不重复跑
                    if level == "natural" and bal == 1:
                        continue
                    tasks.append((group, seed, level, bal))

    todo = [t for t in tasks if t not in done]
    print(f"总任务 {len(tasks)}，已完成 {len(tasks) - len(todo)}，待跑 {len(todo)}")

    errors = []
    t_start = time.time()
    for i, (group, seed, level, bal) in enumerate(todo, 1):
        t0 = time.time()
        cmd = [PYTHON, "run_longtail_ablation.py",
               "--group", str(group), "--seed", str(seed),
               "--level", level, "--bal", str(bal)]
        ok = False
        for attempt in (1, 2):
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            if proc.returncode == 0:
                ok = True
                break
            if attempt == 1:
                print(f"  [重试] G{group} seed={seed} {level} bal={bal} 第1次失败")
        status = "OK" if ok else "FAIL"
        if not ok:
            errors.append((group, seed, level, bal, proc.stderr[-1500:]))
            print(f"[{i}/{len(todo)}] {status} G{group} seed={seed} {level} bal={bal} "
                  f"用时{time.time() - t0:.0f}s")
        else:
            print(f"[{i}/{len(todo)}] {status} G{group} seed={seed} {level} bal={bal} "
                  f"用时{time.time() - t0:.0f}s")
        # 显存释放 + 降温缓冲
        time.sleep(2)

    print(f"\n=== 批次结束，总用时 {(time.time() - t_start) / 60:.1f} min ===")
    if errors:
        print(f"失败 {len(errors)} 个任务：")
        for e in errors:
            print("  ", e[:2], e[2:4], e[4][-300:].replace("\n", " | "))


if __name__ == "__main__":
    main()
