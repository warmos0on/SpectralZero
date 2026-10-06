#!/bin/bash
# 论文轮换折复现 · 多 seed 批跑（可断点续跑版）
#   用法: bash run_paperfolds_seeds.sh 123 456
#
# 断点续跑：从 runs/driver_seeds.log 里读已 `[done ] <tag> ... exit=0` 的折，自动跳过。
# 所以中途停了之后，明天原样再跑一次这条命令即可，不会重跑已完成的折。
#
# 与 run_paperfolds.sh（第一版）的差别：
#   1) 多 seed
#   2) python -u —— 无缓冲，runs/*.log 运行中即可见
#   3) 失败不中断（记 exit code 继续下一折）
#   4) trap 还原 LongKou 配置
#   5) 断点续跑
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
PY=/root/miniconda3/bin/python
mkdir -p runs
DRIVER=runs/driver_seeds.log
touch "$DRIVER"

SEEDS="${*:-123}"
echo "SEEDS = $SEEDS"
echo "已有进度: $(grep -c '^\[done ' "$DRIVER") 折已完成"

LK_JSON=config/LongKou.json
restore_lk () { sed -i 's/"lambda_clip": 1.0/"lambda_clip": 0.0/' "$LK_JSON"; }
trap restore_lk EXIT

# 已成功完成的折（tag 列表）
is_done () { grep -q "^\[done \] $1 .*exit=0" "$DRIVER"; }

run_fold () {  # $1=dataset $2=组名 $3=unseen列表 $4=seed
  local tag="$1_$2_seed$4"
  if is_done "$tag"; then
    echo "[skip ] $tag  （已完成，跳过）"
    return 0
  fi
  echo "[start] $tag  $(date '+%F %H:%M:%S')"
  $PY -u main.py --dataset "$1" --seed "$4" --unseen "$3" > "runs/${tag}.log" 2>&1
  local rc=$?
  echo "[done ] $tag  $(date '+%F %H:%M:%S')  exit=$rc"
}

for SEED in $SEEDS; do
  echo
  echo "########## SEED $SEED ##########"

  echo "----- Indian Pines (6 folds, lambda_clip=0.6) -----"
  run_fold Indian G1 "Corn-notill,Hay-windrowed,Woods"                                $SEED
  run_fold Indian G2 "Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill"       $SEED
  run_fold Indian G3 "Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill"            $SEED
  run_fold Indian G4 "Corn,Alfalfa,Grass-pasture"                                     $SEED
  run_fold Indian G5 "Wheat,Oats,Soybean-notill"                                      $SEED
  run_fold Indian G6 "Soybean-clean"                                                  $SEED

  echo "----- WHHL / LongKou (3 folds, lambda_clip=1.0) -----"
  sed -i 's/"lambda_clip": 0.0/"lambda_clip": 1.0/' "$LK_JSON"
  run_fold LongKou G1 "Corn,Cotton,Water"                                             $SEED
  run_fold LongKou G2 "Sesame,Broad-leaf soybean,Rice"                                $SEED
  run_fold LongKou G3 "Narrow-leaf soybean,Mixed weed,Roads and houses"               $SEED
  restore_lk

  echo "[seed $SEED 全部完成]  $(date '+%F %H:%M:%S')"
done

echo "ALL_SEEDS_DONE  $(date '+%F %H:%M:%S')" > runs/.done_seeds
echo
echo "########## ALL DONE ##########"
