#!/bin/bash
# 本地跑：旧配置（两个月前 archive）+ 只加类平衡采样
#   - 忠实复刻 archive 的 config/Indian.json（patch 11 / lr 2e-4 / lambda_clip 0.8 / spectral_ratio 0.2 /
#     batch 256 / dynamic_fusion ON），CLI 复刻 run_ablation.ps1 的 --epochs 20 --checkpoints 5
#   - 两处刻意的现代化（用户指定）：
#       ① radiation_augmentation=false —— 等价于旧代码里 radiation_noise 恒为 0 的空操作
#       ② **保留全部 5 个评测点**，不取末点
#   - 独立目录运行，不碰主仓库的 config
# 用法: bash run.sh [seeds...]      默认 42 123 456
set -u
cd "$(dirname "$0")" || exit 1
PY="C:/Users/ASUS/.conda/envs/SpectralZero/python.exe"
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 123 456)

mkdir -p runs logs
DRIVER=runs/driver_oldbase.log
touch "$DRIVER"
mv log_*.txt logs_base/ 2>/dev/null

EPOCHS=20; CKPT=5
is_done () { grep -q "^\[done \] $1 .*exit=0" "$DRIVER"; }

run_fold () {  # $1=group $2=unseen $3=seed
  local tag="Indian_$1_seed$3_oldbase"
  if is_done "$tag"; then echo "[skip ] $tag"; return 0; fi
  echo "[start] $tag  $(date '+%F %H:%M:%S')"
  "$PY" -u main.py --dataset Indian --seed "$3" --unseen "$2" \
      --epochs $EPOCHS --checkpoints $CKPT --use_balanced_sampler 0 \
      > "runs/${tag}.log" 2>&1
  local rc=$?
  mv log_*.txt logs_base/ 2>/dev/null
  echo "[done ] $tag  $(date '+%F %H:%M:%S')  exit=$rc"
}

for SEED in "${SEEDS[@]}"; do
  echo; echo "########## 旧配置 · 只加平衡采样 · SEED $SEED ##########"
  run_fold G1 "Corn-notill,Hay-windrowed,Woods"                          $SEED
  run_fold G2 "Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill" $SEED
  run_fold G3 "Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill"      $SEED
  run_fold G4 "Corn,Alfalfa,Grass-pasture"                               $SEED
  run_fold G5 "Wheat,Oats,Soybean-notill"                                $SEED
  run_fold G6 "Soybean-clean"                                            $SEED
  echo "[seed $SEED 完成]  $(date '+%F %H:%M:%S')"
done

echo "OLDREGIME_BASE_DONE  $(date '+%F %H:%M:%S')" > runs/.done_oldbase
echo; echo "########## 全部完成 ##########"
echo "loger 明细: logs_base/    stdout: runs/"
