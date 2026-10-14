#!/bin/bash
# IP 消融对照：base（四项全关 = 论文版基线） vs bal（+ 类平衡采样）
#   用法:  bash run_ablation_ip.sh base [seeds...]
#          bash run_ablation_ip.sh bal  [seeds...]
#   seeds 省略时默认 42 123 456
#
# 设计要点：
#   1) 两个 arm 用同一份代码、同一份 config，**唯一差别**是 bal 多传 --use_balanced_sampler 1
#      （CLI 覆盖，不改 config 文件 → 无残留、无需还原）
#   2) main.py 的 loger 文件名不含开关状态 → 必须隔离，否则 A/B 互相覆盖。
#      本脚本把每折产生的 log_*.txt 即时挪进 logs_<arm>/
#   3) 断点续跑：读 runs/driver_<arm>.log 里 exit=0 的折自动跳过，原样重跑即可
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
PY=/root/miniconda3/bin/python

ARM="${1:-}"; shift 2>/dev/null || true
case "$ARM" in
  base) EXTRA=() ;;
  bal)  EXTRA=(--use_balanced_sampler 1) ;;
  *) echo "用法: bash run_ablation_ip.sh {base|bal} [seeds...]"; exit 1 ;;
esac

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 123 456)

LOGDIR="logs_${ARM}"
mkdir -p runs "$LOGDIR"
DRIVER="runs/driver_${ARM}.log"
touch "$DRIVER"
mv log_*.txt "$LOGDIR"/ 2>/dev/null    # 清走遗留，避免挪错

echo "ARM=$ARM   SEEDS=${SEEDS[*]}   EXTRA=${EXTRA[*]:-none}"
echo "已有进度: $(grep -c '^\[done ' "$DRIVER") 折"

is_done () { grep -q "^\[done \] $1 .*exit=0" "$DRIVER"; }

run_fold () {  # $1=group $2=unseen $3=seed
  local tag="Indian_$1_seed$3_${ARM}"
  if is_done "$tag"; then echo "[skip ] $tag  （已完成）"; return 0; fi
  echo "[start] $tag  $(date '+%F %H:%M:%S')"
  $PY -u main.py --dataset Indian --seed "$3" --unseen "$2" \
      ${EXTRA[@]+"${EXTRA[@]}"} > "runs/${tag}.log" 2>&1
  local rc=$?
  mv log_*.txt "$LOGDIR"/ 2>/dev/null
  echo "[done ] $tag  $(date '+%F %H:%M:%S')  exit=$rc"
}

for SEED in "${SEEDS[@]}"; do
  echo
  echo "########## ARM=$ARM  SEED $SEED ##########"
  run_fold G1 "Corn-notill,Hay-windrowed,Woods"                          $SEED
  run_fold G2 "Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill" $SEED
  run_fold G3 "Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill"      $SEED
  run_fold G4 "Corn,Alfalfa,Grass-pasture"                               $SEED
  run_fold G5 "Wheat,Oats,Soybean-notill"                                $SEED
  run_fold G6 "Soybean-clean"                                            $SEED
  echo "[ARM=$ARM · seed $SEED 完成]  $(date '+%F %H:%M:%S')"
done

echo "ARM_${ARM}_DONE  $(date '+%F %H:%M:%S')" > "runs/.done_${ARM}"
echo
echo "########## ARM=$ARM ALL DONE ##########"
echo "loger 明细: $LOGDIR/     stdout: runs/"
