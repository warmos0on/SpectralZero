#!/bin/bash
# LongKou（WHHL）消融对照：base（论文版基线） vs bal（+ 类平衡采样）
#   用法:  bash run_ablation_lk.sh base [seeds...]
#          bash run_ablation_lk.sh bal  [seeds...]
#   seeds 省略时默认 42
#
# 设计要点（与 IP 那套 run_ablation_ip.sh 同构）：
#   1) 两个 arm 用同一份代码、同一份 config，**唯一差别**是 bal 多传 --use_balanced_sampler 1
#   2) LongKou 的 lambda_clip 需从 config 的 0.0 临时置 1.0（红线 3：0.0 会让投影头零梯度、必挂）
#      —— 用 trap 保证无论正常结束还是被杀都会还原
#   3) main.py 的 loger 文件名不含开关状态 → 按 arm 隔离到 logs_<arm>/
#   4) 断点续跑：读 runs/driver_lk_<arm>.log 里 exit=0 的折自动跳过
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
PY=/root/miniconda3/bin/python

ARM="${1:-}"; shift 2>/dev/null || true
case "$ARM" in
  base) EXTRA=() ;;
  bal)  EXTRA=(--use_balanced_sampler 1) ;;
  *) echo "用法: bash run_ablation_lk.sh {base|bal} [seeds...]"; exit 1 ;;
esac

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42)

LOGDIR="logs_lk_${ARM}"
mkdir -p runs "$LOGDIR"
DRIVER="runs/driver_lk_${ARM}.log"
touch "$DRIVER"
mv log_LongKou_*.txt "$LOGDIR"/ 2>/dev/null

LK_JSON=config/LongKou.json
restore_lk () { sed -i 's/"lambda_clip": 1.0/"lambda_clip": 0.0/' "$LK_JSON"; }
trap restore_lk EXIT

echo "ARM=$ARM   SEEDS=${SEEDS[*]}   EXTRA=${EXTRA[*]:-none}"
echo "已有进度: $(grep -c '^\[done ' "$DRIVER") 折"

is_done () { grep -q "^\[done \] $1 .*exit=0" "$DRIVER"; }

run_fold () {  # $1=组名 $2=unseen列表 $3=seed
  local tag="LongKou_$1_seed$3_${ARM}"
  if is_done "$tag"; then echo "[skip ] $tag  （已完成）"; return 0; fi
  echo "[start] $tag  $(date '+%F %H:%M:%S')"
  $PY -u main.py --dataset LongKou --seed "$3" --unseen "$2" \
      ${EXTRA[@]+"${EXTRA[@]}"} > "runs/${tag}.log" 2>&1
  local rc=$?
  mv log_LongKou_*.txt "$LOGDIR"/ 2>/dev/null
  echo "[done ] $tag  $(date '+%F %H:%M:%S')  exit=$rc"
}

for SEED in "${SEEDS[@]}"; do
  echo
  echo "########## ARM=$ARM  SEED $SEED ##########"
  # WHHL 的 lambda_clip：config 写 0.0（论文 λ_loss=1.0 的字面换算），实跑需 1.0（红线 3）
  sed -i 's/"lambda_clip": 0.0/"lambda_clip": 1.0/' "$LK_JSON"
  echo "  lambda_clip 已置 1.0：$(grep lambda_clip $LK_JSON | tr -d ' ')"
  run_fold G1 "Corn,Cotton,Water"                                  $SEED
  run_fold G2 "Sesame,Broad-leaf soybean,Rice"                     $SEED
  run_fold G3 "Narrow-leaf soybean,Mixed weed,Roads and houses"    $SEED
  restore_lk
  echo "[ARM=$ARM · seed $SEED 完成]  $(date '+%F %H:%M:%S')"
done

echo "ARM_LK_${ARM}_DONE  $(date '+%F %H:%M:%S')" > "runs/.done_lk_${ARM}"
echo
echo "########## ARM=$ARM ALL DONE ##########"
echo "loger 明细: $LOGDIR/     stdout: runs/"
