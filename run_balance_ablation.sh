#!/bin/bash
# 类平衡采样消融（对照 = 论文版基线：四项全关）
#   只改一个开关：--use_balanced_sampler 1
#   用法: bash run_balance_ablation.sh            # 默认 seeds 42 123 456，跑 IP 6 折
#         bash run_balance_ablation.sh 42         # 指定 seed
#         bash run_balance_ablation.sh 42 --whhl  # 追加 WHHL 3 折
#
# 为什么不能用 run_paperfolds_seeds.sh 直接跑：
#   main.py 的 loger 文件名 = log_<dataset>_seed<seed>_g<md5(unseen)[:4]>.txt，
#   **不含开关状态** → 直接跑会覆盖基线那批 log。本脚本靠归档 + 即时挪走解决。
#
# 断点续跑：读 runs/driver_bal.log 里 exit=0 的折自动跳过，原样重跑即可。
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
PY=/root/miniconda3/bin/python
mkdir -p runs baseline_logs bal_logs
DRIVER=runs/driver_bal.log
touch "$DRIVER"

# 解析参数
SEEDS=()
WITH_WHHL=0
for a in "$@"; do
  case "$a" in
    --whhl) WITH_WHHL=1 ;;
    *) SEEDS+=("$a") ;;
  esac
done
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 123 456)

echo "SEEDS = ${SEEDS[*]}   WHHL = $WITH_WHHL"

# 1) 首次运行：把基线那批 log 归档，避免同名混淆
if [ ! -f baseline_logs/.archived ]; then
  n=$(ls log_*.txt 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then
    mv log_*.txt baseline_logs/ 2>/dev/null
    echo "已归档 $n 个基线日志 → baseline_logs/"
  fi
  touch baseline_logs/.archived
else
  # 万一工作目录又冒出 log_*.txt，也挪走
  mv log_*.txt baseline_logs/ 2>/dev/null
fi

is_done () { grep -q "^\[done \] $1 .*exit=0" "$DRIVER"; }

run_fold () {  # $1=dataset $2=group $3=unseen $4=seed
  local tag="$1_$2_seed$4_bal"
  if is_done "$tag"; then echo "[skip ] $tag  （已完成）"; return 0; fi
  echo "[start] $tag  $(date '+%F %H:%M:%S')"
  $PY -u main.py --dataset "$1" --seed "$4" --unseen "$3" \
      --use_balanced_sampler 1 > "runs/${tag}.log" 2>&1
  local rc=$?
  # 把本次产生的 loger 文件挪进 bal_logs/，工作区保持干净、绝不与基线混淆
  mv log_*.txt bal_logs/ 2>/dev/null
  echo "[done ] $tag  $(date '+%F %H:%M:%S')  exit=$rc"
}

for SEED in "${SEEDS[@]}"; do
  echo
  echo "########## SEED $SEED ##########"
  echo "----- Indian Pines (6 折, use_balanced_sampler=1) -----"
  run_fold Indian G1 "Corn-notill,Hay-windrowed,Woods"                          $SEED
  run_fold Indian G2 "Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill" $SEED
  run_fold Indian G3 "Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill"      $SEED
  run_fold Indian G4 "Corn,Alfalfa,Grass-pasture"                               $SEED
  run_fold Indian G5 "Wheat,Oats,Soybean-notill"                                $SEED
  run_fold Indian G6 "Soybean-clean"                                            $SEED

  if [ "$WITH_WHHL" -eq 1 ]; then
    echo "----- WHHL / LongKou (3 折, use_balanced_sampler=1) -----"
    # 注意：LongKou 的 lambda_clip 需按既有口径临时置 1.0（见 run_paperfolds_seeds.sh）
    sed -i 's/"lambda_clip": 0.0/"lambda_clip": 1.0/' config/LongKou.json
    run_fold LongKou G1 "Corn,Cotton,Water"                                      $SEED
    run_fold LongKou G2 "Sesame,Broad-leaf soybean,Rice"                         $SEED
    run_fold LongKou G3 "Narrow-leaf soybean,Mixed weed,Roads and houses"        $SEED
    sed -i 's/"lambda_clip": 1.0/"lambda_clip": 0.0/' config/LongKou.json
  fi

  echo "[seed $SEED 全部完成]  $(date '+%F %H:%M:%S')"
done

echo "BAL_DONE  $(date '+%F %H:%M:%S')" > runs/.done_bal
echo
echo "########## ALL DONE ##########"
echo "结果日志在 bal_logs/ （同名基线日志在 baseline_logs/）"
