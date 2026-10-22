#!/bin/bash
# Indian Pines 类平衡采样（sampler_power=1.0，与 LongKou A/B 的 bal 臂同设置）· seed 42
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
PY=/root/miniconda3/bin/python
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42)
status () { echo "$(date '+%F %T') | IP BAL | $*" >> runs/STATUS.txt; }
CODES="main.py datasets.py model_sz.py pipeline.py utils.py eval.py"
mkdir -p runs

status "启动"
for SEED in "${SEEDS[@]}"; do
  OUT="runs/ip_bal_seed${SEED}"; mkdir -p "$OUT"
  cp config/Indian.json "$OUT/config_used.json"
  md5sum $CODES > "$OUT/code_md5.txt"
  { echo "arm=bal"; echo "sampler_power=1.0"; echo "seed=$SEED"
    echo "started=$(date '+%F %T')"; } > "$OUT/meta.txt"
  for FOLD in "G1|Corn-notill,Hay-windrowed,Woods" \
              "G2|Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill" \
              "G3|Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill" \
              "G4|Corn,Alfalfa,Grass-pasture" \
              "G5|Wheat,Oats,Soybean-notill" \
              "G6|Soybean-clean"; do
    G="${FOLD%%|*}"; UN="${FOLD##*|}"
    [ -f "$OUT/${G}.done" ] && { echo "[skip] $G"; continue; }
    echo "[start] $G ip-bal seed=$SEED $(date '+%F %T')"; status "$G 开始"
    $PY -u main.py --dataset Indian --seed "$SEED" --unseen "$UN" \
        --use_balanced_sampler 1 > "$OUT/${G}.stdout.log" 2>&1
    RC=$?
    echo "$RC" > "$OUT/${G}.exit"
    for f in log_Indian_seed${SEED}_g*.txt; do [ -f "$f" ] && mv "$f" "$OUT/"; done
    if [ $RC -eq 0 ]; then echo "$(date '+%F %T')" > "$OUT/${G}.done"; status "$G 完成"
    else status "$G 失败 exit=$RC"; fi
  done
done
status "全部结束"
