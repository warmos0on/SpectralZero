#!/bin/bash
# LongKou 温和平衡采样（sampler_power=0.5）· seed 42
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
PY=/root/miniconda3/bin/python
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42)

LK=config/LongKou.json
cp "$LK" /tmp/lk_config.bak
restore_lk () { cp /tmp/lk_config.bak "$LK"; }
trap restore_lk EXIT
status () { echo "$(date '+%F %T') | LongKou MILD | $*" >> runs/STATUS.txt; }
CODES="main.py datasets.py model_sz.py pipeline.py utils.py eval.py"
mkdir -p runs

sed -i 's/"lambda_clip": 0.0/"lambda_clip": 1.0/' "$LK"
status "启动 sampler_power=0.5"

for SEED in "${SEEDS[@]}"; do
  OUT="runs/lk_mild_seed${SEED}"; mkdir -p "$OUT"
  cp "$LK" "$OUT/config_used.json"
  md5sum $CODES > "$OUT/code_md5.txt"
  { echo "arm=mild"; echo "sampler_power=0.5"; echo "seed=$SEED"
    echo "started=$(date '+%F %T')"; } > "$OUT/meta.txt"
  for FOLD in "G1|Corn,Cotton,Water" "G2|Sesame,Broad-leaf soybean,Rice" "G3|Narrow-leaf soybean,Mixed weed,Roads and houses"; do
    G="${FOLD%%|*}"; UN="${FOLD##*|}"
    [ -f "$OUT/${G}.done" ] && { echo "[skip] $G"; continue; }
    echo "[start] $G mild seed=$SEED $(date '+%F %T')"; status "$G 开始"
    $PY -u main.py --dataset LongKou --seed "$SEED" --unseen "$UN" \
        --use_balanced_sampler 1 --sampler_power 0.5 > "$OUT/${G}.stdout.log" 2>&1
    RC=$?
    echo "$RC" > "$OUT/${G}.exit"
    for f in log_LongKou_seed${SEED}_g*.txt; do [ -f "$f" ] && mv "$f" "$OUT/"; done
    if [ $RC -eq 0 ]; then echo "$(date '+%F %T')" > "$OUT/${G}.done"; status "$G 完成"
    else status "$G 失败 exit=$RC"; fi
  done
done
restore_lk
status "mild 全部结束"
