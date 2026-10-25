#!/bin/bash
# LongKou 温和平衡采样（sampler_power=0.5）· seed 42
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
PY=/root/miniconda3/bin/python
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42)

LK=config/LongKou.json
# 幂等：先归位到文档值 0.0 再备份，避免上次被 kill 留下的 1.0 被当成原始值（同 run_lk_ab.sh）
$PY - "$LK" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d['lambda_clip'] = 0.0
json.dump(d, open(p, 'w'), indent=2)
PY
cp "$LK" /tmp/lk_config.bak
restore_lk () { cp /tmp/lk_config.bak "$LK"; }
trap restore_lk EXIT
status () { echo "$(date '+%F %T') | LongKou MILD | $*" >> runs/STATUS.txt; }
CODES="main.py datasets.py model_sz.py pipeline.py utils.py eval.py"
mkdir -p runs

sed -i 's/"lambda_clip": 0.0/"lambda_clip": 1.0/' "$LK"
# 红线 3：λ=0 必挂。sed 对空格敏感，失配时是空操作 —— 开跑前必须读回确认。
GOT=$($PY -c "import json;print(json.load(open('$LK'))['lambda_clip'])")
if [ "$GOT" != "1.0" ]; then
  status "!! lambda_clip=$GOT（应 1.0），中止"
  echo "!! lambda_clip=$GOT（应 1.0），中止" >&2
  exit 2
fi
status "启动 sampler_power=0.5（lambda_clip=1.0 已确认）"

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
