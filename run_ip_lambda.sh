#!/bin/bash
# Indian Pines · λ 裁决：lambda_clip ∈ {0.4, 0.6}
#   用法: bash run_ip_lambda.sh [seed...]     默认 42
#
# 背景：config 的 0.6 = 字面换算 1−λ_loss(0.4)；自洽读法（λ_loss 即 InfoNCE 权重）
#       下应为 0.4。本实验只改 λ，其余全部论文口径，同代码同机器，用来裁决哪一种读法
#       更接近论文逐类值（论文 IP AA=81.45）。
# 失败不中断；断点续跑靠 <arm>/<Gx>.done。
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
PY=/root/miniconda3/bin/python
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42)

IND=config/Indian.json
cp "$IND" /tmp/indian_config.bak
restore () { cp /tmp/indian_config.bak "$IND"; }
trap restore EXIT

mkdir -p runs
CODES="main.py datasets.py model_sz.py pipeline.py utils.py eval.py"
status () { echo "$(date '+%F %T') | IP lambda | $*" >> runs/STATUS.txt; }

echo "#### IP λ 裁决 启动 $(date '+%F %T') seeds=${SEEDS[*]} ####"
status "IP λ 开始"

for SEED in "${SEEDS[@]}"; do
  for LC in 0.4 0.6; do
    ARM="lc${LC}"
    OUT="runs/ip_${ARM}_seed${SEED}"
    mkdir -p "$OUT"
    $PY - "$LC" <<'PY'
import json,sys
p='config/Indian.json'; d=json.load(open(p)); d['lambda_clip']=float(sys.argv[1])
json.dump(d,open(p,'w'),indent=2)
PY
    cp "$IND" "$OUT/config_used.json"
    md5sum $CODES > "$OUT/code_md5.txt"
    { echo "arm=$ARM"; echo "lambda_clip=$LC"; echo "seed=$SEED"
      echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
      echo "started=$(date '+%F %T')"; } > "$OUT/meta.txt"

    echo; echo "########## IP λ=$LC SEED=$SEED $(date '+%F %T') ##########"
    status "IP λ=$LC seed=$SEED 开始"
    for FOLD in "G1|Corn-notill,Hay-windrowed,Woods" \
                "G2|Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill" \
                "G3|Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill" \
                "G4|Corn,Alfalfa,Grass-pasture" \
                "G5|Wheat,Oats,Soybean-notill" \
                "G6|Soybean-clean"; do
      G="${FOLD%%|*}"; UN="${FOLD##*|}"
      if [ -f "$OUT/${G}.done" ]; then echo "[skip] $G (λ=$LC)"; continue; fi
      echo "[start] $G λ=$LC seed=$SEED $(date '+%F %T')"
      status "IP λ=$LC seed=$SEED $G 开始"
      $PY -u main.py --dataset Indian --seed "$SEED" --unseen "$UN" > "$OUT/${G}.stdout.log" 2>&1
      RC=$?
      echo "$RC" > "$OUT/${G}.exit"
      for f in log_Indian_seed${SEED}_g*.txt; do [ -f "$f" ] && mv "$f" "$OUT/"; done
      if [ $RC -eq 0 ]; then
        echo "$(date '+%F %T')" > "$OUT/${G}.done"
        echo "[done ] $G λ=$LC seed=$SEED $(date '+%F %T')"
        status "IP λ=$LC seed=$SEED $G 完成"
      else
        echo "[FAIL ] $G λ=$LC seed=$SEED exit=$RC $(date '+%F %T')"
        status "IP λ=$LC seed=$SEED $G 失败 exit=$RC"
      fi
    done
    restore
    status "IP λ=$LC seed=$SEED 结束"
  done
done

restore
status "IP λ 全部结束"
echo; echo "#### IP λ 完成 $(date '+%F %T') ####"
