#!/bin/bash
# LongKou（WHHL）类平衡采样 A/B · 正式 protocol
#   用法: bash run_lk_ab.sh [seed...]        默认 42
#
# 唯一变量：use_balanced_sampler（base=false / bal=true）
# 其余全部论文值；lambda_clip 实跑临时置 1.0（config 0.0 必挂），退出即还原。
# 失败不中断（记 exit code 继续下一折）；断点续跑靠 <Gx>.done。
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
PY=/root/miniconda3/bin/python
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42)

LK=config/LongKou.json
cp "$LK" /tmp/lk_config.bak
restore_lk () { cp /tmp/lk_config.bak "$LK"; }
trap restore_lk EXIT

mkdir -p runs
CODES="main.py datasets.py model_sz.py pipeline.py utils.py eval.py"

status () { echo "$(date '+%F %T') | LongKou A/B | $*" >> runs/STATUS.txt; }

# ---------- 预检 ----------
preflight () {
  local bad=0 key want got
  for k in patch_size:13 batch_size:1024 epochs:50 lr:2e-05 checkpoints:5 spectral_ratio:0.6 train_num:0.2; do
    key="${k%%:*}"; want="${k##*:}"
    got=$($PY -c "import json;print(json.load(open('$LK'))['$key'])")
    [ "$got" != "$want" ] && { echo "  ✗ $key=$got (应 $want)"; bad=1; }
  done
  for f in dynamic_fusion use_zscore use_unseen_negatives use_logit_adjust use_virtual_prototypes; do
    got=$($PY -c "import json;print(json.load(open('$LK'))['$f'])")
    [ "$got" != "False" ] && { echo "  ✗ $f=$got (应 False)"; bad=1; }
  done
  got=$($PY -c "import json;print(json.load(open('$LK'))['lambda_clip'])")
  [ "$got" != "1.0" ] && { echo "  ✗ lambda_clip=$got (实跑应 1.0)"; bad=1; }
  [ $bad -eq 0 ] && echo "  预检通过"
  return $bad
}

echo "#### LongKou A/B 启动 $(date '+%F %T') seeds=${SEEDS[*]} ####"
sed -i 's/"lambda_clip": 0.0/"lambda_clip": 1.0/' "$LK"
preflight || { status "预检失败，中止"; exit 2; }
status "预检通过，开始"

for SEED in "${SEEDS[@]}"; do
  for ARM in base bal; do
    if [ "$ARM" = "bal" ]; then EXTRA=(--use_balanced_sampler 1); else EXTRA=(); fi
    OUT="runs/lk_${ARM}_seed${SEED}"
    mkdir -p "$OUT"
    cp "$LK" "$OUT/config_used.json"
    md5sum $CODES > "$OUT/code_md5.txt"
    { echo "arm=$ARM"; echo "seed=$SEED"
      echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
      echo "python=$($PY -c 'import sys;print(sys.version.split()[0])')"
      echo "torch=$($PY -c 'import torch;print(torch.__version__,torch.version.cuda)')"
      echo "started=$(date '+%F %T')"; echo "extra=${EXTRA[*]:-none}"; } > "$OUT/meta.txt"

    echo; echo "########## ARM=$ARM SEED=$SEED $(date '+%F %T') ##########"
    status "ARM=$ARM SEED=$SEED 开始"
    for FOLD in "G1|Corn,Cotton,Water" "G2|Sesame,Broad-leaf soybean,Rice" "G3|Narrow-leaf soybean,Mixed weed,Roads and houses"; do
      G="${FOLD%%|*}"; UN="${FOLD##*|}"
      if [ -f "$OUT/${G}.done" ]; then echo "[skip] $G ($ARM, seed $SEED)"; continue; fi
      echo "[start] $G arm=$ARM seed=$SEED $(date '+%F %T')"
      status "ARM=$ARM SEED=$SEED $G 开始"
      $PY -u main.py --dataset LongKou --seed "$SEED" --unseen "$UN" ${EXTRA[@]+"${EXTRA[@]}"} \
          > "$OUT/${G}.stdout.log" 2>&1
      RC=$?
      echo "$RC" > "$OUT/${G}.exit"
      for f in log_LongKou_seed${SEED}_g*.txt; do [ -f "$f" ] && mv "$f" "$OUT/"; done
      if [ $RC -eq 0 ]; then
        echo "$(date '+%F %T')" > "$OUT/${G}.done"
        echo "[done ] $G arm=$ARM seed=$SEED $(date '+%F %T')"
        status "ARM=$ARM SEED=$SEED $G 完成"
      else
        echo "[FAIL ] $G arm=$ARM seed=$SEED exit=$RC $(date '+%F %T')"
        status "ARM=$ARM SEED=$SEED $G 失败 exit=$RC"
      fi
    done
    status "ARM=$ARM SEED=$SEED 结束"
  done
done

restore_lk
status "LongKou A/B 全部结束"
echo; echo "#### LongKou A/B 完成 $(date '+%F %T') ####"
