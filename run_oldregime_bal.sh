#!/bin/bash
# 旧配置复刻：只加类平衡采样（对照两个月前 archive 的 F 组）
#
# 目的：验证「旧配置下平衡采样确实提升」这一说法，且**保留全部评测点**（不取末点）。
#       旧实验只跑 20 轮 / 5 个评测点 / 取最后一个；本脚本跑同样轮数，但记录并分析全部 5 点。
#
# 复刻来源：origin/archive-full 的 config/Indian.json + run_ablation.ps1 的调用方式
#   注意：旧 archive 的 Indian config 里有若干「Houston 的值」——**原样复刻**，不改。
#   两处刻意的现代化（用户指定）：
#     ① 评测点全部保留（旧实验只留末点）
#     ② radiation_augmentation 设为 false —— 等价于旧代码里 radiation_noise 恒为 0 的空操作
#
# 用法: bash run_oldregime_bal.sh [seeds...]      默认 42 123 456
set -u
cd /root/autodl-tmp/SZ_oldregime || exit 1
PY=/root/miniconda3/bin/python
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 123 456)

mkdir -p runs logs_old_bal
DRIVER=runs/driver_old_bal.log
touch "$DRIVER"
mv log_*.txt logs_old_bal/ 2>/dev/null

# ---- 写旧配置（原样复刻 archive，只把 radiation 关掉以等价旧代码的空操作）----
cat > config/Indian.json <<'JSON'
{
    "dataset":  "Indian",
    "dataroot":  "./data",
    "train_num":  0.2,
    "attribute_dir":  "./attributes/",
    "pretrain_loc":  "./extract_text/ViT-L-14.pt",
    "random_seed":  114514,
    "device":  "cuda:0",
    "flip_augmentation":  true,
    "radiation_augmentation":  false,
    "mixture_augmentation":  false,
    "patch_size":  11,
    "batch_size":  256,
    "epochs":  150,
    "lr":  0.0002,
    "pin_memory":  true,
    "center_pixel":  true,
    "supervision":  "full",
    "checkpoints":  10,
    "lambda_clip":  0.8,
    "vision_dim":  128,
    "dropout":  0.1,
    "spectral_dim":  128,
    "multi_args":  true,
    "unseen_classes":  ["Corn-notill","Soybean-mintill","Buildings-Grass-Trees-Drives"],
    "spectral_ratio":  0.2,
    "dynamic_fusion":  true,
    "fusion_temperature":  0.07,
    "fusion_prior_strength":  1.0,
    "fusion_min_weight":  0.05,
    "use_zscore":  false,
    "use_balanced_sampler":  true,
    "use_unseen_negatives":  false,
    "only_name":  false
}
JSON
echo "=== 生效配置（关键项）==="
grep -E "patch_size|batch_size|lr|lambda_clip|spectral_ratio|dynamic_fusion|radiation_aug|use_balanced|use_zscore" config/Indian.json

# CLI 复刻 run_ablation.ps1 的 --epochs 20 --checkpoints 5，并开平衡采样
EPOCHS=20; CKPT=5
is_done () { grep -q "^\[done \] $1 .*exit=0" "$DRIVER"; }

run_fold () {  # $1=group $2=unseen $3=seed
  local tag="Indian_$1_seed$3_oldbal"
  if is_done "$tag"; then echo "[skip ] $tag"; return 0; fi
  echo "[start] $tag  $(date '+%F %H:%M:%S')"
  $PY -u main.py --dataset Indian --seed "$3" --unseen "$2" \
      --epochs $EPOCHS --checkpoints $CKPT --use_balanced_sampler 1 \
      > "runs/${tag}.log" 2>&1
  local rc=$?
  mv log_*.txt logs_old_bal/ 2>/dev/null
  echo "[done ] $tag  $(date '+%F %H:%M:%S')  exit=$rc"
}

for SEED in "${SEEDS[@]}"; do
  echo; echo "########## 旧配置复刻 · SEED $SEED ##########"
  run_fold G1 "Corn-notill,Hay-windrowed,Woods"                          $SEED
  run_fold G2 "Grass-trees,Buildings-Grass-Trees-Drives,Soybean-mintill" $SEED
  run_fold G3 "Grass-pasture-mowed,Stone-Steel-Towers,Corn-mintill"      $SEED
  run_fold G4 "Corn,Alfalfa,Grass-pasture"                               $SEED
  run_fold G5 "Wheat,Oats,Soybean-notill"                                $SEED
  run_fold G6 "Soybean-clean"                                            $SEED
  echo "[seed $SEED 完成]  $(date '+%F %H:%M:%S')"
done

echo "OLDREGIME_BAL_DONE  $(date '+%F %H:%M:%S')" > runs/.done_old_bal
echo; echo "########## 全部完成 ##########"
echo "loger 明细: logs_old_bal/    stdout: runs/"
