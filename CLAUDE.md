# SpectralZero · 代码库约定

> 给跑实验 / 改代码的人和 AI 看的硬约定。超参口径以 `论文版基线_改动说明.md` 为准。
> 背景：2026-10 硬盘损毁，实验数据 / 日志 / 权重全部丢失，本仓库是唯一幸存上下文。

## 一、红线（违反 = 实验静默作废）

1. **创新开关必须显式声明**。`use_zscore` / `use_balanced_sampler` / `use_unseen_negatives` / `dynamic_fusion` 的代码默认值已是 **False**（2026-10-04 修复，原为 True 会静默启用）。config 仍须写全开关，跑基线前逐项确认全为 false。
2. **`lambda_clip` 是 InfoNCE 权重** = 论文 Eq.(7) 的 `1 − λ_loss`（`L_total = λ_loss·L_CE + (1−λ_loss)·L_InfoNCE`）。从论文抄超参先换算。
3. **LongKou 的 `lambda_clip=0.0` 不能直接跑**：InfoNCE 权重为 0 时两个投影头和光谱支路零梯度，但测试走 argmax 余弦相似度（`spectral_ratio=0.6`），结果必挂。跑 WHHL 前先定口径（试 0.6 / 1.0 小跑对照），结论记进实验记录。
4. **三数据集超参不互换**：`spectral_ratio` 0.5 / 0.2 / 0.6，`lambda_clip` 0.6 / 0.6 / 0.0，`patch_size` 15 / 11 / 13，lr、epochs、logit_scale 初值各不同。
5. **`batch_size` 因显存下调必须记录**。论文 4090 24GB 用 1024；本机 RTX 4060 Laptop 8GB。
6. **评估是纯开集**：unseen 类全像素作测试、seen 类精度从未评（`test_seen_loader` 是死参数）。报的 "unseen OA" 不要和 seen 精度混着比。
7. **OA/AA 已双写日志**（2026-10-04 修复；此前只留 per-AA 行）。直接跑 `main.py` 仍建议重定向 stdout 作双保险。

## 二、踩坑表

> 标 ✅ 的已修复（2026-10-04，冒烟测试通过）；未标注的仍在。

| 位置 | 坑 |
| --- | --- |
| utils.py:78 | ✅ 已修：`get_train_test_num` 整数分支配额级联污染（改用局部变量） |
| utils.py:49-83 | `train_num` 三态：`<1` 比例（内含 `least=15` 与 `+50` 硬阈值），`==1` 每类 1 个，`>1` 固定数 |
| utils.py:96 | `split_gt` 的 `test_list` 传而不用；seen = 全像素 − 训练，unseen = 全像素（不对称） |
| utils.py:128 | ✅ 已修：`fix_label` 改为纯函数，返回新 dict（调用点已适配） |
| analyze_longtail.py | ✅ 已修：`seen_label` 曾取原始 gt id（漏统计 3 个类），现取重映射后 id |
| model_sz.py:146 | ✅ 已修：`squeeze` 改显式 `squeeze(1)`，B=1 可推理；训练端 B=1 仍跳过（BatchNorm 限制） |
| model_sz.py:191 | `logit_scale` 初值按数据集（LongKou 0.17，其余 0.1），且无 CLIP 的 `clamp(max=100)` |
| model_sz.py:257 | `clip_loss` 两套正样本：`use_unseen_negatives=true` → 真类列（列空间=全部文本）；`false` → batch 对角 arange。切换时对比口径变化 |
| datasets.py:153 | ✅ 已修：`radiation_noise` 恢复真实辐射增强——**注意：`radiation_augmentation: true` 从今起真的生效，基线数字与修复前不可比，须在实验记录声明** |
| datasets.py:105 | ✅ 已修：`ValueError` 补 `raise`；`DATASETS_CONFIG` 仍是死字典且缺 LongKou |
| run_honest_eval.py:42 | ✅ 已修：`--ablation` 裸名自动补 `=1`；`flag=value` 形式原样透传 |
| pipeline.py:7 | `CUDA_LAUNCH_BLOCKING=1` 调试残留，拖慢训练 |
| extract_text/extract_demo.py:23 | `.cuda()` 硬编码，忽略 config 的 `device` |
| config/* | 死配置（改了没任何作用）：`use_logit_adjust` `use_virtual_prototypes` `dropout` `multi_args` `dataroot`（前两个是创新预留槽位，见 `创新点评审结论.md`） |

## 三、命令

```bash
# 环境已配自动激活（bash / PowerShell / cmd 打开 shell 即进入 SpectralZero）
# conda 本体在 D:\program\miniconda3，环境在 C:\Users\ASUS\.conda\envs\SpectralZero
# 未激活时手动：conda activate SpectralZero   （Python 3.12 + PyTorch 2.11 cu128，CUDA 可用）
python main.py --dataset Indian      # 训练入口（默认 Houston）
python run_honest_eval.py --dataset Indian --seeds 42 --epochs 20 --checkpoints 5 --ablation "use_zscore=0"
.\run_ablation.ps1                   # 消融批跑
python make_longtail_caps.py         # 生成按类训练配额上限 JSON
python run_longtail_batch.py         # 长尾批跑；analyze_longtail*.py / analyze_perclass.py 出分析
```

**运行前置（2026-10 时缺失，需重下）**：

- `data/`：Indian.mat / Indian_gt.mat / Houston.mat / Houston_gt.mat / LongKou.mat / LongKou_gt.mat
- `extract_text/ViT-L-14.pt`：CLIP 文本编码器权重
