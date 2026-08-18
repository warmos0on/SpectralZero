import argparse
#用于解析命令行参数
import json
import os
import numpy as np
import torch
import torch.utils.data

from extract_text.extract_demo import extract
import pipeline
import eval
import datasets
import utils
import model_sz
import loger


def entry(args, log: loger.Logger):
    seed = args["random_seed"]
    # Fixed Seed
    utils.fixed_seed(seed)
    # 设置numpy打印选项，显示所有数据，不换行
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # 读取数据集，返回HSI, ground truth, label-classes对
    HSI, gt, label_value = datasets.get_dataset(dataset_name=args["dataset"])
    # 返回seen unseen的label-classes对
    label_value, seen_label_name, unseen_label_name = utils.reloc_class(label_value, args["unseen_classes"])
    # 更新args，添加class_num和bands
    args.update({'class_num': np.max(gt), 'bands': HSI.shape[-1]})
    # 获取text embedding和新的args
    text_embedding, args, train_att = extract(label_value, args)
    # 将label重新映射以满足cross entropy loss
    seen_gt, _ = utils.fix_label(seen_label_name, gt)
    seen_max = len(seen_label_name)
    unseen_gt, _ = utils.fix_label(unseen_label_name, gt, shift=seen_max)
    gt = seen_gt + unseen_gt
    # 获取训练和测试集使用数据的数量
    train_num, test_num = utils.get_train_test_num(gt=gt, train_num=args["train_num"])
    # 研究用：人为构造不同长尾程度（对每个 seen 类的训练配额设上限）
    # cap 文件为 {"类别名": 最大训练像素数}，只在类别属于 seen 时生效
    cap_map = {}
    if args.get("train_cap", None):
        with open(args["train_cap"], "r", encoding="utf-8") as cap_file:
            cap_map = json.load(cap_file)
        for k, name in enumerate(seen_label_name):
            if name in cap_map:
                train_num[k] = min(int(train_num[k]), int(cap_map[name]))
    args.update({'seen_class': seen_max})
    
    padding = int(args['patch_size'] / 2)
    # padding
    HSI = np.pad(HSI, ((padding, padding), (padding, padding), (0, 0)), 'symmetric')
    padding_gt = np.pad(gt, ((padding, padding), (padding, padding)), 'constant')
    
    seen_label, _ = utils.get_seen_unseen_class(label_value)
    
    train_gt, test_gt, test_seen_gt = utils.split_gt(padding_gt, train_num, test_num, seen_label)

    # 逐波段 z-score 标准化：统计量只来自训练(seen)像素，避免 unseen 信息泄漏
    if args.get("use_zscore", True):
        train_mask = train_gt > 0
        train_pixels = HSI[train_mask].astype(np.float32)
        band_mean = train_pixels.mean(axis=0)
        band_std = train_pixels.std(axis=0)
        band_std[band_std == 0] = 1.0
        HSI = ((HSI.astype(np.float32) - band_mean) / band_std).astype(np.float32)

    train_dataset = datasets.HyperProcess(HSI, train_gt, **args)
    test_dataset = datasets.HyperProcess(HSI, test_gt, **args)
    test_seen_dataset = datasets.HyperProcess(HSI, test_seen_gt, **args)

    g = torch.Generator()
    g.manual_seed(seed)
    if args.get("use_balanced_sampler", True):
        # 类别平衡采样：按训练样本数反比加权，缓解类别极度不均衡
        train_labels = np.array(train_dataset.labels, dtype=np.int64)
        class_counts = np.bincount(train_labels - 1, minlength=seen_max).astype(np.float64)
        sample_weights = 1.0 / class_counts[train_labels - 1]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(train_labels), replacement=True, generator=g)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args["batch_size"],
            pin_memory=args["pin_memory"],
            sampler=sampler,
            shuffle=False
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args["batch_size"],
            pin_memory=args["pin_memory"],
            generator=g,
            shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        pin_memory=args["pin_memory"],
        generator=g,
        shuffle=True
    )
    test_seen_loader = torch.utils.data.DataLoader(
        test_seen_dataset,
        batch_size=args["batch_size"],
        pin_memory=args["pin_memory"],
        generator=g,
        shuffle=True
    )

    model = model_sz.SpectralZero(args).to(args["device"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])

    for epoch in range(1, args["epochs"] + 1):
        model, pre, tar = pipeline.train_pipeline(model, text_embedding, train_loader, epoch, args, optimizer)
        if epoch % args["checkpoints"] == 0:
            pre, tar = pipeline.test_pipeline(model, text_embedding, test_loader, test_seen_loader, args)
            
            evals = eval.evaluation(pre, tar)
            unseen_OA, unseen_per_AA, unseen_AA = evals.get_acc()
            kappa = evals.get_kappa()
            unseen_msg = f"unseen OA is {unseen_OA:.2f}, unseen AA is {unseen_AA:.2f}, kappa is {kappa}"
            per_unseen_AA_msg = f"unseen per AA {unseen_per_AA}, kappa is {kappa}"

            print(unseen_msg)
            print(per_unseen_AA_msg)
            log.INFO_log(per_unseen_AA_msg)
    log.CRITICAL_log("Current experment have been finished.\n")
    

if __name__ == "__main__":
    # 解析命令行参数
    args = argparse.ArgumentParser(description="Dataset Choise")
    args.add_argument("--dataset", type=str, default="Houston", choices=["Indian", "Houston", "LongKou"])
    args.add_argument("--seed", type=int, default=None)
    args.add_argument("--unseen", type=str, default=None, help="comma-separated unseen class names")
    args.add_argument("--epochs", type=int, default=None)
    args.add_argument("--checkpoints", type=int, default=None)
    args.add_argument("--use_zscore", type=int, default=None)
    args.add_argument("--use_balanced_sampler", type=int, default=None)
    args.add_argument("--use_unseen_negatives", type=int, default=None)
    args.add_argument("--dynamic_fusion", type=int, default=None)
    args.add_argument("--train_cap", type=str, default=None, help="per-class training count cap JSON (long-tail ablation)")
    args = vars(args.parse_args())

    params = utils.get_config(f"./config/{args['dataset']}.json")
    if args["seed"] is not None:
        params["random_seed"] = args["seed"]
    if args["unseen"] is not None:
        params["unseen_classes"] = [x.strip() for x in args["unseen"].split(",")]
    if args["epochs"] is not None:
        params["epochs"] = args["epochs"]
    if args["checkpoints"] is not None:
        params["checkpoints"] = args["checkpoints"]
    for flag in ["use_zscore", "use_balanced_sampler", "use_unseen_negatives"]:
        if args[flag] is not None:
            params[flag] = bool(args[flag])
    if args["dynamic_fusion"] is not None:
        params["dynamic_fusion"] = bool(args["dynamic_fusion"])
    if args["train_cap"] is not None:
        params["train_cap"] = args["train_cap"]
    # 日志文件名加上 seed 和 group 方便区分
    log_suffix = ""
    if args["seed"] is not None:
        log_suffix += f"_seed{args['seed']}"
    if args["unseen"] is not None:
        import hashlib
        log_suffix += f"_g{hashlib.md5(args['unseen'].encode()).hexdigest()[:4]}"
    if args["train_cap"] is not None:
        cap_name = os.path.splitext(os.path.basename(args["train_cap"]))[0]
        log_suffix += f"_cap{cap_name}"
    logs = loger.Logger(minimum_level="INFO", log_path=r"./log_" + params["dataset"] + log_suffix + ".txt")
    entry(params, logs)
