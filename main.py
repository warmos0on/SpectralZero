import argparse
#用于解析命令行参数
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
    seen_gt, seen_max = utils.fix_label(seen_label_name, gt)
    unseen_gt, _ = utils.fix_label(unseen_label_name, gt, shift=seen_max)
    gt = seen_gt + unseen_gt
    # 获取训练和测试集使用数据的数量
    train_num, test_num = utils.get_train_test_num(gt=gt, train_num=args["train_num"])
    args.update({'seen_class': seen_max})
    
    padding = int(args['patch_size'] / 2)
    # padding
    HSI = np.pad(HSI, ((padding, padding), (padding, padding), (0, 0)), 'symmetric')
    padding_gt = np.pad(gt, ((padding, padding), (padding, padding)), 'constant')
    
    seen_label, _ = utils.get_seen_unseen_class(label_value)
    
    train_gt, test_gt, test_seen_gt = utils.split_gt(padding_gt, train_num, test_num, seen_label)

    train_dataset = datasets.HyperProcess(HSI, train_gt, **args)
    test_dataset = datasets.HyperProcess(HSI, test_gt, **args)
    test_seen_dataset = datasets.HyperProcess(HSI, test_seen_gt, **args)

    g = torch.Generator()
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
    
    for epoch in range(1, args["epochs"] + 1):
        model, pre, tar = pipeline.train_pipeline(model, text_embedding, train_loader, epoch, args)
        if epoch % args["checkpoints"] == 0:
            pre, tar = pipeline.test_pipeline(model, text_embedding, test_loader, test_seen_loader, args)
            
            evals = eval.evaluation(pre, tar)
            unseen_OA, unseen_per_AA, unseen_AA = evals.get_acc()
            kappa = evals.get_kappa()
            unseen_msg = f"unseen OA is {unseen_OA:.2f}, unseen AA is {unseen_AA:.2f}"
            per_unseen_AA_msg = f"unseen per AA {unseen_per_AA}, kappa is {kappa}"

            print(unseen_msg)
            log.INFO_log(per_unseen_AA_msg)
    log.CRITICAL_log("Current experment have been finished.\n")
    

if __name__ == "__main__":
    # 解析命令行参数
    args = argparse.ArgumentParser(description="Dataset Choise")
    # 数据集选择
    args.add_argument("--dataset", type=str, default="Houston", choices=["Indian", "Houston", "LongKou"])
    #将命令行参数转换为字典格式，方便后续使用
    args = vars(args.parse_args())

    params = utils.get_config(f"./config/{args['dataset']}.json")
    # 设置日志记录器
    logs = loger.Logger(minimum_level="INFO", log_path=r"./log_" + params["dataset"] + ".txt")
    entry(params, logs)