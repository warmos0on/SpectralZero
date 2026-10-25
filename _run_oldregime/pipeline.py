import torch
import numpy as np

import utils
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train_pipeline(model, text_projection, train_dataset, epoch, hyperparams, optimizer):
    model.train()
    LEARNING_RATE, device = hyperparams["lr"], hyperparams["device"]
    lambda_clip = hyperparams["lambda_clip"]

    pre, tar = [], []
    
    for batch_data, batch_label in train_dataset:
        batch_data, batch_label = batch_data.to(device), (batch_label - 1).int().to(device)
        # BatchNorm1d 在训练模式下无法对单样本 batch 求统计量（会直接抛错）；
        # 训练集长度 ≡ 1 (mod batch_size) 时末位 batch 为 1，跳过并计入日志。
        # 测试端不再丢样本：model_sz 已改用显式 squeeze(1)，batch=1 可正常前向。
        if batch_label.size(0) == 1:
            print(f"[epoch: {epoch:4}] 跳过末位单样本训练 batch（BatchNorm1d 限制）")
            continue
        optimizer.zero_grad()

        if hyperparams.get("use_unseen_negatives", False):
            # 训练时把完整文本矩阵（seen+unseen）作为类别空间，unseen 仅作负类
            text_input = text_projection
        else:
            text_input = text_projection[batch_label.long()]
        cls_loss, clip_loss, pred = model.forward(batch_data, text_input, batch_label)
        loss = lambda_clip * clip_loss + (1 - lambda_clip) * cls_loss
        
        loss.backward()
        optimizer.step()
        pre.extend(torch.argmax(pred, dim=1).cpu().numpy())
        tar.extend(batch_label.cpu().numpy())
    # print('loss: {:.6f}, loss_cls: {:.6f}, loss_clip: {:.6f}'.format(loss.item(), cls_loss.item(), clip_loss.item()))
    train_OA = np.sum(np.array(pre) == np.array(tar)) / len(tar) * 100
    train_msg = "[epoch: {:4}]  Train Accuracy: {:.5f} | train sample number: {:6}".format(epoch, train_OA, len(tar))
    print(train_msg)
    return model, np.array(pre), np.array(tar)


@torch.no_grad()
def test_pipeline(model, text_projection, test_dataset, test_seen_dataset, hyperparams):
    test_num = len(test_dataset.dataset)
    num = test_num
    device, seen_class = hyperparams["device"], 0
    model.eval()
    seen_class = hyperparams["seen_class"]
    text_projection = text_projection[seen_class: ]
    pre, tar = np.zeros(shape=(num, )), np.zeros(shape=(num, ))
    
    index = 0
    for batch_data, batch_label in test_dataset:
        batch_size = len(batch_label)
        batch_label, batch_data = (batch_label - 1).int().to(device), batch_data.to(device)
        pred, cls_ = model.forward(batch_data, text_projection, batch_label)

        tar[index: index + batch_size] = utils.torch_gpu_to_numpy(batch_label) - seen_class
        pre[index: index + batch_size] = utils.torch_gpu_to_numpy(pred)
        index += batch_size
    return pre, tar
