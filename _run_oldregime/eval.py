import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score


class evaluation:
    def __init__(self, pred, targ, binary=None):
        self.min_label = np.min(targ)
        self.pre = (pred - self.min_label).astype(np.int8)
        self.tar = (targ - self.min_label).astype(np.int8)
        self.binary = binary

    def get_acc(self):
        # 1. 打印一下，让你亲眼看看“答题卡”是怎么错位的！
        print(f"\n[阅卷诊断] 测试集的真实标签种类有: {np.unique(self.tar)}")
        print(f"[阅卷诊断] 模型实际预测出的标签种类有: {np.unique(self.pre)}\n")

        # 2. 自动对齐：把真实标签（比如 12,13,14）重新映射成 0,1,2
        unique_labels = np.unique(self.tar)
        label_map = {val: idx for idx, val in enumerate(np.sort(unique_labels))}
        self.mapped_tar = np.array([label_map[val] for val in self.tar]) # 存下来给Kappa用

        # 3. 使用对齐后的 mapped_tar 来计算混淆矩阵
        confusion_matrixs = confusion_matrix(self.mapped_tar, self.pre)
        
        # 4. 动态获取类别数（之前我们改过的）
        num_classes = confusion_matrixs.shape[0]
        per_class_same = np.zeros(shape=(num_classes, ))
        
        self.unseen_OA = np.sum(np.trace(confusion_matrixs)) / np.sum(confusion_matrixs) * 100
        for classes in range(num_classes):
            if np.sum(confusion_matrixs[classes, :]) != 0:
                per_class_same[classes] = confusion_matrixs[classes, classes] / np.sum(confusion_matrixs[classes, :]) * 100
            else:
                per_class_same[classes] = 0.
        self.unseen_per_AA = per_class_same
        return self.unseen_OA, np.around(self.unseen_per_AA, 2), np.average(self.unseen_per_AA)

    def get_kappa(self):
        # 算 Kappa 的时候也要用对齐后的 mapped_tar！
        return np.around(cohen_kappa_score(self.pre, getattr(self, 'mapped_tar', self.tar)), 5)
