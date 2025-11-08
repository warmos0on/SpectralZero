import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score


class evaluation:
    def __init__(self, pred, targ, binary=None):
        self.min_label = np.min(targ)
        self.pre = (pred - self.min_label).astype(np.int8)
        self.tar = (targ - self.min_label).astype(np.int8)
        self.binary = binary

    def get_acc(self):
        per_class_same = np.zeros(shape=(3, ))
        confusion_matrixs = confusion_matrix(self.tar, self.pre)
        self.unseen_OA = np.sum(np.trace(confusion_matrixs)) / np.sum(confusion_matrixs) * 100
        for classes in range(confusion_matrixs.shape[0]):
            if np.sum(confusion_matrixs[classes, :]) != 0:
                per_class_same[classes] = confusion_matrixs[classes, classes] / np.sum(confusion_matrixs[classes, :]) * 100
            else:
                per_class_same[classes] = 0.
        self.unseen_per_AA = per_class_same
        return self.unseen_OA, np.around(self.unseen_per_AA, 2), np.average(self.unseen_per_AA)

    def get_kappa(self):
        return np.around(cohen_kappa_score(self.pre, self.tar), 5)
