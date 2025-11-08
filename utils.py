import json
import numpy as np
import scipy.io as io
import torch
import random

LABEL_PROJECTION = dict()

def read_mat(mat_dir: str) \
        -> np.ndarray:
    """
    Get mat data.
    -----------------------------------------
    Arguments:
        mat_dir(str): Dir about mat file.
    -----------------------------------------
    Returns: ndarray data.
    """
    mat = io.loadmat(mat_dir)
    for data in mat.values():
        if type(data) == np.ndarray:
            numpy_data = np.array(data)
            return numpy_data
    raise Exception("Wrong mat file!")


def normalize(data, if_255=False) \
        -> np.ndarray:
    """
    Norm data and if set range to [0, 255].
    -------------------------------------------------
    Arguments:
        data: Source data.
    -------------------------------------------------
    Returns: Norm data or norm to [0, 255] data
    """
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return norm_data * 255 if if_255 else norm_data


def get_train_test_num(gt, train_num, least=15):
    """
    Get per class train num and test num.
    -------------------------------------------------
    Arguments:
        gt: All ground truth
        train_num:
        least: The least of train num
    -------------------------------------------------
    Returns: Train num and test num of per class.
    """
    class_num = np.max(gt)

    train_list = np.ones(shape=(class_num,))
    test_list = np.ones(shape=(class_num,))
    if train_num < 1:
        for classes in range(class_num):
            per_class_num = np.sum(gt == classes + 1)
            theo_num = int(per_class_num * train_num)
            real_num = theo_num if per_class_num > theo_num + 50 else least

            train_list[classes] = real_num
            test_list[classes] = (per_class_num - real_num)
    elif train_num == 1:
        pass
    elif train_num > 1:
        train_num = int(train_num)
        for classes in range(class_num):
            per_class_num = np.sum(gt == classes + 1)
            train_num = train_num if per_class_num > train_num else int(per_class_num / 2)
            train_list[classes] = train_num
            test_list[classes] = per_class_num - train_num
    else:
        raise ValueError("Wrong train num!")
    return train_list, test_list

def fixed_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False


def split_gt(gt, train_list, test_list, label_list):
    train_list = train_list.astype(int)
    test_list = test_list.astype(int)

    train_zero_gt = np.zeros_like(gt)
    test_zero_gt = np.zeros_like(gt)
    test_seen_zero_gt = np.zeros_like(gt)

    classes = len(train_list)
    for label in range(1, classes + 1):
        current_label = np.argwhere(gt == label)
        np.random.shuffle(current_label)
        train_num = train_list[label - 1]
        
        train_loc = current_label[0: train_num, :]
        test_loc = current_label
        test_seen_loc = current_label[train_num:, ]
        if label in label_list:
            train_zero_gt[train_loc[:, 0], train_loc[:, 1]] = label
            test_seen_zero_gt[test_seen_loc[:, 0], test_seen_loc[:, 1]] = label
        else:
            test_zero_gt[test_loc[:, 0], test_loc[:, 1]] = label
    return train_zero_gt, test_zero_gt, test_seen_zero_gt


def fix_label(label_name: dict, gt: np.ndarray, shift=0):
    index = 1 + shift
    label = np.zeros_like(gt, dtype=int)
    mask = np.isin(gt, list(label_name.values())).astype(int)
    for key, value in label_name.items():
        current_mask = (gt == value) & (mask == 1)
        label[current_mask] = index
        label_name[key] = index
        index += 1
    return label, np.max(label)


def torch_gpu_to_numpy(data: torch.Tensor):
    return data.detach().cpu().numpy()


def get_seen_unseen_class(label_value, keys=False):
    seen, unseen = label_value["seen"], label_value["unseen"]
    if keys is True:
        return seen, unseen
    seen_label = np.array(list(seen.values()))
    unseen_label = np.array(list(unseen.values()))
    return seen_label, unseen_label


def reloc_class(classes: dict, unseen_class: list):
    class_name = list(classes)
    reloc_dict = {
        "seen": {},
        "unseen": {},
        "all": {}
    }
    for clss in class_name:
        reloc_dict["all"].update({clss: classes[clss]})
        if clss in unseen_class:
            reloc_dict["unseen"].update({clss: classes[clss]})
        else:
            reloc_dict["seen"].update({clss: classes[clss]})
    return reloc_dict, reloc_dict["seen"], reloc_dict["unseen"]


def get_config(dirs: str):
    with open(dirs, "r") as fig:
        params = json.loads(fig.read())
    return params
