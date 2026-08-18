import numpy as np
import torch
import torch.utils
import torch.utils.data

import utils

DATASETS_CONFIG = {
    'Houston': {
        'img': 'Houston.mat',
        'gt': 'Houston_gt.mat',
    },
    'Pavia': {
        'img': 'Pavia.mat',
        'gt': 'Pavia_gt.mat',
    },
    'Indian': {
        'img': 'Indian.mat',
        'gt': 'Indian_gt.mat',
    },
    'Salina': {
        'img': 'Salina.mat',
        'gt': 'Salina_gt.mat',
    },
    'HanChuan': {
        'img': 'HanChuan.mat',
        'gt': 'HanChuan_gt.mat',
    },
    'Houston2018': {
        'img': 'Houston2018.mat',
        'gt': 'Houston2018_gt.mat',
    },
}


def get_dataset(dataset_name, dataset_loc="./data/"):
    # get raw HSI
    HSI = utils.read_mat(f"{dataset_loc}{dataset_name}.mat")
    # get ground truth
    gt = utils.read_mat(f"{dataset_loc}{dataset_name}_gt.mat")

    # label information
    if dataset_name == "Indian":
        label_values = {
            "Grass-trees": 6,
            "Buildings-Grass-Trees-Drives": 15,
            "Corn-notill": 2,
            "Grass-pasture-mowed": 7,
            "Corn": 4,
            "Woods": 14,
            "Soybean-mintill": 11,
            "Soybean-clean": 12,
            "Stone-Steel-Towers": 16,
            "Hay-windrowed": 8,
            "Wheat": 13,
            "Alfalfa": 1,
            "Oats": 9,
            "Soybean-notill": 10,
            "Corn-mintill": 3,
            "Grass-pasture": 5,
        }
    elif dataset_name == "LongKou":
        label_values = {
            "Corn": 1,
            "Cotton": 2,
            "Sesame": 3,
            "Broad-leaf soybean": 4,
            "Narrow-leaf soybean": 5,
            "Rice": 6,
            "Water": 7,
            "Mixed weed": 9,
            "Roads and houses": 8
        }
    elif dataset_name == "Houston":
        label_values = {
            "Healthy grass": 1,
            "Water": 6,
            "Tennis Court": 14,
            "Parking Lot 2": 13,
            "Residential": 7,
            "Stressed grass": 2,
            "Road": 9,
            "Synthetic grass": 3,
            "Running Track": 15,
            "Commercial": 8,
            "Railways": 11,
            "Soil": 5,
            "Trees": 4,
            "Highways": 10,
            "Parking Lot 1": 12,
        }
    elif dataset_name == "PaviaU":
        label_values = {
            "Asphalt": 1,
            "Self-Blocking Bricks": 8,
            "Shadows": 9,
            "Meadows": 2,
            "Painted metal sheets": 5,
            "Bare Soil": 6,
            "Trees": 4,
            "Gravel": 3,
            "Bitumen": 7,
        }
    else:
        ValueError("Wrong dataset name!")
    nan_mask = np.isnan(HSI.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("There are some NaN data in the HSI. The program will mask them as zero.")
    HSI[nan_mask] = 0
    gt[nan_mask] = 0
    return HSI, gt, label_values


class HyperProcess(torch.utils.data.Dataset):
    def __init__(self, data, gt, transform=None, **hyperparams):
        super(HyperProcess, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = {0}
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 p < x < data.shape[0] - p and p < y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 18):
        random_use = np.random.randint(0, 1, size=(1,))
        if random_use == 0:
            return data
        else:
            alpha = np.random.uniform(*alpha_range)
            noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
            return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]
        return data, label
