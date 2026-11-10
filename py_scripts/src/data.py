"""Data loading and augmentation.

The Kaggle PNGs are converted once by prepare_data.py into uint8 numpy arrays
(data/cache/*.npy); this module only reads that cache.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def load_cache(cache_dir):
    x = np.load(os.path.join(cache_dir, "train_x.npy"))        # (50000, 32, 32, 3) uint8
    y = np.load(os.path.join(cache_dir, "train_y.npy"))        # (50000,) int64
    test_x = np.load(os.path.join(cache_dir, "test_x.npy"))    # (N_test, 32, 32, 3) uint8
    test_ids = np.load(os.path.join(cache_dir, "test_ids.npy"))
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    return x, y, test_x, test_ids, meta


def stratified_split(y, val_size, seed):
    """Return (train_idx, val_idx) with val_size images, equal per class."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per_class = val_size // len(classes)
    val_idx = []
    for c in classes:
        idx = np.where(y == c)[0]
        val_idx.append(rng.choice(idx, per_class, replace=False))
    val_idx = np.sort(np.concatenate(val_idx))
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    return train_idx, val_idx


class ArrayDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.transform(Image.fromarray(self.x[i]))
        label = -1 if self.y is None else int(self.y[i])
        return img, label


class Cutout:
    """Zero out one random size x size square (DeVries & Taylor, 2017). Applied to a normalized tensor."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        _, h, w = img.shape
        cy = int(torch.randint(h, (1,)))
        cx = int(torch.randint(w, (1,)))
        y1, y2 = max(0, cy - self.size // 2), min(h, cy + self.size // 2)
        x1, x2 = max(0, cx - self.size // 2), min(w, cx + self.size // 2)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img


def build_transforms(acfg, mean, std):
    train = [T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()]
    policy = acfg.get("policy", "none")
    if policy == "trivialaugment":
        train.append(T.TrivialAugmentWide())
    elif policy == "autoaugment":
        train.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
    elif policy != "none":
        raise ValueError(f"unknown augmentation policy: {policy}")
    train += [T.ToTensor(), T.Normalize(mean, std)]
    if acfg.get("cutout", 0) > 0:
        train.append(Cutout(acfg["cutout"]))

    test = [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(train), T.Compose(test)
