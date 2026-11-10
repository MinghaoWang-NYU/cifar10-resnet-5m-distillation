import copy
import csv
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml


# ----------------------------------------------------------------------------- config / misc
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CSVLogger:
    def __init__(self, path, fields, append=False):
        self.path, self.fields = path, fields
        if not (append and os.path.exists(path)):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fields).writeheader()

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.fields).writerow(row)


# ----------------------------------------------------------------------------- optimisation
def build_optimizer(model, tcfg):
    """Weight decay on conv / linear weights only (not on BatchNorm parameters or biases)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": tcfg["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    if tcfg["optimizer"] == "sgd":
        return torch.optim.SGD(groups, lr=tcfg["lr"], momentum=tcfg["momentum"],
                               nesterov=tcfg.get("nesterov", True))
    if tcfg["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=tcfg["lr"])
    raise ValueError(f"unknown optimizer: {tcfg['optimizer']}")


def warmup_cosine(total_steps, warmup_steps):
    """Per-step LR multiplier: linear warm-up, then cosine decay to 0."""
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


class EMA:
    """Exponential moving average of model weights (BN buffers are averaged too)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # warm-up: smaller decay during the first few thousand steps so early EMA weights are not stuck at init
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


# ----------------------------------------------------------------------------- losses / mixing
def smooth_one_hot(y, num_classes, smoothing):
    off = smoothing / num_classes
    t = torch.full((y.size(0), num_classes), off, device=y.device)
    t.scatter_(1, y.unsqueeze(1), 1.0 - smoothing + off)
    return t


def soft_cross_entropy(logits, target):
    return torch.sum(-target * F.log_softmax(logits.float(), dim=1), dim=1).mean()


def kd_loss(student_logits, teacher_logits, T):
    """Hinton et al. 2015: T^2 * KL(teacher_T || student_T)."""
    s = F.log_softmax(student_logits.float() / T, dim=1)
    t = F.softmax(teacher_logits.float() / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def cutmix(x, target, alpha):
    """CutMix (Yun et al. 2019) on a batch; target is a soft-label matrix."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[2], x.shape[3]
    cut = math.sqrt(1.0 - lam)
    ch, cw = int(H * cut), int(W * cut)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
    x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, lam * target + (1.0 - lam) * target[perm]


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict_logits(model, loader, device, tta_flip=False):
    """Return softmax probabilities; with tta_flip, average over the image and its horizontal flip."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        p = F.softmax(model(x).float(), dim=1)
        if tta_flip:
            p = 0.5 * (p + F.softmax(model(torch.flip(x, dims=[3])).float(), dim=1))
        probs.append(p.cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def accuracy(model, loader, device, tta_flip=False):
    probs, labels = predict_logits(model, loader, device, tta_flip)
    return (probs.argmax(1) == labels).float().mean().item()
