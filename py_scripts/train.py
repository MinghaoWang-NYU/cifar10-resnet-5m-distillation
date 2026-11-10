"""Train a model from a YAML config.

Examples
    python train.py --config configs/main.yaml                      # 45k train / 5k val
    python train.py --config configs/main.yaml --full-train         # all 50k images, no val
    python train.py --config configs/kd.yaml --teacher runs/teacher_wrn28_10/last.pt
    python train.py --config configs/main.yaml --resume             # continue runs/<name>/last.pt

Outputs in runs/<name>/: config.yaml, log.csv (one row per epoch), last.pt, best.pt (val runs only).
"""
import argparse
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import ArrayDataset, build_transforms, load_cache, stratified_split
from src.model import build_model, count_trainable_params
from src.utils import (EMA, CSVLogger, accuracy, build_optimizer, cutmix, get_device, kd_loss,
                       load_config, set_seed, smooth_one_hot, soft_cross_entropy, warmup_cosine)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None, help="run name (default: config name)")
    ap.add_argument("--full-train", action="store_true", help="train on all 50k images (no validation split)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--teacher", default=None, help="teacher checkpoint for knowledge distillation")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.full_train:
        cfg["data"]["full_train"] = True
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.teacher is not None:
        cfg["kd"]["teacher_ckpt"] = args.teacher
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    name = args.name or cfg["name"] + ("_full" if cfg["data"]["full_train"] else "")
    run_dir = os.path.join("runs", name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    set_seed(cfg["seed"])
    device = get_device()
    torch.backends.cudnn.benchmark = True
    dcfg, mcfg, acfg, tcfg, kcfg = cfg["data"], cfg["model"], cfg["aug"], cfg["train"], cfg["kd"]

    # ------------------------------------------------------------------ data
    x, y, _, _, meta = load_cache(dcfg["cache_dir"])
    train_tf, test_tf = build_transforms(acfg, meta["mean"], meta["std"])
    if dcfg["full_train"]:
        train_idx, val_idx = list(range(len(y))), None
    else:
        train_idx, val_idx = stratified_split(y, dcfg["val_size"], cfg["seed"])
    nw = dcfg["num_workers"]
    loader_kw = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    train_loader = DataLoader(ArrayDataset(x[train_idx], y[train_idx], train_tf),
                              batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, **loader_kw)
    val_loader = None
    if val_idx is not None:
        val_loader = DataLoader(ArrayDataset(x[val_idx], y[val_idx], test_tf),
                                batch_size=512, shuffle=False, **loader_kw)

    # ------------------------------------------------------------------ model
    model = build_model(mcfg).to(device).to(memory_format=torch.channels_last)
    n_params = count_trainable_params(model)
    print(f"trainable parameters: {n_params:,}")
    limit = mcfg.get("param_limit")
    if limit is not None:
        assert n_params <= limit, f"model has {n_params:,} trainable parameters > limit {limit:,}"
    ema = EMA(model, tcfg["ema_decay"]) if tcfg.get("ema_decay", 0) > 0 else None

    teacher = None
    if kcfg.get("teacher_ckpt"):
        ck = torch.load(kcfg["teacher_ckpt"], map_location="cpu", weights_only=False)
        teacher = build_model(ck["cfg"]["model"])
        teacher.load_state_dict(ck["ema"] if ck.get("ema") is not None else ck["model"])
        print(f"teacher loaded from {kcfg['teacher_ckpt']} ({count_trainable_params(teacher):,} params)")
        teacher = teacher.to(device).to(memory_format=torch.channels_last).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ optimisation
    opt = build_optimizer(model, tcfg)
    steps_per_epoch = len(train_loader)
    total_steps = tcfg["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(total_steps, tcfg["warmup_epochs"] * steps_per_epoch))
    use_amp = tcfg.get("amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    # torch >= 2.3: torch.amp.GradScaler("cuda"); older versions: torch.cuda.amp.GradScaler
    _GradScaler = (lambda **kw: torch.amp.GradScaler("cuda", **kw)) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler
    scaler = _GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    start_epoch, best_acc = 0, 0.0
    last_path, best_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if ema is not None:
            ema.module.load_state_dict(ck["ema"])
            ema.updates = ck.get("ema_updates", 0)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc = ck["epoch"] + 1, ck.get("best_acc", 0.0)
        print(f"resumed from epoch {start_epoch}")

    logger = CSVLogger(os.path.join(run_dir, "log.csv"),
                       ["epoch", "lr", "train_loss", "train_acc", "val_acc", "val_acc_ema", "time_s"],
                       append=args.resume)

    # ------------------------------------------------------------------ training loop
    num_classes = mcfg.get("num_classes", 10)
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            target = smooth_one_hot(yb, num_classes, tcfg.get("label_smoothing", 0.0))
            if acfg.get("cutmix_prob", 0) > 0 and torch.rand(1).item() < acfg["cutmix_prob"]:
                xb, target = cutmix(xb, target, acfg.get("cutmix_alpha", 1.0))

            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                loss = soft_cross_entropy(logits, target)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(xb)       # teacher sees the same augmented batch
                    a = kcfg["alpha"]
                    loss = (1 - a) * loss + a * kd_loss(logits, t_logits, kcfg["temperature"])

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)

            loss_sum += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == target.argmax(1)).sum().item()
            seen += xb.size(0)

        row = {"epoch": epoch + 1, "lr": f"{opt.param_groups[0]['lr']:.5f}",
               "train_loss": f"{loss_sum / seen:.4f}", "train_acc": f"{correct / seen:.4f}",
               "val_acc": "", "val_acc_ema": "", "time_s": f"{time.time() - t0:.1f}"}
        score = None
        if val_loader is not None:
            row["val_acc"] = f"{accuracy(model, val_loader, device):.4f}"
            score = float(row["val_acc"])
            if ema is not None:
                row["val_acc_ema"] = f"{accuracy(ema.module, val_loader, device):.4f}"
                score = float(row["val_acc_ema"])
        logger.log(row)
        print("  ".join(f"{k}={v}" for k, v in row.items() if v != ""), flush=True)

        ck = {"model": model.state_dict(), "ema": ema.module.state_dict() if ema else None,
              "ema_updates": ema.updates if ema else 0,
              "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
              "epoch": epoch, "best_acc": best_acc, "cfg": cfg, "n_params": n_params}
        if score is not None and score > best_acc:
            best_acc = ck["best_acc"] = score
            torch.save(ck, best_path)
        torch.save(ck, last_path)

    if val_loader is not None:
        print(f"best validation accuracy: {best_acc:.4f}")
    print(f"done. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
