"""Generate submission.csv from a trained checkpoint.

    python predict.py --ckpt runs/kd_full/last.pt                   # EMA weights + flip TTA (default)
    python predict.py --ckpt runs/main/best.pt --val                # also report accuracy on the 5k val split
    python predict.py --ckpt runs/main/last.pt --weights model --no-tta

The rows follow the id order of sample_submission.csv (prepare_data.py stores test_ids in that order).
"""
import argparse
import csv

import torch
from torch.utils.data import DataLoader

from src.data import ArrayDataset, build_transforms, load_cache, stratified_split
from src.model import build_model, count_trainable_params
from src.utils import accuracy, get_device, predict_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--no-tta", action="store_true", help="disable horizontal-flip test-time augmentation")
    ap.add_argument("--val", action="store_true", help="also evaluate on the validation split used in training")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg["model"])
    state = ck["ema"] if args.weights == "ema" and ck.get("ema") is not None else ck["model"]
    model.load_state_dict(state)
    print(f"trainable parameters: {count_trainable_params(model):,}")
    model = model.to(device).to(memory_format=torch.channels_last).eval()
    tta = not args.no_tta

    x, y, test_x, test_ids, meta = load_cache(cfg["data"]["cache_dir"])
    _, test_tf = build_transforms(cfg["aug"], meta["mean"], meta["std"])

    if args.val:
        if cfg["data"]["full_train"]:
            print("checkpoint was trained on all 50k images; no held-out validation split")
        else:
            _, val_idx = stratified_split(y, cfg["data"]["val_size"], cfg["seed"])
            vl = DataLoader(ArrayDataset(x[val_idx], y[val_idx], test_tf), batch_size=512)
            print(f"val acc (no TTA):   {accuracy(model, vl, device, tta_flip=False):.4f}")
            print(f"val acc (flip TTA): {accuracy(model, vl, device, tta_flip=True):.4f}")

    tl = DataLoader(ArrayDataset(test_x, None, test_tf), batch_size=512)
    probs, _ = predict_logits(model, tl, device, tta_flip=tta)
    preds = probs.argmax(1).tolist()

    assert len(preds) == len(test_ids)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(test_ids.tolist(), preds):
            w.writerow([i, int(p)])
    print(f"wrote {len(preds)} predictions to {args.out}; "
          f"predicted class counts {torch.bincount(torch.tensor(preds), minlength=10).tolist()}")


if __name__ == "__main__":
    main()
