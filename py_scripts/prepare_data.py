"""Convert the Kaggle competition files into numpy arrays (run once).

Expected input (the extracted Kaggle "Download All" zip):
    data/raw/train_images/train_00000.png ...
    data/raw/test_images/test_0000.png ...
    data/raw/train_labels.csv   (id, filename, label)
    data/raw/test.csv           (id, filename)
    data/raw/sample_submission.csv

Output (data/cache/):
    train_x.npy (50000, 32, 32, 3) uint8, train_y.npy (50000,) int64,
    test_x.npy (N, 32, 32, 3) uint8, test_ids.npy (N,) int64 in sample_submission order,
    meta.json  (per-channel mean / std of the training images)

Usage:  python prepare_data.py --raw data/raw --out data/cache
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_images(folder, filenames):
    arr = np.empty((len(filenames), 32, 32, 3), dtype=np.uint8)
    for i, fn in enumerate(filenames):
        img = Image.open(os.path.join(folder, fn)).convert("RGB")
        assert img.size == (32, 32), f"{fn}: unexpected size {img.size}"
        arr[i] = np.asarray(img)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/cache")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- training set
    rows = sorted(read_csv(os.path.join(args.raw, "train_labels.csv")), key=lambda r: int(r["id"]))
    train_x = load_images(os.path.join(args.raw, "train_images"), [r["filename"] for r in rows])
    train_y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    assert train_y.min() >= 0 and train_y.max() <= 9

    # ---- test set, ordered exactly like sample_submission.csv
    test_rows = {int(r["id"]): r["filename"] for r in read_csv(os.path.join(args.raw, "test.csv"))}
    sub_ids = [int(r["id"]) for r in read_csv(os.path.join(args.raw, "sample_submission.csv"))]
    assert set(sub_ids) == set(test_rows), "test.csv ids do not match sample_submission.csv ids"
    test_x = load_images(os.path.join(args.raw, "test_images"), [test_rows[i] for i in sub_ids])
    test_ids = np.array(sub_ids, dtype=np.int64)

    # ---- normalisation statistics from the training images only
    xf = train_x.astype(np.float64) / 255.0
    mean = xf.mean(axis=(0, 1, 2)).tolist()
    std = xf.std(axis=(0, 1, 2)).tolist()

    np.save(os.path.join(args.out, "train_x.npy"), train_x)
    np.save(os.path.join(args.out, "train_y.npy"), train_y)
    np.save(os.path.join(args.out, "test_x.npy"), test_x)
    np.save(os.path.join(args.out, "test_ids.npy"), test_ids)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)

    print(f"train: {train_x.shape}, class counts {np.bincount(train_y).tolist()}")
    print(f"test : {test_x.shape}")
    print(f"mean {np.round(mean, 4).tolist()}  std {np.round(std, 4).tolist()}")


if __name__ == "__main__":
    main()
