import argparse
from pathlib import Path

from PIL import Image
import numpy as np
from torchvision import datasets

from pad_chars74K import pad_to_multiple

def export(ds, out, multiple, band, q):
    for d in range(10):
        (out / str(d)).mkdir(parents=True, exist_ok=True)
    for i, (img, label) in enumerate(ds):
        x = pad_to_multiple(np.asarray(img), multiple, band, q)
        Image.fromarray(np.ascontiguousarray(x)).save(out / str(label) / f"{i:05d}.png")
    return len(ds), x.shape

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--band", type=int, default=2, help="width of the edge strip used for the padding color")
    ap.add_argument("--q", type=int, default=16, help="quantization step per channel")
    ap.add_argument("--multiple", type=int, default=10, help="output side length is rounded up to a multiple of this")
    args = ap.parse_args()

    root = Path(args.data_dir)
    for split, train in [("train", True), ("test", False)]:
        ds = datasets.MNIST(root / "MNIST", train=train)
        n, shape = export(ds, root / "mnist_pad" / split, args.multiple, args.band, args.q)
        print(f"MNIST {split}: {n} images {shape}")
    for split in ["train", "test"]:
        ds = datasets.SVHN(root / "SVHN", split=split)
        n, shape = export(ds, root / "svhn_pad" / split, args.multiple, args.band, args.q)
        print(f"SVHN {split}: {n} images {shape}")

if __name__ == "__main__":
    main()
