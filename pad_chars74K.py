import argparse
from pathlib import Path

from PIL import Image
import numpy as np

def edge_mode_color(strip, q=16):
    bins = (strip // q).astype(np.int32)
    vals, inv, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    return strip[inv.ravel() == counts.argmax()].mean(axis=0).round().astype(np.uint8)

def pad_axis(img, d, axis, band=2, q=16):
    if d == 0:
        return img
    a, b = d // 2, d - (d // 2)
    c = img.shape[2]
    head = np.take(img, range(band), axis=axis).reshape(-1, c)
    tail = np.take(img, range(img.shape[axis] - band, img.shape[axis]), axis=axis).reshape(-1, c)
    shape_a, shape_b = list(img.shape), list(img.shape)
    shape_a[axis], shape_b[axis] = a, b
    pad_a = np.broadcast_to(edge_mode_color(head, q), shape_a)
    pad_b = np.broadcast_to(edge_mode_color(tail, q), shape_b)
    return np.concatenate([pad_a, img, pad_b], axis=axis)

def pad_to_multiple(img, multiple=10, band=2, q=16):
    gray = img.ndim == 2
    if gray:
        img = img[:, :, None]
    h, w, _ = img.shape
    side = -(-max(h, w) // multiple) * multiple
    img = pad_axis(img, side - w, axis=1, band=band, q=q)
    img = pad_axis(img, side - h, axis=0, band=band, q=q)
    return img[:, :, 0] if gray else img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/Chars74K/English/Img/GoodImg/Bmp")
    ap.add_argument("--out_dir", default="data/chars74k_sq")
    ap.add_argument("--band", type=int, default=2, help="width of the edge strip used for the padding color")
    ap.add_argument("--q", type=int, default=16, help="quantization step per channel")
    ap.add_argument("--multiple", type=int, default=10, help="pad to a multiple of this number")
    args = ap.parse_args()

    out = Path(args.out_dir)
    n = 0
    for digit in range(10):
        src = Path(args.src_dir) / f"Sample{digit + 1:03d}" # Sample001 = 0
        (out / str(digit)).mkdir(parents=True, exist_ok=True)
        for fp in sorted(src.glob("*.png")):
            img = np.asarray(Image.open(fp).convert("RGB"))
            sq = pad_to_multiple(img, args.multiple, args.band, args.q)
            Image.fromarray(np.ascontiguousarray(sq)).save(out / str(digit) / fp.name)
            n += 1
    print(f"images: {n}")

if __name__ == "__main__":
    main()
