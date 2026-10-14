import argparse
from pathlib import Path

from PIL import Image
import numpy as np

def edge_mode_color(strip, q=16): # strip: (N, 3) uint8。量子化して最頻ビンを選び、そのビン内の元色の平均を返す
    bins = (strip // q).astype(np.int32)
    keys = bins[:, 0] * 10000 + bins[:, 1] * 100 + bins[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[counts.argmax()]
    return strip[keys == top].mean(axis=0).round().astype(np.uint8)

def pad_to_square(img, band=2, q=16):
    h, w, _ = img.shape
    if h == w:
        return img
    d = abs(h - w)
    a, b = d // 2, d - (d // 2)
    if h > w:
        ca = edge_mode_color(img[:, :band].reshape(-1, 3), q)
        cb = edge_mode_color(img[:, -band:].reshape(-1, 3), q)
        left = np.broadcast_to(ca, (h, a, 3))
        right = np.broadcast_to(cb, (h, b, 3))
        return np.concatenate([left, img, right], axis=1)
    else:
        ca = edge_mode_color(img[:band].reshape(-1, 3), q)
        cb = edge_mode_color(img[-band:].reshape(-1, 3), q)
        top = np.broadcast_to(ca, (a, w, 3))
        bottom = np.broadcast_to(cb, (b, w, 3))
        return np.concatenate([top, img, bottom], axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/Chars74K/English/Img/GoodImg/Bmp")
    ap.add_argument("--out_dir", default="data/chars74k_sq")
    ap.add_argument("--band", type=int, default=2, help="width of the edge strip used for the padding color")
    ap.add_argument("--q", type=int, default=16, help="quantization step per channel")
    args = ap.parse_args()

    out = Path(args.out_dir)
    n = 0
    for digit in range(10):
        src = Path(args.src_dir) / f"Sample{digit + 1:03d}" # Sample001 = 0
        (out / str(digit)).mkdir(parents=True, exist_ok=True)
        for fp in sorted(src.glob("*.png")):
            img = np.asarray(Image.open(fp).convert("RGB"))
            sq = pad_to_square(img, args.band, args.q)
            Image.fromarray(np.ascontiguousarray(sq)).save(out / str(digit) / fp.name)
            n += 1
    print(f"images: {n}")

if __name__ == "__main__":
    main()
