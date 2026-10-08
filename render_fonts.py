import argparse
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont, ImageOps

DIGITS = "0123456789"
RENDER = 160

def has_all_digits(font_path): # Path -> bool
    try:
        cmap = TTFont(font_path, fontNumber=0, lazy=True).getBestCmap() or {}
    except Exception:
        return False
    return all(ord(d) in cmap for d in DIGITS)


def render_digit(font, digit, size, fill): #(font: ImageFont.FreeTypeFont, digit: str, size: int, fill: float) -> Image.Image | None:
    big = Image.new("L", (RENDER * 2, RENDER *2), 255) #白い大きめの下書きキャンバス
    ImageDraw.Draw(big).text((RENDER // 2, RENDER // 2), digit, font=font, fill=0)
    bbox = ImageOps.invert(big).getbbox()
    if bbox is None:
        return None
    glyph = big.crop(bbox)

    box = max(1, round(size * fill))
    scale = box / max(glyph.size)
    w, h = max(1, round(glyph.width * scale)), max(1, round(glyph.height * scale))
    glyph = glyph.resize((w, h), Image.LANCZOS)

    canvas = Image.new("L", (size, size), 255)
    canvas.paste(glyph, ((size - w) // 2, (size - h) // 2))
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts_dir", default="fonts", help="path to the google/fonts repository (or any folder of .ttf files)")
    ap.add_argument("--out_dir", default="data/gfonts")
    ap.add_argument("--size", type=int, default=30, help="output image size in pixels (size x size)")
    ap.add_argument("--fill", type=float, default=0.8, help="fraction of the canvas taken by the digit's longer side")
    args = ap.parse_args()

    out = Path(args.out_dir)
    for d in DIGITS:
        (out / d).mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    for fp in sorted(Path(args.fonts_dir).rglob("*.ttf")):
        if not has_all_digits(fp):
            skipped += 1
            continue
        try:
            font = ImageFont.truetype(str(fp), RENDER)
            images = [render_digit(font, d, args.size, args.fill) for d in DIGITS]
        except Exception:
            skipped += 1
            continue
        if any(im is None for im in images):
            skipped += 1
            continue

        family = fp.parent.parent.name if fp.parent.name == "static" else fp.parent.name
        for d, im in zip(DIGITS, images):
            im.save(out / d / f"{family}__{fp.stem}.png")
        kept += 1

    print(f"fonts renderd: {kept}, skipped: {skipped}, images: {kept * 10} ({args.size}x{args.size})")

if __name__ == "__main__":
    main()
