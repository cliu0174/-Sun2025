"""Create a labelled contact sheet from rendered PDF page PNGs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("page_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=280)
    args = parser.parse_args()

    pages = sorted(args.page_dir.glob("page-*.png"))
    if not pages:
        raise SystemExit(f"No page PNGs found in {args.page_dir}")
    with Image.open(pages[0]) as first:
        thumb_height = round(args.thumb_width * first.height / first.width)

    margin = 18
    label_height = 26
    rows = math.ceil(len(pages) / args.columns)
    sheet = Image.new(
        "RGB",
        (
            margin + args.columns * (args.thumb_width + margin),
            margin + rows * (thumb_height + label_height + margin),
        ),
        "#D9DDE3",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)

    for index, path in enumerate(pages):
        row, column = divmod(index, args.columns)
        x = margin + column * (args.thumb_width + margin)
        y = margin + row * (thumb_height + label_height + margin)
        with Image.open(path) as page:
            thumb = page.convert("RGB").resize(
                (args.thumb_width, thumb_height), Image.Resampling.LANCZOS
            )
        sheet.paste(thumb, (x, y + label_height))
        draw.text((x, y), f"Page {index + 1}", fill="#111827", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, dpi=(150, 150))


if __name__ == "__main__":
    main()
