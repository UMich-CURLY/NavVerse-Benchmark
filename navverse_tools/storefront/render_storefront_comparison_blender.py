"""Render original/repaired storefront pairs and build a comparison sheet."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from render_repaired_contact_sheet_blender import render_storefront


def parse_args() -> argparse.Namespace:
    argv = (
        sys.argv[sys.argv.index("--") + 1 :]
        if "--" in sys.argv
        else sys.argv[1:]
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--contact-only", action="store_true")
    return parser.parse_args(argv)


def make_comparison_sheet(
    pairs: list[tuple[str, Path, Path, int]],
    destination: Path,
    columns: int,
) -> None:
    view_width = 400
    view_height = 300
    label_height = 36
    title_height = 60
    tile_width = 2 * view_width
    tile_height = view_height + label_height
    rows = math.ceil(len(pairs) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, title_height + rows * tile_height),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    small_font = ImageFont.truetype(font_path, 16)
    title_font = ImageFont.truetype(font_path, 26)
    draw.text(
        (18, 13),
        "Storefront mesh comparison — BEFORE (left) / REPAIRED (right), +Y front",
        fill=(242, 242, 242),
        font=title_font,
    )

    for index, (name, before_path, after_path, vertex_count) in enumerate(pairs):
        column = index % columns
        row = index // columns
        left = column * tile_width
        top = title_height + row * tile_height
        for offset, path in ((0, before_path), (view_width, after_path)):
            with Image.open(path) as source:
                rendered = source.convert("RGB").resize(
                    (view_width, view_height), Image.Resampling.LANCZOS
                )
            sheet.paste(rendered, (left + offset, top))
        draw.rectangle((left + 8, top + 8, left + 105, top + 35), fill=(135, 42, 42))
        draw.text((left + 17, top + 11), "BEFORE", fill="white", font=small_font)
        draw.rectangle(
            (left + view_width + 8, top + 8, left + view_width + 118, top + 35),
            fill=(31, 115, 69),
        )
        draw.text(
            (left + view_width + 17, top + 11),
            "REPAIRED",
            fill="white",
            font=small_font,
        )
        is_placeholder = vertex_count <= 4
        if is_placeholder:
            draw.rectangle(
                (left + 2, top + 2, left + tile_width - 2, top + view_height - 2),
                outline=(235, 55, 55),
                width=6,
            )
        draw.rectangle(
            (left, top + view_height, left + tile_width, top + tile_height),
            fill=(10, 10, 10),
        )
        label = f"{name}{' — SOURCE PLACEHOLDER' if is_placeholder else ''}"
        draw.text(
            (left + 8, top + view_height + 7),
            label,
            fill=(255, 90, 90) if is_placeholder else (238, 238, 238),
            font=font,
        )
    sheet.save(destination, compress_level=4)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    originals = sorted(
        path
        for path in root.glob("*/*.usd")
        if not path.stem.endswith(".repaired")
    )
    report = json.loads((root / "storefront_mesh_repair_report.json").read_text())
    vertex_counts = {
        Path(item["input"]).stem: sum(
            mesh["input_vertices"] for mesh in item.get("meshes", [])
        )
        for item in report
    }

    pairs = []
    for index, original in enumerate(originals, start=1):
        name = original.stem
        repaired = original.with_name(f"{name}.repaired.usd")
        before_path = output_dir / f"{name}.before.png"
        after_path = output_dir / f"{name}.repaired.png"
        if args.contact_only:
            if not before_path.is_file() or not after_path.is_file():
                raise FileNotFoundError(name)
        else:
            print(f"[{index}/{len(originals)}] {name}: before", flush=True)
            render_storefront(original, before_path, args.width, args.height)
            print(f"[{index}/{len(originals)}] {name}: repaired", flush=True)
            render_storefront(repaired, after_path, args.width, args.height)
        pairs.append((name, before_path, after_path, vertex_counts[name]))

    destination = output_dir / "storefront_before_after_contact_sheet.png"
    make_comparison_sheet(pairs, destination, args.columns)
    print(f"CONTACT_SHEET: {destination}", flush=True)


if __name__ == "__main__":
    main()
