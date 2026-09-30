"""Render repaired storefronts from the front-right and build a contact sheet."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont


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
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--contact-only", action="store_true")
    return parser.parse_args(argv)


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    lower = Vector((math.inf, math.inf, math.inf))
    upper = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            lower = Vector(map(min, lower, point))
            upper = Vector(map(max, upper, point))
    return lower, upper


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def render_storefront(
    usd_path: Path, output_path: Path, width: int, height: int
) -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.wm.usd_import(filepath=str(usd_path), import_materials=True)
    if result != {"FINISHED"}:
        raise RuntimeError(f"Could not import {usd_path}")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh in {usd_path}")

    lower, upper = world_bounds(meshes)
    center = (lower + upper) / 2
    size = upper - lower
    major_size = max(size.x, size.z)

    bpy.ops.object.camera_add(
        location=center
        + Vector((0.9 * major_size, 1.6 * major_size, 0.25 * size.z))
    )
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    projected_width = size.x + 0.55 * size.y
    camera.data.ortho_scale = 1.22 * max(
        size.z + 0.2 * size.y, projected_width * height / width
    )
    point_at(camera, center)
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.show_specular_highlight = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(output_path)
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Storefront QA World")
    scene.world.color = (0.035, 0.035, 0.035)
    scene.view_settings.look = "AgX - Medium High Contrast"
    bpy.ops.render.render(write_still=True)
    return sum(len(obj.data.vertices) for obj in meshes)


def make_contact_sheet(
    images: list[tuple[Path, int]],
    destination: Path,
    columns: int,
    width: int,
    height: int,
) -> None:
    label_height = 34
    title_height = 54
    rows = math.ceil(len(images) / columns)
    sheet = Image.new(
        "RGB",
        (columns * width, title_height + rows * (height + label_height)),
        (25, 25, 25),
    )
    draw = ImageDraw.Draw(sheet)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    title_font = ImageFont.truetype(font_path, 25)
    draw.text(
        (18, 12),
        "Repaired storefronts — front-right 3/4 view (+Y front); red = source placeholder",
        fill=(240, 240, 240),
        font=title_font,
    )
    for index, (image_path, vertex_count) in enumerate(images):
        column = index % columns
        row = index // columns
        left = column * width
        top = title_height + row * (height + label_height)
        with Image.open(image_path) as rendered:
            sheet.paste(rendered.convert("RGB"), (left, top))
        is_placeholder = vertex_count <= 4
        if is_placeholder:
            draw.rectangle(
                (left + 2, top + 2, left + width - 2, top + height - 2),
                outline=(230, 55, 55),
                width=6,
            )
        draw.rectangle(
            (left, top + height, left + width, top + height + label_height),
            fill=(12, 12, 12),
        )
        draw.text(
            (left + 8, top + height + 6),
            f"{image_path.stem}{' — SOURCE PLACEHOLDER' if is_placeholder else ''}",
            fill=(255, 90, 90) if is_placeholder else (238, 238, 238),
            font=font,
        )
    sheet.save(destination, optimize=True)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(root.glob("*/*.repaired.usd"))
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise SystemExit(f"No repaired USD files found below {root}")

    outputs = []
    report = json.loads((root / "storefront_mesh_repair_report.json").read_text())
    vertex_counts = {
        Path(item["input"]).stem: sum(
            mesh["input_vertices"] for mesh in item.get("meshes", [])
        )
        for item in report
    }
    for index, usd_path in enumerate(inputs, start=1):
        name = usd_path.name.removesuffix(".repaired.usd")
        output_path = output_dir / f"{name}.png"
        if args.contact_only:
            if not output_path.is_file():
                raise FileNotFoundError(output_path)
            vertex_count = vertex_counts[name]
        else:
            print(f"[{index}/{len(inputs)}] {name}", flush=True)
            vertex_count = render_storefront(
                usd_path, output_path, args.width, args.height
            )
        outputs.append((output_path, vertex_count))

    contact_path = output_dir / "storefront_repaired_contact_sheet.png"
    make_contact_sheet(outputs, contact_path, args.columns, args.width, args.height)
    print(f"CONTACT_SHEET: {contact_path}", flush=True)


if __name__ == "__main__":
    main()
