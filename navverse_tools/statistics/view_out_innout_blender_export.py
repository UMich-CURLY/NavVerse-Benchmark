#!/usr/bin/env python3
"""
在 Blender 内运行：导入 USD，
- 红点 object：从 /World/Objaverse 和 /World/RoadObjects 下收集（二者在 USD 中未必是 Xform，Blender 中仍按子物体取位置）
- 黄点 storefront：只取 /World/store 的**直接子 xform**，每个子 xform 一个点，不再递归到内部物体
用法（在终端）:
  blender --background --python view_out_innout_blender_export.py -- --city AMSTERDAM
  blender --background --python view_out_innout_blender_export.py -- --city AMSTERDAM --output /path/to/positions.json
"""
import json
import os
import sys

# Blender 传入的 argv 中，'--' 之后才是本脚本参数
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1 :]
else:
    argv = []

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Blender: 从 USD 导出 object/storefront 位置到 JSON")
    p.add_argument("--city", type=str, default="AMSTERDAM", help="城市名，如 AMSTERDAM")
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p.add_argument("--vc-plus-dir", type=str,
                   default=os.environ.get("VC_PLUS_DIR", os.path.join(_REPO_ROOT, "navverse_data", "vc_apr14")),
                   help="vc_plus 根目录（或设置 $VC_PLUS_DIR）")
    p.add_argument("--output", "-o", type=str, default=None, help="输出 JSON 路径，默认与脚本同目录")
    return p.parse_args(argv)

# USD 中：红点 object 来自这两个 Xform
OBJAVERSE_XFORM_NAME = "Objaverse"
ROAD_OBJECTS_XFORM_NAME = "RoadObjects"
# 黄点 storefront 来自 /World/store（不在这两个 xform 下）
STORE_XFORM_NAME = "store"


def get_all_descendants(obj):
    """递归收集某物体下的所有子物体（含孙辈）。"""
    out = []
    for child in obj.children:
        out.append(child)
        out.extend(get_all_descendants(child))
    return out


def _is_objaverse_or_road(obj):
    """对象是否对应 /World/Objaverse 或 /World/RoadObjects（Blender 可能用 prim 名或带路径）。"""
    n = obj.name
    return (
        n == OBJAVERSE_XFORM_NAME or n == ROAD_OBJECTS_XFORM_NAME
        or n.startswith(OBJAVERSE_XFORM_NAME) or n.startswith(ROAD_OBJECTS_XFORM_NAME)
        or OBJAVERSE_XFORM_NAME in n or ROAD_OBJECTS_XFORM_NAME in n
    )


def _is_store(obj):
    """对象是否对应 /World/store。"""
    n = obj.name
    return n == STORE_XFORM_NAME or n.startswith(STORE_XFORM_NAME) or ("/" + STORE_XFORM_NAME) in n or n.endswith("/" + STORE_XFORM_NAME)


def _get_world_children(bpy):
    """获取 /World 下的子物体：先找名为 World 的对象，否则用所有无父级对象。"""
    world = bpy.data.objects.get("World")
    if world is not None and world.children:
        return list(world.children)
    # 无 World 对象时，/World 的子节点可能是顶层对象（parent 为 None）
    return [o for o in bpy.data.objects if o.parent is None]


def main():
    args = parse_args()
    city = args.city.strip().upper()
    usd_path = os.path.join(os.path.abspath(args.vc_plus_dir), city, f"{city}_merged.usd")

    if not os.path.isfile(usd_path):
        print(f"USD 文件不存在: {usd_path}", file=sys.stderr)
        return 1

    import bpy

    # 清空当前场景（删除默认物体）
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 导入 USD
    try:
        bpy.ops.wm.usd_import(filepath=usd_path)
    except Exception as e:
        print(f"Blender USD 导入失败: {e}", file=sys.stderr)
        return 1

    # 按层级取 /World 的子节点（Blender 可能是 World 的 children 或顶层对象）
    world_children = _get_world_children(bpy)

    # 诊断：打印 /World 下各 xform 名称及子物体数量
    print("DEBUG /World 下对象及子物体数:", file=sys.stderr)
    for o in world_children:
        n_desc = len(get_all_descendants(o))
        print(f"  {o.name!r} -> {n_desc} 个后代", file=sys.stderr)

    # 红点 object：仅从 /World/Objaverse 和 /World/RoadObjects 下选取
    object_positions = []
    for obj in world_children:
        if _is_objaverse_or_road(obj):
            for child in get_all_descendants(obj):
                loc = child.matrix_world.translation
                object_positions.append([float(loc.x), float(loc.y)])

    # 黄点 storefront：只取 store 的**直接子 xform**，每个子 xform 一个点（不递归内部物体）
    storefront_positions = []
    for obj in world_children:
        if _is_store(obj):
            for child in obj.children:
                loc = child.matrix_world.translation
                storefront_positions.append([float(loc.x), float(loc.y)])

    data = {
        "city": city,
        "object_positions": object_positions,
        "storefront_positions": storefront_positions,
    }

    out_path = args.output
    if out_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, f"usd_positions_{city}.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"已导出: {out_path} (object: {len(object_positions)}, storefront: {len(storefront_positions)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
