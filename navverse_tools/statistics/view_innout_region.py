#!/usr/bin/env python3
"""
以室内场景为中心的 100m 正方形区域 BEV 图：读取 {city}_innout.usd 得到室内位置，
在该位置取边长 100m 的正方形，显示范围内的 object/storefront，并用红色五角星标出室内位置。

依赖：与 view_out_innout.py 相同（BEV、positions JSON、可选 pxr 读 object）。
用法:
  python view_innout_region.py --city AMSTERDAM
  python view_innout_region.py --city AMSTERDAM --positions-json /path/to/usd_positions_AMSTERDAM.json -o out.png
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_ENV_DIR = os.path.dirname(SCRIPT_DIR)
if ROBOT_ENV_DIR not in sys.path:
    sys.path.insert(0, ROBOT_ENV_DIR)

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.astar import load_bev_map, generate_obstacle_map

DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "episode_data")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_VC_PLUS_DIR = os.environ.get("VC_PLUS_DIR", os.path.join(_REPO_ROOT, "navverse_data", "vc_apr14"))
SQUARE_SIDE_M = 200.0  # 以室内为中心的正方形边长（米）


def world_to_pixel(world_xy, world_center, px_per_meter, image_width, image_height):
    wp = (np.array(world_xy)[:2] - world_center[:2]) * px_per_meter
    px = wp[0] + image_width // 2
    py = image_height // 2 - wp[1]
    return (px, py)


# start_result_navigation 是一个区域（室内场景），以该区域的中心位置为坐标。
# 候选 prim 路径（与 navmesh_innout / merge_usd_innout 一致，不同 stage 可能在不同层级）
INDOOR_PRIM_PATHS = [
    "/World/ground/terrain/start_result_navigation",
    "/World/start_result_navigation",
    "/start_result_navigation",
]


def _get_indoor_position_from_prim(stage, prim_path):
    """
    室内位置 = start_result_navigation 这个 Xform 本身的 translate。
    取该 prim（Xform）的世界变换平移 (x, y) 作为五角星位置；若非 Xformable 则用 bbox 中心。
    """
    try:
        from pxr import Usd, UsdGeom
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            return None
        if prim.IsA(UsdGeom.Xformable):
            xform = UsdGeom.Xformable(prim)
            world_xform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = world_xform.ExtractTranslation()
            return (float(t[0]), float(t[1]))
        # 非 Xformable 时用 bbox 中心
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
        bound = bbox_cache.ComputeWorldBound(prim)
        range3d = bound.ComputeAlignedBox()
        min_pt = range3d.GetMin()
        max_pt = range3d.GetMax()
        center = (min_pt + max_pt) / 2.0
        return (float(center[0]), float(center[1]))
    except Exception:
        return None


def load_indoor_position_from_innout_usd(innout_usd_path, indoor_prim_path=None):
    """
    仅从 {city}_innout.usd 读取室内场景位置：用 start_result_navigation 这个 Xform 本身的 translate
    （世界变换平移）作为五角星位置。
    """
    try:
        from pxr import Usd
    except ImportError:
        return None
    if not os.path.isfile(innout_usd_path):
        return None
    try:
        stage = Usd.Stage.Open(innout_usd_path)
        paths_to_try = [indoor_prim_path] if indoor_prim_path else INDOOR_PRIM_PATHS
        for prim_path in paths_to_try:
            xy = _get_indoor_position_from_prim(stage, prim_path)
            if xy is not None:
                return (xy, prim_path)
        return None
    except Exception:
        return None


def load_positions_from_json(json_path):
    if not os.path.isfile(json_path):
        return None, None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        objects_xy = [tuple(p) for p in data.get("object_positions", [])]
        storefronts_xy = [tuple(p) for p in data.get("storefront_positions", [])]
        return objects_xy, storefronts_xy
    except Exception as e:
        print(f"读取位置 JSON 失败: {e}")
        return None, None


def load_object_positions_from_usd(usd_path):
    object_positions = []
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return object_positions
    if not os.path.isfile(usd_path):
        return object_positions
    try:
        stage = Usd.Stage.Open(usd_path)
        for parent_path in ("/World/RoadObjects", "/World/Objaverse"):
            parent = stage.GetPrimAtPath(parent_path)
            if not parent:
                continue
            for prim in Usd.PrimRange(parent):
                if prim == parent:
                    continue
                if not prim.IsA(UsdGeom.Xformable):
                    continue
                xformable = UsdGeom.Xformable(prim)
                world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                t = world_transform.ExtractTranslation()
                object_positions.append((t[0], t[1]))
    except Exception:
        pass
    return object_positions


def main():
    parser = argparse.ArgumentParser(description="以室内为中心的 100m 正方形 BEV，红五角星标室内位置")
    parser.add_argument("--city", type=str, default="AMSTERDAM")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--positions-json", type=str, default=None)
    parser.add_argument("--vc-plus-dir", type=str, default=DEFAULT_VC_PLUS_DIR)
    parser.add_argument("--scale-positions", type=float, default=100.0)
    parser.add_argument("--indoor-scale", type=float, default=1.0, help="USD 室内坐标单位：1=米（默认），100=厘米（需除以 100 转为米）")
    parser.add_argument("--indoor-prim", type=str, default=None, help="室内 prim 路径，默认按序尝试: /World/ground/terrain/start_result_navigation, /World/start_result_navigation, /start_result_navigation")
    parser.add_argument("--list-prims", action="store_true", help="仅列出 innout USD 中 /World 下子 prim 路径后退出，用于核对室内 prim")
    args = parser.parse_args()

    city = args.city.strip().upper()
    scene_name = f"vc_{city.lower()}_store"
    vc_plus_dir = os.path.abspath(args.vc_plus_dir)
    innout_usd = os.path.join(vc_plus_dir, city, f"{city}_innout.usd")

    # 可选：只列出 /World 下 prim，便于核对室内场景是哪一个
    if getattr(args, "list_prims", False):
        try:
            from pxr import Usd
            stage = Usd.Stage.Open(innout_usd)
            world = stage.GetPrimAtPath("/World")
            if world:
                def walk(p, prefix=""):
                    for c in p.GetChildren():
                        path = f"{prefix}/{c.GetName()}"
                        print(path)
                        walk(c, path)
                walk(world, "/World")
            else:
                print("/World 不存在")
        except Exception as e:
            print(f"列出 prim 失败: {e}")
        return 0

    # 室内位置（世界坐标 x, y）：仅读 {city}_innout.usd，没有则跳过
    indoor_result = load_indoor_position_from_innout_usd(innout_usd, indoor_prim_path=args.indoor_prim)
    if indoor_result is None:
        tried = args.indoor_prim or " | ".join(INDOOR_PRIM_PATHS)
        print(f"无法从 {innout_usd} 读取室内位置（已尝试: {tried}）")
        return 1
    indoor_xy_raw, indoor_prim_used = indoor_result
    print(f"室内位置 = start_result_navigation Xform translate，prim: {indoor_prim_used}  raw(USD): ({indoor_xy_raw[0]:.2f}, {indoor_xy_raw[1]:.2f})")
    # indoor_scale：1=坐标已是米，100=坐标是厘米需除以 100 转为米
    indoor_scale = float(args.indoor_scale)
    indoor_xy = (indoor_xy_raw[0] / indoor_scale, indoor_xy_raw[1] / indoor_scale)
    # BEV
    data_dir = os.path.abspath(args.data_dir)
    bev_map_path = os.path.join(data_dir, scene_name, "bev_map.npz")
    if not os.path.isfile(bev_map_path):
        print(f"BEV 文件不存在: {bev_map_path}")
        return 1

    bev_depth, info = load_bev_map(bev_map_path)
    image_height, image_width = bev_depth.shape
    obstacle_map = generate_obstacle_map(bev_depth)
    px_per_meter = np.array(info["px_per_meter"])
    world_center = np.array(info["world_center"])
    px_per_meter_scalar = float(np.mean(px_per_meter))

    def _world_to_pixel(world_xy):
        return world_to_pixel(world_xy, world_center, px_per_meter, image_width, image_height)

    # 100m 正方形在像素下的边长与中心
    side_px = int(SQUARE_SIDE_M * px_per_meter_scalar)
    indoor_px = _world_to_pixel(indoor_xy)
    pcx, pcy = indoor_px[0], indoor_px[1]
    x0 = int(pcx - side_px / 2)
    y0 = int(pcy - side_px / 2)
    x0 = max(0, min(x0, image_width - side_px))
    y0 = max(0, min(y0, image_height - side_px))
    # 保证正方形在图像内，可能略微缩小
    side_px = min(side_px, image_width - x0, image_height - y0)
    if side_px <= 0:
        print("室内位置对应的 100m 正方形与 BEV 无交集")
        return 1
    # 实际显示边长（米），用于核对
    side_m_actual = side_px / px_per_meter_scalar
    print(f"室内位置(米): ({indoor_xy[0]:.2f}, {indoor_xy[1]:.2f})  px_per_meter: {px_per_meter_scalar:.1f}  正方形: {side_px}px = {side_m_actual:.1f}m (目标 {SQUARE_SIDE_M}m)")

    # 底图
    bev_data = np.load(bev_map_path)
    if "rgb" in bev_data:
        bg_rgb = bev_data["rgb"]
        if bg_rgb.ndim == 3 and bg_rgb.shape[-1] >= 3:
            bg_rgb = bg_rgb[..., :3].copy()
            if bg_rgb.max() > 1.0:
                bg_rgb = bg_rgb.astype(np.float32) / 255.0
        else:
            bg_rgb = None
    else:
        bg_rgb = None
    if bg_rgb is None:
        rgb_path = bev_map_path.replace(".npz", "_rgb.png")
        if os.path.isfile(rgb_path):
            import cv2
            bg_rgb = cv2.imread(rgb_path)
            if bg_rgb is not None:
                bg_rgb = cv2.cvtColor(bg_rgb, cv2.COLOR_BGR2RGB)
                if bg_rgb.max() > 1.0:
                    bg_rgb = bg_rgb.astype(np.float32) / 255.0
    if bg_rgb is None:
        bg_rgb = np.stack([obstacle_map.astype(np.float32)] * 3, axis=-1)
        bg_rgb[obstacle_map == 0] = [0.85, 0.90, 0.95]
        bg_rgb[obstacle_map == 1] = [0.35, 0.40, 0.50]

    # 裁剪为 100m 正方形区域
    bg_rgb = bg_rgb[y0 : y0 + side_px, x0 : x0 + side_px]

    # Object / storefront 位置（与 view_out_innout 一致：JSON + 可选 pxr 回退）
    positions_json = args.positions_json
    if positions_json is None:
        positions_json = os.path.join(SCRIPT_DIR, f"usd_positions_{city}.json")
    positions_json = os.path.abspath(positions_json)
    object_positions_world, storefront_positions_world = load_positions_from_json(positions_json)
    if object_positions_world is None:
        object_positions_world = []
    if storefront_positions_world is None:
        storefront_positions_world = []
    object_from_pxr = False
    if not object_positions_world:
        merged_usd = os.path.join(vc_plus_dir, city, f"{city}_merged.usd")
        object_positions_world = load_object_positions_from_usd(merged_usd)
        object_from_pxr = bool(object_positions_world)

    scale = float(args.scale_positions)
    scale_object = 1.0 if object_from_pxr else scale
    object_px = [_world_to_pixel((x * scale_object, y * scale_object)) for x, y in object_positions_world]
    storefront_px = [_world_to_pixel((x * scale, y * scale)) for x, y in storefront_positions_world]
    # 只保留在裁剪区域内的点，并转为相对裁剪区域的坐标
    object_px = [(px - x0, py - y0) for px, py in object_px if x0 <= px < x0 + side_px and y0 <= py < y0 + side_px]
    storefront_px = [(px - x0, py - y0) for px, py in storefront_px if x0 <= px < x0 + side_px and y0 <= py < y0 + side_px]
    indoor_in_crop = (pcx - x0, pcy - y0)

    # 绘图
    TARGET_SIZE = 960
    out_dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=(TARGET_SIZE / out_dpi, TARGET_SIZE / out_dpi), dpi=out_dpi)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(bg_rgb, aspect="auto", extent=(0, side_px, side_px, 0), interpolation="nearest")
    ax.set_xlim(0, side_px)
    ax.set_ylim(side_px, 0)

    pt_scale = (TARGET_SIZE / 960.0) ** 2
    if storefront_px:
        ax.scatter([p[0] for p in storefront_px], [p[1] for p in storefront_px], c="yellow", s=60 * pt_scale, alpha=0.9, zorder=2)
    if object_px:
        ax.scatter([p[0] for p in object_px], [p[1] for p in object_px], c="blue", s=60 * pt_scale, alpha=0.95, zorder=3, edgecolors="deepskyblue", linewidths=0.5)
    ax.scatter([indoor_in_crop[0]], [indoor_in_crop[1]], c="red", s=2000 * pt_scale, marker="*", zorder=4, edgecolors="darkred", linewidths=0.5)

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(SCRIPT_DIR, f"view_innout_region_{scene_name}.png")
    plt.savefig(out_path, dpi=out_dpi, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"已保存: {out_path} (室内中心 100m 正方形，红五角星=室内位置)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
