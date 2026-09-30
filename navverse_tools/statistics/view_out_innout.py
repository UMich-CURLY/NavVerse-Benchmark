#!/usr/bin/env python3
"""
BEV 地图上叠加 object/storefront 位置（红点=object，黄点=storefront）。
位置数据由 Blender 脚本从 USD 导出为 JSON，本脚本只负责读 JSON 并画图。

步骤 1（在 Blender 中）:
  blender --background --python view_out_innout_blender_export.py -- --city AMSTERDAM

步骤 2（普通 Python，需 matplotlib / numpy / utils）:
  python view_out_innout.py --city AMSTERDAM
"""
import argparse
import json
import os
import sys

# 保证从任意目录运行时都能导入 utils（robot_env 包）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_ENV_DIR = os.path.dirname(SCRIPT_DIR)
if ROBOT_ENV_DIR not in sys.path:
    sys.path.insert(0, ROBOT_ENV_DIR)

import numpy as np

# 无界面时用 Agg 后端，便于 Blender 后台或 SSH 运行
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.astar import load_bev_map, generate_obstacle_map

# 默认路径（可按需通过参数覆盖）
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "episode_data")
DEFAULT_EPISODES_DIR = os.path.join(SCRIPT_DIR, "..", "episodes")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_VC_PLUS_DIR = os.environ.get("VC_PLUS_DIR", os.path.join(_REPO_ROOT, "navverse_data", "vc_apr14"))

STOREFRONT_NAMES = [
    "restaurant", "cafe", "fast_food", "parking_entrance", "bar", "bank", "pub", "atm",
    "bicycle_rental", "parking", "pharmacy", "toilets", "theatre", "library", "dentist",
    "school", "post_office", "bureau_de_change", "doctors", "bicycle_repair_station",
    "clinic", "community_centre", "car_rental", "police", "arts_centre", "cinema",
    "kindergarten", "university", "coworking_space", "college",
]


def world_to_pixel(world_xy, world_center, px_per_meter, image_width, image_height):
    """将世界坐标 (x, y) 转为 BEV 图像像素坐标。"""
    wp = (np.array(world_xy)[:2] - world_center[:2]) * px_per_meter
    px = wp[0] + image_width // 2
    py = image_height // 2 - wp[1]
    return (px, py)


def _content_square_crop(rgb, image_width, image_height):
    """
    检测填充色，找到「非填充」内容的边界框，在其内取最大内接正方形 (x0, y0, side)。
    只保留该正方形内像素，超出直线边界的全部裁掉。
    rgb: (H, W, 3) float [0,1] 或 [0,255]
    """
    h, w = rgb.shape[0], rgb.shape[1]
    scale = 1.0 if rgb.max() <= 1.0 else 255.0
    # 用四角 + 四条边的采样估计填充色
    border = []
    for y in [0, h - 1]:
        for x in range(0, w, max(1, w // 50)):
            border.append(rgb[y, x])
    for x in [0, w - 1]:
        for y in range(0, h, max(1, h // 50)):
            border.append(rgb[y, x])
    border = np.array(border)
    pad_color = np.median(border, axis=0)
    # 与填充色距离超过阈值的视为内容（非填充）
    diff = np.linalg.norm(rgb.reshape(-1, 3) - pad_color, axis=1)
    diff = diff.reshape(h, w)
    thresh = 0.06 if scale == 1.0 else 15.0
    content = (diff > thresh).astype(np.uint8)
    # 内容边界框
    ys, xs = np.where(content > 0)
    if ys.size == 0 or xs.size == 0:
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return x0, y0, side
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    cw = x_max - x_min + 1
    ch = y_max - y_min + 1
    # 在内容框内取最大内接正方形，以内容中心为正方形中心
    side = min(cw, ch)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    x0 = cx - side // 2
    y0 = cy - side // 2
    x0 = max(0, min(x0, w - side))
    y0 = max(0, min(y0, h - side))
    return x0, y0, side


# 室内场景五角星：从 {city}_innout.usd 读 start_result_navigation Xform 的 translate（与 view_innout_region 一致）
INDOOR_PRIM_PATHS = [
    "/World/ground/terrain/start_result_navigation",
    "/World/start_result_navigation",
    "/start_result_navigation",
]


def _get_indoor_position_from_prim(stage, prim_path):
    """取 start_result_navigation Xform 的世界变换平移 (x, y)；非 Xformable 则用 bbox 中心。"""
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
    """从 {city}_innout.usd 读室内位置：start_result_navigation Xform 的 translate。返回 (x, y) 或 None。"""
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
                return xy
        return None
    except Exception:
        return None


def load_positions_from_json(json_path):
    """从 Blender 导出的 JSON 读取 object/storefront 世界坐标 (x, y)。"""
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
    """
    用 pxr 从 USD 的 /World/RoadObjects 和 /World/Objaverse 下读取世界坐标 (x, y)。
    Blender 因 .glb 引用失败可能不创建这两棵子树，绘图脚本用此作回退。
    """
    object_positions = []
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        print("[回退] 未安装 pxr (USD)，无法从 USD 读取 object 位置。请在该环境中安装: pip install usd-core")
        return object_positions
    if not os.path.isfile(usd_path):
        print(f"[回退] USD 文件不存在: {usd_path}")
        return object_positions
    try:
        stage = Usd.Stage.Open(usd_path)
        road_path = "/World/RoadObjects"
        objaverse_path = "/World/Objaverse"
        for parent_path in (road_path, objaverse_path):
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
        if object_positions:
            print(f"[回退] 用 pxr 从 USD 读取到 {len(object_positions)} 个 object 位置")
        else:
            print("[回退] USD 中 /World/RoadObjects 与 /World/Objaverse 下无有效子节点，红点将为空")
    except Exception as e:
        print(f"[回退] pxr 读取 USD 失败: {e}")
    return object_positions


def main():
    parser = argparse.ArgumentParser(description="BEV 上叠加 object/storefront 位置并保存图像")
    parser.add_argument("--city", type=str, default="AMSTERDAM", help="城市名，如 AMSTERDAM")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="episode_data 根目录")
    parser.add_argument("--episodes-dir", type=str, default=DEFAULT_EPISODES_DIR, help="episodes JSON 目录")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出图像路径，默认自动生成")
    parser.add_argument("--positions-json", type=str, default=None, help="Blender 导出的位置 JSON；默认与脚本同目录的 usd_positions_{city}.json")
    parser.add_argument("--vc-plus-dir", type=str, default=DEFAULT_VC_PLUS_DIR, help="vc_plus 根目录；object 为空时用 pxr 从此处读 USD 作回退")
    parser.add_argument("--scale-positions", type=float, default=100.0, help="USD/store 有 scale 0.01，从 Blender/pxr 读到的坐标需乘此系数再与 BEV 对齐；若红黄点全在图外可改为 1")
    parser.add_argument("--no-ref-path", action="store_true", help="不绘制 reference path")
    parser.add_argument("--draw-indoor-star", action="store_true", help="从 {city}_innout.usd 读室内位置并画红五角星")
    parser.add_argument("--indoor-scale", type=float, default=1.0, help="室内坐标单位：1=米（默认），100=厘米")
    args = parser.parse_args()

    city = args.city.strip().upper()
    scene_name = f"vc_{city.lower()}_store"

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

    # 底图：优先用 episode_data 里的 BEV RGB（npz 的 rgb 或 bev_map_rgb.png）
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
        # 无 RGB 时退回：灰底
        bg_rgb = np.stack([obstacle_map.astype(np.float32)] * 3, axis=-1)
        bg_rgb[obstacle_map == 0] = [0.85, 0.90, 0.95]
        bg_rgb[obstacle_map == 1] = [0.35, 0.40, 0.50]

    def _world_to_pixel(world_xy):
        return world_to_pixel(world_xy, world_center, px_per_meter, image_width, image_height)

    # 从 Blender 导出的 JSON 读取 object / storefront 位置
    positions_json = args.positions_json
    if positions_json is None:
        positions_json = os.path.join(SCRIPT_DIR, f"usd_positions_{city}.json")
    positions_json = os.path.abspath(positions_json)
    object_positions_world, storefront_positions_world = load_positions_from_json(positions_json)
    if object_positions_world is None and storefront_positions_world is None:
        print(f"未找到位置 JSON 或读取失败: {positions_json}")
        print("请先运行 Blender 脚本导出位置:")
        print(f"  blender --background --python view_out_innout_blender_export.py -- --city {city}")
        return 1
    if object_positions_world is None:
        object_positions_world = []
    if storefront_positions_world is None:
        storefront_positions_world = []
    # Blender 因 .glb 引用失败常不创建 RoadObjects/Objaverse，object 为空时用 pxr 从 USD 回退
    object_from_pxr = False
    if not object_positions_world:
        print("JSON 中 object 为空，尝试用 pxr 从 USD 回退...")
        usd_path = os.path.join(os.path.abspath(args.vc_plus_dir), city, f"{city}_merged.usd")
        object_positions_world = load_object_positions_from_usd(usd_path)
        object_from_pxr = bool(object_positions_world)

    # storefront 来自 Blender（/World/store 有 scale 0.01）→ 需乘 scale_positions；object 来自 pxr 时已是世界坐标，不乘
    scale = float(args.scale_positions)
    scale_object = 1.0 if object_from_pxr else scale
    object_px = [_world_to_pixel((x * scale_object, y * scale_object)) for x, y in object_positions_world]
    storefront_px = [_world_to_pixel((x * scale, y * scale)) for x, y in storefront_positions_world]
    object_px = [(px, py) for px, py in object_px if 0 <= px < image_width and 0 <= py < image_height]
    storefront_px = [(px, py) for px, py in storefront_px if 0 <= px < image_width and 0 <= py < image_height]

    # 检测填充色，取「内容」边界内的最大内接正方形，只保留该正方形内像素，超出直线边界的全部裁掉
    orig_width, orig_height = image_width, image_height
    x0, y0, side = _content_square_crop(bg_rgb, image_width, image_height)
    bg_rgb = bg_rgb[y0 : y0 + side, x0 : x0 + side]
    object_px = [(px - x0, py - y0) for px, py in object_px if x0 <= px < x0 + side and y0 <= py < y0 + side]
    storefront_px = [(px - x0, py - y0) for px, py in storefront_px if x0 <= px < x0 + side and y0 <= py < y0 + side]
    image_width, image_height = side, side

    # 绘图：底图用 episode_data 的 BEV RGB；无 legend/刻度/标题，无白边；输出分辨率 960x960
    TARGET_SIZE = 960
    out_dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=(TARGET_SIZE / out_dpi, TARGET_SIZE / out_dpi), dpi=out_dpi)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(bg_rgb, aspect="auto", extent=(0, image_width, image_height, 0), interpolation="nearest")
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)

    # 点大小随输出尺寸放大，在 960x960 下可见（s 为 points^2，约 40px/90px 直径）
    pt_scale = (TARGET_SIZE / 960.0) ** 2
    if storefront_px:
        ax.scatter([p[0] for p in storefront_px], [p[1] for p in storefront_px], c="yellow", s=60 * pt_scale, alpha=0.9, zorder=2)
    if object_px:
        ax.scatter([p[0] for p in object_px], [p[1] for p in object_px], c="blue", s=60 * pt_scale, alpha=0.95, zorder=3, edgecolors="deepskyblue", linewidths=0.5)

    # 可选：室内场景五角星（从 {city}_innout.usd 读 start_result_navigation 的 translate）
    if args.draw_indoor_star:
        vc_plus_dir = os.path.abspath(args.vc_plus_dir)
        innout_usd = os.path.join(vc_plus_dir, city, f"{city}_innout.usd")
        indoor_xy_raw = load_indoor_position_from_innout_usd(innout_usd)
        if indoor_xy_raw is not None:
            indoor_scale = float(args.indoor_scale)
            indoor_xy = (indoor_xy_raw[0] / indoor_scale, indoor_xy_raw[1] / indoor_scale)
            indoor_px = world_to_pixel(indoor_xy, world_center, px_per_meter, orig_width, orig_height)
            ix, iy = indoor_px[0] - x0, indoor_px[1] - y0
            if 0 <= ix < side and 0 <= iy < side:
                ax.scatter([ix], [iy], c="red", s=2000 * pt_scale, marker="*", zorder=4, edgecolors="darkred", linewidths=0.5)

    # 可选：reference path
    if not args.no_ref_path:
        episodes_dir = os.path.abspath(args.episodes_dir)
        episode_file = os.path.join(episodes_dir, f"{scene_name}.json")
        if os.path.isfile(episode_file):
            with open(episode_file, "r") as f:
                episode_config = json.load(f)
            ref_waypoints_px = []
            for episode_id in [3]:
                if episode_id not in episode_config:
                    continue
                closest_goal_idx = episode_config[episode_id]["closest_goal_idx"]
                ref_path = episode_config[episode_id]["goals"][closest_goal_idx]["reference_path"]
                last_point = None
                for point in ref_path:
                    point = np.array(point)
                    if last_point is not None and np.linalg.norm(point - last_point) < 0.5:
                        continue
                    last_point = point
                    wp = (point[:2] - world_center[:2]) * px_per_meter
                    ref_waypoints_px.append((wp[0] + orig_width // 2, orig_height // 2 - wp[1]))
            # 参考路径从原图坐标转到裁剪后坐标
            ref_waypoints_px = [(px - x0, py - y0) for px, py in ref_waypoints_px if x0 <= px < x0 + side and y0 <= py < y0 + side]
            if ref_waypoints_px:
                ax.plot([p[0] for p in ref_waypoints_px], [p[1] for p in ref_waypoints_px], "b-", linewidth=1, alpha=0.7)
                ax.scatter([ref_waypoints_px[0][0]], [ref_waypoints_px[0][1]], c="blue", s=20, marker="o")
                ax.scatter([ref_waypoints_px[-1][0]], [ref_waypoints_px[-1][1]], c="blue", s=20, marker="*")

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(SCRIPT_DIR, f"view_out_innout_{scene_name}.png")
    plt.savefig(out_path, dpi=out_dpi, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
