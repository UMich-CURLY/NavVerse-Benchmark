import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from navverse.utils.astar import BevAStarReferencePlanner
from navverse.vln_args import load_runtime_config


_WORKER_CONTEXT = {}


def path_length(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=1).sum())


def reaches_goal(path, goal, tolerance):
    if path is None or len(path) == 0:
        return False
    return float(np.linalg.norm(np.asarray(path[-1])[:2] - np.asarray(goal)[:2])) <= float(tolerance)


def prepare_reference_points(ref_path, max_segment_m):
    compact = []
    last_point = None
    min_spacing_m = min(0.5, float(max_segment_m) * 0.5)
    for point in ref_path:
        point = np.asarray(point, dtype=float)
        if last_point is not None and np.linalg.norm(point[:2] - last_point[:2]) < min_spacing_m:
            continue
        compact.append(point)
        last_point = point
    if len(compact) < 2 or max_segment_m <= 0.0:
        return compact

    densified = [compact[0]]
    for start, goal in zip(compact[:-1], compact[1:]):
        distance = float(np.linalg.norm(goal[:2] - start[:2]))
        steps = max(1, int(np.ceil(distance / max_segment_m)))
        for step_idx in range(1, steps + 1):
            t = step_idx / steps
            densified.append(start + t * (goal - start))
    return densified


def draw_path(image, planner, path, color, width=2):
    if path is None or len(path) == 0:
        return
    pts = []
    for point in path:
        x, y, _ = planner.world_to_grid(point)
        pts.append([int(round(x)), int(round(y))])
    pts = np.asarray(pts, dtype=np.int32)
    if len(pts) > 1:
        cv2.polylines(image, [pts], False, color, width, lineType=cv2.LINE_AA)
    cv2.circle(image, tuple(pts[0]), 5, (0, 180, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, tuple(pts[-1]), 5, (220, 0, 0), -1, lineType=cv2.LINE_AA)


def write_overlay(path, planner, sample, planned_path):
    obstacle_map = planner.obstacle_map
    vis = np.full((*obstacle_map.shape, 3), 245, dtype=np.uint8)
    vis[obstacle_map > 0] = (35, 35, 35)
    draw_path(vis, planner, np.asarray(sample["reference_path"], dtype=float), (255, 120, 0), width=2)
    draw_path(vis, planner, planned_path, (0, 0, 255), width=2)
    cv2.imwrite(str(path), vis)


def select_safer_segment(planner, path, start, goal, astar_config):
    direct_path = planner.direct_world_path(start, goal, spacing=float(astar_config["waypoint"]["spacing_m"]) * 0.5)
    if not planner.is_world_path_collision_free(direct_path):
        return path
    path_min, path_avg = planner.world_path_clearance(path)
    direct_min, direct_avg = planner.world_path_clearance(direct_path)
    min_clearance = float(astar_config["planner"]["min_path_clearance_m"])
    if direct_avg > path_avg + 0.05 and (path_min < min_clearance or direct_min >= path_min):
        return direct_path
    return path


def try_plan_segment(planner, start, goal, time_budget, goal_tolerance, astar_config):
    path = planner.plan_reference_path(start, goal, time_budget=float(time_budget))
    if reaches_goal(path, goal, goal_tolerance):
        return select_safer_segment(planner, path, start, goal, astar_config)
    retry_multiplier = float(astar_config["segmentation"]["retry_time_budget_multiplier"])
    if retry_multiplier > 1.0:
        path = planner.plan_reference_path(start, goal, time_budget=float(time_budget) * retry_multiplier)
        if reaches_goal(path, goal, goal_tolerance):
            return select_safer_segment(planner, path, start, goal, astar_config)
    return path


def direct_segment_if_clear(planner, start, goal, goal_tolerance, astar_config):
    direct_path = planner.direct_world_path(start, goal, spacing=float(astar_config["waypoint"]["spacing_m"]) * 0.5)
    if planner.is_world_path_collision_free(direct_path) and reaches_goal(direct_path, goal, goal_tolerance):
        return direct_path
    return None


def astar_direct_connection_ok(planner, start, goal, smoothing_config):
    spacing_m = float(smoothing_config.get("shortcut_check_spacing_m", 0.25))
    min_clearance_m = float(smoothing_config.get("min_shortcut_clearance_m", 0.0))
    direct_path = planner.direct_world_path(start, goal, spacing=max(0.05, spacing_m))
    if not planner.is_world_path_collision_free(direct_path):
        return False
    if min_clearance_m > 0.0:
        min_clearance, _ = planner.world_path_clearance(direct_path)
        if min_clearance < min_clearance_m:
            return False
    return True


def point_line_offset_xy(point, start, goal):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    segment = goal[:2] - start[:2]
    denom = float(np.linalg.norm(segment))
    if denom < 1e-9:
        return float(np.linalg.norm(point[:2] - start[:2]))
    return float(abs(np.cross(segment, point[:2] - start[:2]) / denom))


def turn_angle_abs(prev, cur, nxt):
    prev = np.asarray(prev, dtype=float)
    cur = np.asarray(cur, dtype=float)
    nxt = np.asarray(nxt, dtype=float)
    v1 = cur[:2] - prev[:2]
    v2 = nxt[:2] - cur[:2]
    if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
        return 0.0
    yaw1 = np.arctan2(v1[1], v1[0])
    yaw2 = np.arctan2(v2[1], v2[0])
    return float(abs((yaw2 - yaw1 + np.pi) % (2 * np.pi) - np.pi))


def shortcut_world_path(planner, path, smoothing_config):
    if len(path) < 3:
        return path
    max_span_m = float(smoothing_config.get("max_shortcut_span_m", 8.0))
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        best_j = i + 1
        for j in range(len(path) - 1, i + 1, -1):
            if max_span_m > 0.0 and np.linalg.norm(path[j][:2] - path[i][:2]) > max_span_m:
                continue
            if astar_direct_connection_ok(planner, path[i], path[j], smoothing_config):
                best_j = j
                break
        out.append(path[best_j])
        i = best_j
    return out


def prune_astar_jitter(planner, path, smoothing_config):
    if len(path) < 3:
        return path
    max_passes = max(1, int(smoothing_config.get("jitter_max_passes", 3)))
    max_chord_m = float(smoothing_config.get("jitter_max_chord_m", 3.0))
    max_offset_m = float(smoothing_config.get("jitter_max_offset_m", 0.45))
    max_turn_rad = np.deg2rad(float(smoothing_config.get("jitter_max_turn_deg", 60.0)))
    points = list(path)
    for _ in range(max_passes):
        changed = False
        out = [points[0]]
        i = 1
        while i < len(points) - 1:
            prev = out[-1]
            cur = points[i]
            nxt = points[i + 1]
            chord = float(np.linalg.norm(nxt[:2] - prev[:2]))
            if chord <= max_chord_m:
                offset = point_line_offset_xy(cur, prev, nxt)
                turn = turn_angle_abs(prev, cur, nxt)
                if (
                    offset <= max_offset_m
                    and turn <= max_turn_rad
                    and astar_direct_connection_ok(planner, prev, nxt, smoothing_config)
                ):
                    changed = True
                    i += 1
                    continue
            out.append(cur)
            i += 1
        out.append(points[-1])
        points = out
        if not changed:
            break
    return points


def smooth_world_path(planner, path, astar_config, source):
    path = [np.asarray(point, dtype=float) for point in path]
    smoothing_config = astar_config.get("smoothing", {})
    info = {
        "enabled": False,
        "source": source,
        "input_points": int(len(path)),
        "output_points": int(len(path)),
        "shortcut_points": int(len(path)),
        "jitter_points": int(len(path)),
        "input_length_m": path_length(path),
        "output_length_m": path_length(path),
    }
    if not smoothing_config.get("enabled", False) or source == "failed" or len(path) < 3:
        return np.asarray(path, dtype=float), info
    info["enabled"] = True
    smoothed = path
    if smoothing_config.get("shortcut_enabled", True):
        smoothed = shortcut_world_path(planner, smoothed, smoothing_config)
    info["shortcut_points"] = int(len(smoothed))
    if smoothing_config.get("jitter_prune_enabled", True):
        smoothed = prune_astar_jitter(planner, smoothed, smoothing_config)
    info["jitter_points"] = int(len(smoothed))
    info["output_points"] = int(len(smoothed))
    info["output_length_m"] = path_length(smoothed)
    return np.asarray(smoothed, dtype=float), info


def plan_guided_segments(planner, ref_points, astar_config, goal_tolerance):
    segmentation_config = _segmentation_config(astar_config)
    segment_time_budget = float(segmentation_config["segment_time_budget_s"])
    total_time_budget = float(segmentation_config["segment_total_budget_s"])
    start_time = time.time()
    total_path = []
    start = ref_points[0]
    for goal in ref_points[1:]:
        remaining_time = total_time_budget - (time.time() - start_time)
        if remaining_time <= 0.0:
            return None
        segment_budget = min(segment_time_budget, remaining_time)
        path = try_plan_segment(planner, start, goal, segment_budget, goal_tolerance, astar_config)
        if not reaches_goal(path, goal, goal_tolerance):
            path = direct_segment_if_clear(planner, start, goal, goal_tolerance, astar_config)
            if path is None:
                return None
        if total_path:
            total_path.extend(path[1:])
        else:
            total_path.extend(path)
        start = path[-1]
    return np.asarray(total_path, dtype=float) if len(total_path) >= 2 else None


def _planner_config(astar_config):
    return astar_config["planner"]


def _global_planner_config(astar_config):
    return astar_config["global_planner"]


def _segmentation_config(astar_config):
    return astar_config["segmentation"]


def create_planner(astar_data_dir, scene_meta, scene_id, astar_config):
    planner_config = _planner_config(astar_config)
    bev_path = Path(astar_data_dir) / scene_meta[scene_id]["bev_map"]
    return BevAStarReferencePlanner(
        bev_path,
        max_climb_height=float(planner_config["max_climb_height_m"]),
        robot_width=float(planner_config["robot_width_m"]),
        robot_height=float(planner_config["robot_height_m"]),
        step_size=float(planner_config["step_size_m"]),
        step_size_yaw=int(planner_config["step_size_yaw_deg"]),
        reach_threshold=float(planner_config["reach_threshold_m"]),
        reach_yaw=float(planner_config["reach_yaw_deg"]),
        clearance_m=float(planner_config["clearance_m"]),
        preferred_clearance_m=float(planner_config["preferred_clearance_m"]),
        clearance_cost=float(planner_config["clearance_cost"]),
        min_obstacle_area_px=int(planner_config["min_obstacle_area_px"]),
    )


def sample_reference_points(sample, astar_config):
    segmentation_config = _segmentation_config(astar_config)
    reference_path = [np.asarray(point, dtype=float) for point in sample["reference_path"]]
    goal_position = sample.get("goal_position")
    if goal_position is not None:
        target_goal = np.asarray(goal_position, dtype=float)[:3]
        if target_goal.shape[0] >= 2:
            if target_goal.shape[0] < 3:
                target_goal = np.array([target_goal[0], target_goal[1], reference_path[-1][2]], dtype=float)
            if np.linalg.norm(reference_path[-1][:2] - target_goal[:2]) > 0.05:
                reference_path.append(target_goal)
            else:
                reference_path[-1] = target_goal
    ref_points = prepare_reference_points(reference_path, float(segmentation_config["max_segment_m"]))
    goal = np.asarray(ref_points[-1], dtype=float)
    goal_tolerance = float(segmentation_config["goal_tolerance_m"])
    if sample.get("goal_radius") is not None:
        goal_tolerance = max(goal_tolerance, max(0.0, float(sample["goal_radius"])) + 1.0)
    return ref_points, goal, goal_tolerance


def run_config(planner, sample, astar_config, config_name):
    global_planner_config = _global_planner_config(astar_config)
    ref_points, goal, goal_tolerance = sample_reference_points(sample, astar_config)
    start = ref_points[0]
    started = time.time()
    source = None
    planned_path = np.asarray([], dtype=float)
    if config_name == "yaw":
        planned_path = planner.plan_reference_path(start, goal, time_budget=float(global_planner_config["time_budget_s"]))
        if reaches_goal(planned_path, goal, goal_tolerance):
            planned_path = select_safer_segment(planner, planned_path, start, goal, astar_config)
            source = "global_direct"
    elif config_name == "grid2d":
        planned_path = planner.plan_reference_path_2d(
            start,
            goal,
            time_budget=float(global_planner_config["grid2d_time_budget_s"]),
            resolution_m=float(global_planner_config["grid2d_resolution_m"]),
            clearance_m=float(global_planner_config["grid2d_clearance_m"]),
            snap_distance_m=float(global_planner_config["grid2d_snap_m"]),
        )
        if reaches_goal(planned_path, goal, goal_tolerance):
            source = "global_grid2d"
        else:
            direct_path = try_plan_segment(
                planner,
                start,
                goal,
                float(global_planner_config["time_budget_s"]),
                goal_tolerance,
                astar_config,
            )
            if reaches_goal(direct_path, goal, goal_tolerance):
                planned_path = direct_path
                source = "global_direct"
    else:
        raise ValueError(f"unknown config: {config_name}")
    if not reaches_goal(planned_path, goal, goal_tolerance):
        guided_path = plan_guided_segments(planner, ref_points, astar_config, goal_tolerance)
        if guided_path is not None:
            planned_path = guided_path
            source = "global_segmented"
        else:
            source = source or "failed"
    if planned_path is None:
        planned_path = np.asarray([], dtype=float)
    if len(planned_path) and np.linalg.norm(np.asarray(planned_path[-1])[:2] - goal[:2]) > goal_tolerance:
        target_connection = np.asarray([planned_path[-1], goal], dtype=float)
        if planner.is_world_path_collision_free(target_connection):
            planned_path = np.asarray(list(planned_path) + [goal], dtype=float)
    smoothing_info = None
    if len(planned_path):
        planned_path, smoothing_info = smooth_world_path(planner, planned_path, astar_config, source)
    elapsed = time.time() - started
    end_distance = float("inf") if len(planned_path) == 0 else float(np.linalg.norm(planned_path[-1, :2] - goal[:2]))
    reached = bool(end_distance <= goal_tolerance)
    collision_free = bool(len(planned_path) > 0 and planner.is_world_path_collision_free(planned_path))
    min_clearance, avg_clearance = planner.world_path_clearance(planned_path) if len(planned_path) else (0.0, 0.0)
    return {
        "config": config_name,
        "source": source,
        "planned": bool(reached),
        "collision_free": collision_free,
        "plan_time_s": elapsed,
        "path_points": int(len(planned_path)),
        "path_length_m": path_length(planned_path) if len(planned_path) else 0.0,
        "end_distance_m": end_distance,
        "goal_tolerance_m": goal_tolerance,
        "goal_position": np.asarray(goal, dtype=float).tolist(),
        "goal_radius_m": sample.get("goal_radius"),
        "smoothing": smoothing_info,
        "min_clearance_m": min_clearance,
        "avg_clearance_m": avg_clearance,
        "path": np.asarray(planned_path, dtype=float).tolist() if len(planned_path) else [],
    }, planned_path


def init_worker(astar_data_dir, scene_meta, astar_config, requested_configs, save_overlays, overlay_dir):
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "astar_data_dir": astar_data_dir,
            "scene_meta": scene_meta,
            "astar_config": astar_config,
            "requested_configs": requested_configs,
            "save_overlays": save_overlays,
            "overlay_dir": overlay_dir,
            "planner_cache": {},
        }
    )


def worker_run_sample(index_and_sample):
    index, sample = index_and_sample
    scene_id = sample["scene_id"]
    planner_cache = _WORKER_CONTEXT["planner_cache"]
    planner = planner_cache.get(scene_id)
    if planner is None:
        planner = create_planner(
            _WORKER_CONTEXT["astar_data_dir"],
            _WORKER_CONTEXT["scene_meta"],
            scene_id,
            _WORKER_CONTEXT["astar_config"],
        )
        planner_cache[scene_id] = planner

    sample_result = {
        "sample_id": sample["sample_id"],
        "scene_id": scene_id,
        "reference_path_length_m": sample.get("reference_path_length_m"),
        "original_astar_source": sample.get("original_astar_source"),
        "configs": {},
    }
    for config_name in _WORKER_CONTEXT["requested_configs"]:
        result, planned_path = run_config(planner, sample, _WORKER_CONTEXT["astar_config"], config_name)
        sample_result["configs"][config_name] = result
        if _WORKER_CONTEXT["save_overlays"]:
            overlay_path = Path(_WORKER_CONTEXT["overlay_dir"]) / f"{sample['sample_id']}_{config_name}.png"
            write_overlay(overlay_path, planner, sample, planned_path)
            result["overlay"] = str(overlay_path)
    return index, sample_result


def main():
    parser = argparse.ArgumentParser(description="Run A* planners offline on data saved by navverse_astar_online_multiprocessing.py.")
    parser.add_argument("--astar_data_dir", default="navverse_data/astar_data")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--configs", default="yaw,grid2d", help="Comma-separated offline planners to run: yaw,grid2d.")
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args, config_overrides = parser.parse_known_args()
    requested_configs = [item.strip() for item in str(args.configs).split(",") if item.strip()]
    unknown_configs = sorted(set(requested_configs) - {"yaw", "grid2d"})
    if unknown_configs:
        raise ValueError(f"Unknown offline planner configs: {unknown_configs}")
    runtime_config = load_runtime_config(args.config, config_overrides)
    astar_config = OmegaConf.to_container(runtime_config.astar, resolve=True)

    astar_data_dir = Path(args.astar_data_dir)
    metadata = json.loads((astar_data_dir / "metadata.json").read_text())
    output_path = Path(args.output_json)
    overlay_dir = output_path.with_suffix("")
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    samples = list(metadata.get("samples", []))
    worker_count = max(1, min(int(args.workers), len(samples) or 1))
    indexed_samples = list(enumerate(samples))
    if worker_count == 1:
        init_worker(str(astar_data_dir), metadata["scenes"], astar_config, requested_configs, args.save_overlays, str(overlay_dir))
        indexed_results = [worker_run_sample(item) for item in indexed_samples]
    else:
        print(f"[OFFLINE] Running {len(samples)} samples with {worker_count} workers")
        with mp.Pool(
            processes=worker_count,
            initializer=init_worker,
            initargs=(str(astar_data_dir), metadata["scenes"], astar_config, requested_configs, args.save_overlays, str(overlay_dir)),
        ) as pool:
            indexed_results = []
            chunksize = max(1, min(10, int(np.ceil(len(indexed_samples) / (worker_count * 4)))))
            for done_count, item in enumerate(pool.imap_unordered(worker_run_sample, indexed_samples, chunksize=chunksize), start=1):
                indexed_results.append(item)
                if done_count == len(indexed_samples) or done_count % 10 == 0:
                    print(f"[OFFLINE] Completed {done_count}/{len(indexed_samples)} samples")
    results = [result for _, result in sorted(indexed_results, key=lambda item: item[0])]

    payload = {
        "astar_data_dir": str(astar_data_dir),
        "args": vars(args),
        "config_overrides": config_overrides,
        "astar_config": astar_config,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[OFFLINE] Wrote {len(results)} sample results to {output_path}")


if __name__ == "__main__":
    main()
