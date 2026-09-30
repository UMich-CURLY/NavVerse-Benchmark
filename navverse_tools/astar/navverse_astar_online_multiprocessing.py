import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from dotenv import dotenv_values


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def episode_distance(episode):
    goal = episode["goals"][episode["closest_goal_idx"]]
    ref_path = goal.get("source_reference_path") or goal["reference_path"]
    if len(ref_path) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(float(cur[0]) - float(prev[0]), float(cur[1]) - float(prev[1]))
            for prev, cur in zip(ref_path[:-1], ref_path[1:])
        )
    )


def scene_group(label):
    if label.startswith("grCommercial"):
        return "gr"
    if label.startswith("vc_"):
        return "vc"
    if label.startswith("innout_"):
        return "innout"
    return "other"


def jsonable(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def select_episode_labels(args):
    from navverse.episode import VLNEpisode

    episodes = VLNEpisode.from_json_folder(args.episode_folder)
    by_label = {episode.episode_label: episode for episode in episodes}
    if args.batch_episode_labels:
        missing = [label for label in args.batch_episode_labels if label not in by_label]
        if missing:
            raise ValueError(f"Unknown episode labels: {missing}")
        if args.batch_max_ref_path_m > 0.0:
            too_long = [
                (label, episode_distance(by_label[label]))
                for label in args.batch_episode_labels
                if episode_distance(by_label[label]) > args.batch_max_ref_path_m
            ]
            if too_long:
                detail = ", ".join(f"{label}={distance:.1f}m" for label, distance in too_long)
                raise ValueError(
                    f"Selected episodes exceed --batch_max_ref_path_m={args.batch_max_ref_path_m:.1f}: {detail}. "
                    "Raise the limit or pass --batch_max_ref_path_m 0 to run hard historical episodes."
                )
        return list(dict.fromkeys(args.batch_episode_labels))

    labels = []
    allowed_groups = set(args.batch_scene_groups)
    candidates = []
    for episode in episodes:
        label = episode.episode_label
        group = scene_group(label)
        if group not in allowed_groups:
            continue
        distance = episode_distance(episode)
        if args.batch_min_ref_path_m > 0.0 and distance < args.batch_min_ref_path_m:
            continue
        if args.batch_max_ref_path_m > 0.0 and distance > args.batch_max_ref_path_m:
            continue
        if args.batch_requires_innout_exit and group == "innout":
            text = f"{episode.get('instruction', '')} {episode.get('objnav', '')}".lower()
            if not any(keyword in text for keyword in ("outside", "exit", "outdoor", "street")):
                continue
        candidates.append((group, distance, label))

    for group in args.batch_scene_groups:
        group_candidates = sorted(
            [item for item in candidates if item[0] == group],
            key=lambda item: item[1],
            reverse=True,
        )
        labels.extend(label for _, _, label in group_candidates[: args.batch_per_group])
    return labels


def split_labels(labels, worker_count):
    chunks = [[] for _ in range(worker_count)]
    for index, label in enumerate(labels):
        chunks[index % worker_count].append(label)
    return [chunk for chunk in chunks if chunk]


def load_child_summary(path):
    if not path.exists():
        return {"episodes": []}
    return json.loads(path.read_text())


def scene_bev_files(scene_folder, astar_data_dir, scene_id):
    if not scene_folder:
        return {}
    src_dir = Path(scene_folder) / "episode_data" / scene_id
    if not src_dir.exists():
        return {}
    dst_dir = Path(astar_data_dir) / "scenes" / scene_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    bev_name = "nav_bev_map.npz" if (src_dir / "nav_bev_map.npz").exists() else "bev_map.npz"
    copied = {}
    for name in {
        bev_name,
        bev_name.replace(".npz", "_info.txt"),
        bev_name.replace(".npz", "_depth.png"),
        bev_name.replace(".npz", "_rgb.png"),
        "bev_map.npz",
        "bev_map_info.txt",
        "bev_map_depth.png",
        "bev_map_rgb.png",
    }:
        src = src_dir / name
        if src.exists():
            dst = dst_dir / name
            shutil.copy2(src, dst)
            copied[name] = str(dst.relative_to(astar_data_dir))
    if bev_name in copied:
        copied["bev_map"] = copied[bev_name]
    return copied


def source_reference_path(episode, goal_idx):
    goal = episode["goals"][goal_idx]
    return goal.get("source_reference_path") or goal.get("reference_path", [])


def done_flag(done):
    if done is None:
        return False
    if hasattr(done, "detach"):
        return bool(done.detach().cpu().numpy().any())
    if hasattr(done, "any"):
        return bool(done.any())
    return bool(done)


def collect_measurements(vln_sim):
    info = getattr(vln_sim, "info", None)
    if isinstance(info, dict) and "measurements" in info:
        return jsonable(info["measurements"])
    return {}


def ref_goal_distance(vln_sim, episode):
    body_pos, _ = vln_sim.env.get_body_pose()
    goal = episode["goals"][episode["closest_goal_idx"]]
    target = goal.get("location", goal["reference_path"][-1])
    return float(math.hypot(float(body_pos[0]) - float(target[0]), float(body_pos[1]) - float(target[1])))


def remaining_reference_path(vln_sim, episode):
    body_pos, _ = vln_sim.env.get_body_pose()
    ref_path = source_reference_path(episode, episode["closest_goal_idx"])
    body_point = [float(body_pos[0]), float(body_pos[1]), float(body_pos[2])]
    nearest_idx = min(
        range(len(ref_path)),
        key=lambda idx: math.hypot(float(ref_path[idx][0]) - body_point[0], float(ref_path[idx][1]) - body_point[1]),
    )
    start_idx = min(len(ref_path) - 1, nearest_idx + 1)
    remaining = [body_point]
    remaining.extend(ref_path[start_idx:])
    if len(remaining) == 1:
        remaining.append(ref_path[-1])
    return remaining


def save_offline_sample(args, vln_sim, episode, result):
    scene_id = episode["scene_id"]
    goal_idx = int(episode["closest_goal_idx"])
    goal = episode["goals"][goal_idx]
    sample_id = f"{args.astar_data_run_id}_{episode.episode_label}"
    astar_data_dir = Path(args.astar_data_dir)
    scene_files = scene_bev_files(args.scene_folder, astar_data_dir, scene_id)
    sample = {
        "sample_id": sample_id,
        "run_id": args.astar_data_run_id,
        "episode_label": episode.episode_label,
        "episode_id": episode.get("episode_id"),
        "scene_id": scene_id,
        "scene_group": scene_group(episode.episode_label),
        "goal_name": goal.get("name") or goal.get("object_name") or goal.get("instance"),
        "goal_position": jsonable(goal.get("location")),
        "goal_radius": None if goal.get("radius") is None else float(goal.get("radius")),
        "start_position": jsonable(source_reference_path(episode, goal_idx)[0]),
        "reference_path": jsonable(source_reference_path(episode, goal_idx)),
        "reference_path_length_m": episode_distance(episode),
        "astar_world_path": jsonable(getattr(vln_sim, "last_astar_world_path", None) or []),
        "astar_waypoints": jsonable(getattr(vln_sim, "last_astar_waypoints", None) or []),
        "astar_plan_info": jsonable(getattr(vln_sim, "last_astar_plan_info", {}) or {}),
        "online_result": jsonable(result),
        "scene_files": scene_files,
    }
    sample_path = astar_data_dir / "samples" / f"{sample_id}.json"
    write_json(sample_path, sample)
    return str(sample_path.relative_to(astar_data_dir))


def write_astar_metadata(astar_data_dir, summary):
    astar_data_dir = Path(astar_data_dir)
    samples = []
    scenes = {}
    for episode in summary.get("episodes", []):
        sample_rel = episode.get("offline_sample")
        if not sample_rel:
            continue
        sample_path = astar_data_dir / sample_rel
        if not sample_path.exists():
            continue
        sample = json.loads(sample_path.read_text())
        samples.append(sample)
        scene_id = sample["scene_id"]
        scenes.setdefault(scene_id, sample.get("scene_files", {}))
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": summary.get("run_id"),
        "source_online_results": summary.get("output_json"),
        "scenes": scenes,
        "samples": samples,
    }
    write_json(astar_data_dir / "metadata.json", metadata)


def parent_parser():
    dot_args = dotenv_values(".env")
    parser = argparse.ArgumentParser(description="Run online A* evaluation with multiple Isaac worker processes.")
    parser.add_argument(
        "--episode_folder",
        default=dot_args.get("EPISODE_FOLDER", "navverse_data/episodes"),
    )
    parser.add_argument("--batch_episode_labels", nargs="*", default=None)
    parser.add_argument("--batch_scene_groups", nargs="+", default=["gr", "vc", "innout"], choices=["gr", "vc", "innout"])
    parser.add_argument("--batch_per_group", type=int, default=3)
    parser.add_argument("--batch_min_ref_path_m", type=float, default=8.0)
    parser.add_argument("--batch_max_ref_path_m", type=float, default=50.0)
    parser.add_argument("--batch_requires_innout_exit", type=str2bool, default=False)
    parser.add_argument("--batch_execute", type=str2bool, default=True)
    parser.add_argument("--max_frames_per_episode", type=int, default=900)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch_output_json", default="dump/astar_online/results.json")
    parser.add_argument("--child_output_dir", default="dump/astar_online/children")
    parser.add_argument("--astar_data_dir", default="navverse_data/astar_data")
    parser.add_argument("--astar_data_run_id", default=None)
    parser.add_argument("--stop_on_failure", type=str2bool, default=False)
    return parser


def run_parent():
    parser = parent_parser()
    args, worker_extra_args = parser.parse_known_args()
    labels = select_episode_labels(args)
    if not labels:
        raise ValueError("No episodes selected for online A* evaluation")

    run_id = args.astar_data_run_id or time.strftime("run_%Y%m%d_%H%M%S", time.gmtime())
    args.astar_data_run_id = run_id
    output_path = Path(args.batch_output_json)
    child_output_dir = Path(args.child_output_dir)
    worker_count = max(1, min(int(args.workers), len(labels)))
    chunks = split_labels(labels, worker_count)
    summary = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "output_json": str(output_path),
        "selected_episode_labels": labels,
        "batch_min_ref_path_m": args.batch_min_ref_path_m,
        "batch_max_ref_path_m": args.batch_max_ref_path_m,
        "workers": len(chunks),
        "episodes": [],
        "astar_data_dir": args.astar_data_dir,
    }
    write_json(output_path, summary)

    processes = []
    for worker_index, chunk in enumerate(chunks):
        child_json = child_output_dir / f"worker_{worker_index:02d}.json"
        cmd = [
            sys.executable,
            "-u",
            __file__,
            "--worker",
            "--worker_index",
            str(worker_index),
            "--worker_labels",
            *chunk,
            "--batch_output_json",
            str(child_json),
            "--batch_execute",
            str(args.batch_execute).lower(),
            "--max_frames_per_episode",
            str(args.max_frames_per_episode),
            "--astar_data_dir",
            args.astar_data_dir,
            "--astar_data_run_id",
            run_id,
            "--disable_socket_server",
            "true",
            "--foxglove_port",
            "0",
            "astar.debug.save_visualizations=true",
            *worker_extra_args,
        ]
        print(f"[ONLINE] Worker {worker_index} labels={chunk}")
        processes.append((worker_index, child_json, subprocess.Popen(cmd)))

    failed = False
    for worker_index, child_json, process in processes:
        returncode = process.wait()
        child_summary = load_child_summary(child_json)
        for result in child_summary.get("episodes", []):
            result["worker_index"] = worker_index
            result["child_returncode"] = returncode
            summary["episodes"].append(result)
        write_json(output_path, summary)
        if args.stop_on_failure and (returncode != 0 or any(not item.get("reached_goal") for item in child_summary.get("episodes", []))):
            failed = True

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_astar_metadata(args.astar_data_dir, summary)
    write_json(output_path, summary)
    print(f"[ONLINE] Saved summary: {output_path}")
    print(f"[ONLINE] Saved offline A* data: {Path(args.astar_data_dir) / 'metadata.json'}")
    if failed:
        raise SystemExit(2)


def configure_worker_parser():
    from isaaclab.app import AppLauncher

    import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
    import navverse.vln_args as vln_cli_args

    parser = argparse.ArgumentParser(description="Online A* worker")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--worker_labels", nargs="+", required=True)
    parser.add_argument("--batch_execute", type=vln_cli_args.str2bool, default=True)
    parser.add_argument("--max_frames_per_episode", type=int, default=900)
    parser.add_argument("--batch_output_json", type=str, required=True)
    parser.add_argument("--slow_waypoint_follower", type=vln_cli_args.str2bool, default=False)
    parser.add_argument("--astar_data_dir", default="navverse_data/astar_data")
    parser.add_argument("--astar_data_run_id", required=True)
    rsl_rl_cli_args.add_rsl_rl_args(parser)
    vln_cli_args.add_vln_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(
        enable_locomotion_reward=False,
        disable_camera=False,
        trajectory_planner="astar",
        config="configs/default.yaml",
        on_demand_render=True,
        headless=True,
    )
    return parser, AppLauncher, vln_cli_args


def run_recovery_motion(vln_sim, episode, result, args, replan_record):
    if int(getattr(vln_sim, "command_dim", 0)) != 3:
        return
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    astar_execution = args.astar.execution
    frames = max(0, int(astar_execution.recovery_motion_frames))
    if frames == 0:
        return
    body_pos, body_quat = vln_sim.env.get_body_pose()
    body_yaw = float(R.from_quat(body_quat).as_euler("ZYX")[0])
    remaining = remaining_reference_path(vln_sim, episode)
    target = remaining[-1]
    for point in remaining[1:]:
        if math.hypot(float(point[0]) - float(body_pos[0]), float(point[1]) - float(body_pos[1])) >= 0.8:
            target = point
            break
    target_yaw = math.atan2(float(target[1]) - float(body_pos[1]), float(target[0]) - float(body_pos[0]))
    yaw_error = math.atan2(math.sin(target_yaw - body_yaw), math.cos(target_yaw - body_yaw))
    turn_vel = 0.0
    if abs(yaw_error) >= math.radians(10.0):
        turn_vel = math.copysign(min(float(astar_execution.recovery_turn_vel), max(0.25, abs(yaw_error) * 0.8)), yaw_error)
    lateral_vel = 0.0
    if str(episode.scene_id).startswith("grCommercial"):
        lateral_vel = (1.0 if result["replan_count"] % 2 == 0 else -1.0) * float(astar_execution.recovery_lateral_vel)
    command = vln_sim.commands.new_tensor([float(astar_execution.recovery_backward_vel), lateral_vel, float(turn_vel)])
    vln_sim.clear_waypoints()
    before_xy_yaw = [float(body_pos[0]), float(body_pos[1]), float(body_yaw)]
    for _ in range(frames):
        vln_sim.commands[vln_sim.robot_index] = command
        vln_sim.step()
        result["frame_count"] += 1
        if done_flag(vln_sim.done):
            result["terminated"] = True
            break
    vln_sim.commands[vln_sim.robot_index] = vln_sim.commands.new_zeros((vln_sim.command_dim,))
    after_pos, after_quat = vln_sim.env.get_body_pose()
    after_yaw = float(R.from_quat(after_quat).as_euler("ZYX")[0])
    replan_record["recovery"] = {
        "frames": frames,
        "command": [float(astar_execution.recovery_backward_vel), float(lateral_vel), float(turn_vel)],
        "yaw_error_deg": float(np.degrees(yaw_error)),
        "target_xy": [float(target[0]), float(target[1])],
        "before_xy_yaw": before_xy_yaw,
        "after_xy_yaw": [float(after_pos[0]), float(after_pos[1]), float(after_yaw)],
    }


def maybe_replan_if_stuck(vln_sim, episode, result, args, best_distance, frames_since_progress):
    astar_execution = args.astar.execution
    if not astar_execution.replan_on_stuck:
        return best_distance, frames_since_progress
    if result["replan_count"] >= astar_execution.max_replans:
        return best_distance, frames_since_progress
    if frames_since_progress < astar_execution.stuck_window_frames:
        return best_distance, frames_since_progress

    current_distance = ref_goal_distance(vln_sim, episode)
    if best_distance - current_distance >= astar_execution.stuck_min_progress_m:
        return min(best_distance, current_distance), 0

    replan_record = {
        "frame": result["frame_count"],
        "distance_before_m": current_distance,
        "waypoints_before": int(len(vln_sim.waypoints or [])),
        "success": False,
        "error": None,
    }
    try:
        run_recovery_motion(vln_sim, episode, result, args, replan_record)
        goal = episode["goals"][episode["closest_goal_idx"]]
        waypoints = vln_sim.generate_astar_path(
            episode["scene_id"],
            remaining_reference_path(vln_sim, episode),
            goal_position=goal.get("location"),
            goal_radius=goal.get("radius"),
        )
        if not waypoints:
            raise RuntimeError("replan generated no waypoints")
        vln_sim.set_waypoints(waypoints)
        vln_sim.waiting_for_first_action = False
        result["replan_count"] += 1
        replan_record["success"] = True
        replan_record["waypoints_after"] = int(len(waypoints))
        return current_distance, 0
    except Exception as exc:
        replan_record["error"] = repr(exc)
        return best_distance, 0
    finally:
        result.setdefault("replans", []).append(replan_record)


def run_worker():
    parser, AppLauncher, vln_cli_args = configure_worker_parser()
    args = vln_cli_args.parse_args(parser)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb

    from navverse.sim import VLNSim

    settings = carb.settings.get_settings()

    def app_is_running():
        try:
            return simulation_app.is_running()
        except Exception:
            return False

    def app_is_exiting():
        try:
            return simulation_app.is_exiting()
        except Exception:
            return True

    def app_backend_is_running():
        try:
            return simulation_app.app.is_running()
        except Exception:
            return False

    summary = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "worker_index": args.worker_index,
        "worker_labels": args.worker_labels,
        "trajectory_planner": args.trajectory_planner,
        "astar_config": args.astar.config_name,
        "batch_execute": args.batch_execute,
        "max_frames_per_episode": args.max_frames_per_episode,
        "episodes": [],
    }
    output_path = Path(args.batch_output_json)
    vln_sim = None
    try:
        vln_sim = VLNSim(args)
        if args.slow_waypoint_follower:
            import numpy as np

            vln_sim.waypoint_follower.max_vel = np.array([0.7, 0.2, 0.8])
            vln_sim.waypoint_follower.min_vel = np.array([0.15, 0.05, 0.15])
            vln_sim.waypoint_follower.kp = np.array([1.6, 0.4, 1.8])

        for batch_index, episode_label in enumerate(args.worker_labels):
            if not app_is_running():
                break
            episode_start = time.time()
            result = {
                "batch_index": batch_index,
                "episode_label": episode_label,
                "scene_group": scene_group(episode_label),
                "planned": False,
                "planning_error": None,
                "planned_waypoints": 0,
                "astar_plan_info": None,
                "reference_path_m": None,
                "executed": bool(args.batch_execute),
                "frame_count": 0,
                "reached_goal": False,
                "terminated": False,
                "terminations": None,
                "measurements": {},
                "final_ref_goal_distance_m": None,
                "replan_count": 0,
                "replans": [],
                "elapsed_s": None,
                "command_stats": None,
                "debug_dir": str(Path(vln_sim._astar_config().debug.dir) / episode_label),
                "offline_sample": None,
            }
            print(f"[WORKER {args.worker_index}] [{batch_index + 1}/{len(args.worker_labels)}] {episode_label}", flush=True)
            try:
                vln_sim.load_episode(episode_label)
                vln_sim.step()
                episode = vln_sim.current_episode
                result["reference_path_m"] = episode_distance(episode)
                vln_sim.set_ref_waypoints(episode)
                vln_sim.waiting_for_first_action = False
                result["planned"] = vln_sim.waypoints is not None and len(vln_sim.waypoints) > 0
                result["planned_waypoints"] = int(len(vln_sim.waypoints or []))
                result["astar_plan_info"] = jsonable(getattr(vln_sim, "last_astar_plan_info", {}))
                result["offline_sample"] = save_offline_sample(args, vln_sim, episode, result)

                best_ref_distance = ref_goal_distance(vln_sim, episode)
                frames_since_progress = 0
                command_abs_sum = [0.0, 0.0, 0.0]
                command_abs_max = [0.0, 0.0, 0.0]
                command_samples = 0

                while args.batch_execute and app_is_running():
                    if result["frame_count"] >= args.max_frames_per_episode:
                        break
                    if vln_sim.waypoint_follower.arrived_at_goal:
                        break
                    if done_flag(vln_sim.done):
                        result["terminated"] = True
                        break
                    vln_sim.step()
                    result["frame_count"] += 1
                    if int(getattr(vln_sim, "command_dim", 0)) == 3:
                        command = vln_sim.commands[vln_sim.robot_index].detach().cpu().tolist()
                        for axis in range(3):
                            value = abs(float(command[axis]))
                            command_abs_sum[axis] += value
                            command_abs_max[axis] = max(command_abs_max[axis], value)
                        command_samples += 1
                    current_ref_distance = ref_goal_distance(vln_sim, episode)
                    if best_ref_distance - current_ref_distance >= args.astar.execution.stuck_min_progress_m:
                        best_ref_distance = current_ref_distance
                        frames_since_progress = 0
                    else:
                        frames_since_progress += 1
                    best_ref_distance, frames_since_progress = maybe_replan_if_stuck(
                        vln_sim,
                        episode,
                        result,
                        args,
                        best_ref_distance,
                        frames_since_progress,
                    )

                result["reached_goal"] = bool(vln_sim.waypoint_follower.arrived_at_goal)
                result["terminated"] = result["terminated"] or done_flag(vln_sim.done)
                result["final_ref_goal_distance_m"] = ref_goal_distance(vln_sim, episode)
                if command_samples > 0:
                    result["command_stats"] = {
                        "samples": command_samples,
                        "mean_abs_vx_vy_vw": [value / command_samples for value in command_abs_sum],
                        "max_abs_vx_vy_vw": command_abs_max,
                    }
                info = getattr(vln_sim, "info", {})
                if isinstance(info, dict):
                    result["terminations"] = jsonable(info.get("terminations"))
                result["measurements"] = collect_measurements(vln_sim)
                vln_sim.clear_waypoints()
            except Exception as exc:
                result["planning_error"] = repr(exc)
                print(f"[WORKER {args.worker_index}] ERROR {episode_label}: {exc}", flush=True)
                traceback.print_exc()
            finally:
                result["elapsed_s"] = time.time() - episode_start
                summary["episodes"].append(result)
                write_json(output_path, summary)
                print(
                    f"[WORKER {args.worker_index}] {episode_label}: "
                    f"planned={result['planned']} reached={result['reached_goal']} "
                    f"frames={result['frame_count']} elapsed={result['elapsed_s']:.1f}s",
                    flush=True,
                )
    except KeyboardInterrupt:
        print(f"[WORKER {args.worker_index}] Keyboard interrupt", flush=True)
    finally:
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(output_path, summary)
        try:
            settings.set("/app/player/playSimulations", False)
        except Exception:
            pass
        if vln_sim is not None:
            vln_sim.close()
        if app_backend_is_running() and not app_is_exiting():
            simulation_app.close(wait_for_replicator=False)


def main():
    if "--worker" in sys.argv:
        run_worker()
    else:
        run_parent()


if __name__ == "__main__":
    main()
