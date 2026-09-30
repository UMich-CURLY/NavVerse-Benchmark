import argparse
import json
import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Rebuild NavVerse navmeshes from IsaacSim-loaded stages.")
parser.add_argument("--navmesh_scene_ids", nargs="*", default=None)
parser.add_argument("--navmesh_scene_types", nargs="*", default=["vcp"])
parser.add_argument("--navmesh_audit_json", default="logs/vcp_navmesh_isaac_audit.json")
parser.add_argument("--force_navmesh_rebuild", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--visualize_navmesh_scene_id", default=None)
parser.add_argument("--visualize_navmesh_output", default="logs/navmesh_alignment.png")
parser.add_argument("--navmesh_post_load_steps", type=int, default=2)
parser.add_argument("--navmesh_enable_cameras", action="store_true")

import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
import navverse.vln_args as vln_cli_args

rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = vln_cli_args.parse_args(parser)
args.enable_contact_sensor = True
args.disable_socket_server = True
args.enable_cameras = bool(args.navmesh_enable_cameras)
args.disable_camera = not args.navmesh_enable_cameras
if args.disable_camera:
    args.on_demand_render = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from navverse.episode import VLNEpisode  # noqa: E402
from navverse.config import list_scene_ids, load_config, resolve_task_config_path  # noqa: E402
import navverse.navmesh as navmesh_utils  # noqa: E402
from navverse.sim import VLNSim  # noqa: E402
from navverse.task_generator import TaskGenerator  # noqa: E402


def collect_scene_ids(task_config):
    if args.navmesh_scene_ids:
        return args.navmesh_scene_ids
    return [
        scene_id
        for scene_id in list_scene_ids(task_config)
        if scene_id.split("_", 1)[0] in set(args.navmesh_scene_types)
    ]


def navmesh_paths(scene_id):
    navmesh_path = Path(args.scene_folder) / "navmesh" / f"{scene_id}_navmesh.bin"
    return navmesh_path, Path(f"{navmesh_path}.meta.json")


def remove_navmesh_cache(scene_id):
    navmesh_path, meta_path = navmesh_paths(scene_id)
    removed = []
    for path in (navmesh_path, meta_path):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def expected_metadata(task_generator, scene_config, navmesh_interface):
    metadata = {
        "scene_id": scene_config["scene_id"],
        "scene_type": scene_config.get("scene_type"),
        "settings": navmesh_interface.serialized_settings(),
    }
    if scene_config.get("scene_type") == "innout":
        metadata["innout_build_version"] = navmesh_utils.INNOUT_NAVMESH_BUILD_VERSION
        metadata["innout_max_dist_m"] = float(
            task_generator._sample("innout_navmesh_radius_m", 90.0)
        )
    preprocess_key = navmesh_utils.navmesh_preprocess_cache_key(
        scene_config.get("scene_type"),
        task_generator.task_config.get("navmesh_preprocess", {}),
    )
    if preprocess_key is not None:
        metadata["preprocess"] = preprocess_key
    return metadata


def audit_navmesh(scene_id, task_generator):
    scene_config = task_generator.episode_data(scene_id)
    navmesh_path, meta_path = navmesh_paths(scene_id)
    checker = navmesh_utils.NavmeshInterface(
        up_axis="Z",
        preprocess_config=task_generator.task_config.get("navmesh_preprocess", {}),
    )
    checker.update_settings(task_generator.task_config.get("navmesh", {}))
    expected = expected_metadata(task_generator, scene_config, checker)
    record = {
        "scene_id": scene_id,
        "navmesh_path": str(navmesh_path),
        "meta_path": str(meta_path),
        "navmesh_exists": navmesh_path.exists(),
        "meta_exists": meta_path.exists(),
    }
    if not navmesh_path.exists() or not meta_path.exists():
        record["ok"] = False
        record["reason"] = "missing_navmesh_or_metadata"
        return record

    try:
        metadata = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        record["ok"] = False
        record["reason"] = f"metadata_read_failed: {exc!r}"
        return record

    record["metadata"] = metadata
    record["metadata_matches"] = all(metadata.get(key) == value for key, value in expected.items())

    try:
        checker.load_navmesh(str(navmesh_path))
        vertices, triangles = checker.get_navmesh_polygons()
        record["vertex_count"] = int(vertices.shape[0])
        record["triangle_count"] = int(triangles.shape[0])
        if vertices.shape[0] > 0:
            record["navmesh_z_min"] = float(np.min(vertices[:, 2]))
            record["navmesh_z_max"] = float(np.max(vertices[:, 2]))
        preprocess_details = metadata.get("preprocess_details") or {}
        terrain_min = preprocess_details.get("terrain_top_min")
        terrain_max = preprocess_details.get("terrain_top_max")
        if terrain_min is not None and terrain_max is not None and vertices.shape[0] > 0:
            lower = float(terrain_min) - 1.0
            upper = float(terrain_max) + 6.0
            record["height_window_ok"] = lower <= record["navmesh_z_min"] <= upper and record["navmesh_z_max"] <= upper
        else:
            record["height_window_ok"] = None
        record["ok"] = bool(
            record["metadata_matches"]
            and record["vertex_count"] > 0
            and record["triangle_count"] > 0
            and record["height_window_ok"] is not False
        )
        if not record["ok"]:
            record["reason"] = "audit_checks_failed"
    except Exception as exc:
        record["ok"] = False
        record["reason"] = f"navmesh_load_failed: {exc!r}"
        record["traceback"] = traceback.format_exc()
    return record


def write_audit(records):
    path = Path(args.navmesh_audit_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def visualize_navmesh_alignment(scene_id, task_generator, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    scene_config = task_generator.episode_data(scene_id)
    navmesh_path, _ = navmesh_paths(scene_id)
    navmesh_interface = navmesh_utils.NavmeshInterface(
        up_axis="Z",
        preprocess_config=task_generator.task_config.get("navmesh_preprocess", {}),
    )
    navmesh_interface.load_navmesh(str(navmesh_path))
    vertices, triangles = navmesh_interface.get_navmesh_polygons()

    selected_prims = [
        task_generator.manager_env.scene.stage.GetPrimAtPath(path)
        for path in scene_config.get("navmesh_include", ["/World/ground/terrain"])
    ]
    selected_prims = [prim for prim in selected_prims if prim.IsValid()]
    terrain_prim = navmesh_interface._find_terrain_surface_prim(selected_prims)
    terrain_points = np.empty((0, 3), dtype=np.float32)
    if terrain_prim is not None:
        terrain_points = navmesh_interface._terrain_top_surface_points(terrain_prim)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=180)
    if terrain_points.shape[0] > 0:
        max_points = 200_000
        if terrain_points.shape[0] > max_points:
            step = int(np.ceil(terrain_points.shape[0] / max_points))
            terrain_points = terrain_points[::step]
        ax.scatter(
            terrain_points[:, 0],
            terrain_points[:, 1],
            s=0.03,
            c=terrain_points[:, 2],
            cmap="Greys",
            alpha=0.35,
            linewidths=0,
            label="terrain top mesh",
        )
    if vertices.shape[0] > 0 and triangles.shape[0] > 0:
        polys = vertices[triangles][:, :, :2]
        collection = PolyCollection(
            polys,
            facecolors=(0.0, 0.55, 1.0, 0.28),
            edgecolors=(0.0, 0.18, 0.8, 0.35),
            linewidths=0.08,
            label="navmesh",
        )
        ax.add_collection(collection)
        ax.scatter(vertices[:, 0], vertices[:, 1], s=0.2, c="#0077cc", alpha=0.35, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{scene_id} navmesh vs terrain top mesh")
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.autoscale()
    ax.grid(True, linewidth=0.25, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def rebuild_scene(scene_id, task_generator, vln_sim):
    scene_config = task_generator.episode_data(scene_id)
    if args.force_navmesh_rebuild:
        removed = remove_navmesh_cache(scene_id)
    else:
        removed = []

    task_generator.navmesh_interface = navmesh_utils.NavmeshInterface(
        up_axis="Z",
        preprocess_config=task_generator.task_config.get("navmesh_preprocess", {}),
    )
    episode = VLNEpisode(scene_config)
    vln_sim.reset(episode)
    for _ in range(max(args.navmesh_post_load_steps, 0)):
        vln_sim.step()
    task_generator.check_navmesh(scene_id)
    record = audit_navmesh(scene_id, task_generator)
    record["removed_cache_files"] = removed
    record["path"] = scene_config.get("path")
    if scene_id == args.visualize_navmesh_scene_id:
        record["visualization_path"] = visualize_navmesh_alignment(
            scene_id,
            task_generator,
            args.visualize_navmesh_output,
        )
    return record


def main():
    args.tg_config_path = str(resolve_task_config_path(args.tg_config_path, scene_type=args.navmesh_scene_types[0]))
    task_config = load_config(args.tg_config_path)
    scene_ids = collect_scene_ids(task_config)
    if not scene_ids:
        raise RuntimeError("No scene ids requested")

    task_generator = TaskGenerator(args)
    vln_sim = VLNSim(args)
    first_episode = VLNEpisode(task_generator.episode_data(scene_ids[0]))
    vln_sim.reset(first_episode)
    vln_sim.step()
    task_generator.bind_vln_sim(vln_sim)

    records = []
    start_all = time.perf_counter()
    try:
        for index, scene_id in enumerate(scene_ids, start=1):
            print(f"[ISAAC_NAVMESH] {index}/{len(scene_ids)} {scene_id}", flush=True)
            start = time.perf_counter()
            try:
                record = rebuild_scene(scene_id, task_generator, vln_sim)
            except Exception as exc:
                record = {
                    "scene_id": scene_id,
                    "ok": False,
                    "reason": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            record["index"] = index
            record["total"] = len(scene_ids)
            record["wall_s"] = time.perf_counter() - start
            records.append(record)
            write_audit(records)
            print(
                "[ISAAC_NAVMESH] result "
                + json.dumps(
                    {
                        key: record.get(key)
                        for key in [
                            "scene_id",
                            "ok",
                            "reason",
                            "wall_s",
                            "vertex_count",
                            "triangle_count",
                            "navmesh_z_min",
                            "navmesh_z_max",
                            "visualization_path",
                        ]
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        write_audit(records)
        print(
            "ISAAC_NAVMESH_DONE "
            + json.dumps(
                {
                    "audit_json": args.navmesh_audit_json,
                    "records": len(records),
                    "success": sum(1 for r in records if r.get("ok")),
                    "fail": sum(1 for r in records if not r.get("ok")),
                    "wall_s": time.perf_counter() - start_all,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
    os.kill(os.getpid(), signal.SIGKILL)
