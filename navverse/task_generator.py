from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import yaml
import os
import re
import shutil
import time
import contextlib
import numpy as np
import torch
from collections import Counter, deque
import cv2
from pxr import Semantics, Usd, UsdGeom
from navverse.episode import VLNEpisode, save_episodes
from navverse.episode_recording import EpisodeRecorder, update_episode_with_astar_path
from navverse.config import (
    list_scene_ids,
    load_config,
    parse_scene_id,
    resolve_scene_spec,
    resolve_task_config_path,
)
from navverse.vln_args import load_runtime_config
import isaacsim.core.utils.prims as prim_utils
import navverse.navmesh as navmesh_utils
from navverse.termination_cfg import DEFAULT_EPISODE_TIMEOUT_S
from navverse.path_following import calc_yaw
from navverse.vln_env_wrapper import set_robot_pose
from navverse.utils.astar import BevAStarReferencePlanner
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from threading import Lock
from queue import Queue

class TaskGenerator:
    """
    Task Generator for InNOut benchmark.

    @note runtime config is resolved per scene into scene, flow, sample, check, and module sections
    @note scene_id = scene_type + scene_name
    @note episode_id = scene_type + scene_name + episode_name
    @note sim is in charge of communication with client and keyboard
    @note env is in charge of internal environment control
    @note during testing, ui updates task_config, task_config updates env
    """
    def __init__(self, args):
        self.args = args
        config_path = resolve_task_config_path(
            getattr(args, "tg_config_path", None),
            scene_id=getattr(args, "test_scene_id", None),
        )
        self.args.tg_config_path = str(config_path)
        self.task_config = load_config(config_path)
        self.parse_config(self.task_config)
        # setup navmesh tools
        self.navmesh_interface = navmesh_utils.NavmeshInterface(
            up_axis='Z',
            preprocess_config=self.task_config.get("navmesh_preprocess", {}),
        )
        self.astar_reference_planner_cache = {}
        # State for episode checking
        self.episode_queue = deque()
        self.current_episode = None
        self.current_episode_start_time = None
        self.check_status_callback = None
        self.data_folder = os.path.join(args.scene_folder, "episode_data")
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(os.path.join(self.data_folder, "status"), exist_ok=True)
        self.target_prim_counts = Counter()
        self.reference_path_reject_counts = Counter()
        self.cfg = SimpleNamespace(scene={}, flow={}, sample={}, check={})

    def parse_config(self, task_config):
        self.scene_id_list = list_scene_ids(task_config)
        self.navmesh_preset_list = ["default"]
        self.rule_pattern_list = ["path", "name", "gr", "vc", "vc_store", "vcp", "innout"]

    def parse_scene_id(self, scene_id):
        return parse_scene_id(scene_id)

    def get_scene_spec(self, scene_id):
        return resolve_scene_spec(self.task_config, scene_id)

    def episode_data(self, scene_id):
        return self.get_scene_spec(scene_id)["scene"]

    def _set_runtime_config(self, scene_id):
        spec = self.get_scene_spec(scene_id)
        task_generation = spec.get("task_generation") or {}
        self.cfg = SimpleNamespace(
            scene=spec["scene"],
            flow=task_generation.get("flow") or {},
            sample=task_generation.get("sample") or {},
            check=task_generation.get("check") or {},
            astar=spec.get("astar") or {},
            navmesh=spec.get("navmesh") or {},
            navmesh_preprocess=spec.get("navmesh_preprocess") or {},
            sim=spec.get("sim") or {},
            waypoint_follower=spec.get("waypoint_follower") or {},
        )
        if self.vln_sim is not None:
            self.vln_sim.episode_data_namespace = self.cfg.flow.get("episode_data_namespace")
        return self.cfg

    def _episode_base_data(self):
        data = deepcopy(self.cfg.scene)
        data.setdefault("config", getattr(self.args, "tg_config_path", None))
        episode_data_namespace = self._flow("episode_data_namespace")
        if episode_data_namespace:
            data["episode_data_namespace"] = str(episode_data_namespace)
        robot_spawn_height = self.cfg.check.get("robot_spawn_height")
        if robot_spawn_height is not None:
            data["robot_spawn_height"] = robot_spawn_height
        return data

    def _flow(self, key, default=None):
        return self.cfg.flow.get(key, default)

    def _sample(self, key, default=None):
        return self.cfg.sample.get(key, default)

    def _check(self, key, default=None):
        return self.cfg.check.get(key, default)

    def _contact_check(self, key, default=None):
        return (self.cfg.check.get("contact") or {}).get(key, default)

    def _visibility_check(self, key, default=None):
        return (self.cfg.check.get("goal_visibility") or {}).get(key, default)

    def _start_settle_check(self, key, default=None):
        return (self.cfg.check.get("start_settle") or {}).get(key, default)

    def _start_settle_enabled(self):
        return bool(self._flow("start_settle_enabled", False)) and int(self._start_settle_check("check_steps", 0)) > 0

    def episode_output_path(self, scene_id):
        return Path(self._flow("episode_output_dir", "episodes")) / f"{scene_id}.json"

    def episode_data_scene_folder(self, scene_id):
        namespace = self._flow("episode_data_namespace")
        data_dir = Path(self.data_folder)
        if namespace:
            data_dir /= str(namespace)
        return data_dir / scene_id

    def _status_file(self, scene_id):
        namespace = self._flow("status_namespace")
        status_dir = Path(self.data_folder) / "status"
        if namespace:
            status_dir = status_dir / str(namespace)
        return status_dir / f"{scene_id}.json"
    
    def update_config(self, scene_config):
        scene_id = scene_config['scene_id']
        _, scene_name = self.parse_scene_id(scene_id)
        scene_section = self.task_config.setdefault("scene", {})
        scene_defaults = scene_section.setdefault("default", {})
        scene_name_config = scene_section.setdefault("scenes", {}).setdefault(scene_name, {})
        for key, value in scene_config.items():
            if key in {"episode_number", "episode_timeout", "max_episode_attempts", "rule_pattern"}:
                flow_config = scene_name_config.setdefault("task_generation", {}).setdefault("flow", {})
                flow_config[key] = value
                continue
            if key in scene_defaults:
                scene_defaults[key] = scene_config[key]
            if key in scene_name_config:
                scene_name_config[key] = scene_config[key]
        
    def save_config(self):
        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True
        with open(self.args.tg_config_path, "w") as f:
            yaml.dump(self.task_config, f, Dumper=NoAliasDumper)
    
    def bind_vln_sim(self, vln_sim):
        self.vln_sim = vln_sim
        self.env = vln_sim.env
        self.manager_env = vln_sim.manager_env

    def _set_generation_render_interval(self, interval):
        if self.args.headless and self.args.on_demand_render:
            self.env.env.cfg.sim.render_interval = 10**9
        else:
            self.env.env.cfg.sim.render_interval = interval

    def _episode_timeout_s(self, scene_type):
        return float(self._flow("episode_timeout", DEFAULT_EPISODE_TIMEOUT_S))

    def _reset_episode_quality_state(self):
        self.contact_collision_count = 0
        self.contact_collision_max_force_n = 0.0
        self.contact_collision_last_step = -10**9
        self.contact_collision_start_step = int(getattr(self.manager_env, "common_step_counter", 0) or 0)

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _contact_force_sensor(self):
        scene = getattr(self.manager_env, "scene", None)
        sensors = getattr(scene, "sensors", {}) or {}
        return sensors.get("contact_forces") if hasattr(sensors, "get") else None

    def _contact_force_array(self, sensor):
        data = getattr(sensor, "data", None)
        forces = self._to_numpy(getattr(data, "net_forces_w", None))
        if forces is None:
            forces = self._to_numpy(getattr(data, "force_matrix_w", None))
        if forces is None:
            return None

        forces = np.asarray(forces, dtype=float)
        if forces.ndim == 4:
            return np.linalg.norm(forces[0], axis=-1).max(axis=-1)
        if forces.ndim == 3:
            return forces[0]
        if forces.ndim == 2:
            return forces
        return None

    @staticmethod
    def _contact_body_indices(sensor, body_count):
        data = getattr(sensor, "data", None)
        body_names = getattr(sensor, "body_names", None) or getattr(data, "body_names", None)
        if body_names is None:
            return [0] if body_count == 1 else []
        return [
            idx for idx, name in enumerate(body_names)
            if str(name) == "body" or str(name).endswith("/body")
        ]

    def _contact_body_force_norm(self):
        sensor = self._contact_force_sensor()
        if sensor is None:
            return float("nan")

        env_forces = self._contact_force_array(sensor)
        if env_forces is None:
            return float("nan")

        body_indices = self._contact_body_indices(sensor, env_forces.shape[0])
        if not body_indices:
            return float("nan")

        selected = env_forces[body_indices]
        if selected.ndim == 1:
            return float(np.nanmax(np.abs(selected)))
        return float(np.nanmax(np.linalg.norm(selected, axis=-1)))

    def _update_contact_collision_state(self):
        force_n = self._contact_body_force_norm()
        if not np.isfinite(force_n):
            return force_n

        self.contact_collision_max_force_n = max(self.contact_collision_max_force_n, force_n)
        threshold_n = float(self._contact_check("force_threshold_n", 5.0))
        cooldown_steps = int(self._contact_check("collision_cooldown_steps", 10))
        sim_step = int(getattr(self.manager_env, "common_step_counter", 0) or 0)
        grace_steps = int(self._contact_check("collision_grace_steps", 0))
        if sim_step - int(getattr(self, "contact_collision_start_step", sim_step)) < grace_steps:
            return force_n
        if force_n >= threshold_n and sim_step - self.contact_collision_last_step >= cooldown_steps:
            self.contact_collision_count += 1
            self.contact_collision_last_step = sim_step
            print(
                f"[TG] contact collision event {self.contact_collision_count}: "
                f"body_force={force_n:.1f}N threshold={threshold_n:.1f}N"
            )
        return force_n

    def _contact_collision_limit_exceeded(self):
        max_events = int(self._contact_check("max_contact_collisions", 3))
        return self.contact_collision_count >= max_events

    def _arrived_for_task_generation(self, measurements):
        return bool(self.vln_sim.waypoint_follower.arrived_at_goal)

    def _configure_task_generation_follower(self):
        self.vln_sim.configure_waypoint_follower_profile("task_generation", reset_to_default=False)
        final_dist = float(self._check("follower_term_dist_m", 0.3))
        if final_dist > 0.0:
            self.vln_sim.waypoint_follower.term_dist = max(
                float(self.vln_sim.waypoint_follower.term_dist),
                final_dist,
            )

    def _mark_stop_for_task_generation_success(self):
        if hasattr(self.env, "set_stop_called"):
            self.env.set_stop_called(self.vln_sim.robot_index, True)
            self.env.measure_manager.update_measures()
        return self.env.measure_manager.get_measurements()

    def _episode_quality_failure(self, measurements, recorder):
        quality = recorder.quality_summary()
        max_pov_black_ratio = float(self._check("max_pov_black_frame_ratio", 0.05))
        if quality["pov_black_ratio"] > max_pov_black_ratio:
            return (
                f"pov_black_frame_ratio={quality['pov_black_ratio']:.2f} "
                f"> {max_pov_black_ratio:.2f}"
            )

        max_third_person_black_ratio = float(
            self._check("max_third_person_black_frame_ratio", 0.10)
        )
        if quality["third_person_black_ratio"] > max_third_person_black_ratio:
            return (
                f"third_person_black_frame_ratio={quality['third_person_black_ratio']:.2f} "
                f"> {max_third_person_black_ratio:.2f}"
            )

        max_collision_rate = float(self._check("max_collision_rate", 0.60))
        collision_rate = float(measurements.get("collision_rate", 0.0) or 0.0)
        if collision_rate > max_collision_rate:
            return f"collision_rate={collision_rate:.2f} > {max_collision_rate:.2f}"

        route_failure = self._distractor_route_trajectory_failure(recorder)
        if route_failure is not None:
            return route_failure

        return None

    def _distractor_route_trajectory_failure(self, recorder):
        episode = self.current_episode
        if episode.get("placenav_route_type") != "distractor_stores":
            return None
        positions = np.asarray(
            [record["body_position"][:2] for record in recorder.records],
            dtype=float,
        )
        if len(positions) == 0:
            return "distractor route has no recorded body positions"
        next_frame = 0
        for stop in episode.get("placenav_route_stops", []):
            center = np.asarray(stop.get("target_center", []), dtype=float)
            tolerance = float(stop.get("reach_tolerance_m", 0.0) or 0.0)
            if center.shape[0] < 2 or tolerance <= 0.0:
                return "distractor route stop is missing reach metadata"
            if next_frame >= len(positions):
                return f"{stop.get('role', 'route')} stop was not reached in order"
            distances = np.linalg.norm(positions[next_frame:] - center[:2], axis=1)
            reached = np.flatnonzero(distances <= tolerance)
            if len(reached) == 0:
                return (
                    f"{stop.get('role', 'route')} stop distance={float(np.min(distances)):.2f} "
                    f"> {tolerance:.2f}"
                )
            next_frame += int(reached[0]) + 1
        return None

    @contextlib.contextmanager
    def _scene_execution_args(self):
        config = self._flow("execution_args", {})
        previous = {}
        missing = set()
        for key, value in config.items():
            if hasattr(self.args, key):
                previous[key] = getattr(self.args, key)
            else:
                missing.add(key)
            setattr(self.args, key, value)
        try:
            yield
        finally:
            for key in config:
                if key in missing:
                    delattr(self.args, key)
                else:
                    setattr(self.args, key, previous[key])

    def generate_episodes(self, scene_id):
        # load scene config
        self.replay_episode_data_only = False
        self._set_runtime_config(scene_id)
        self._set_generation_render_interval(50)
        self.rule_pattern = self._flow("rule_pattern", 'name')
        self.num_episodes = int(self._flow("episode_number", 50))
        self.generate_finished = False

        # check how many episodes to generate
        # read episode file
        episode_path = self.episode_output_path(self.cfg.scene["scene_id"])
        if episode_path.exists():
            with episode_path.open('r') as f:
                episodes = json.load(f)
        else:
            episodes = []
        if self._flow("goal_visibility_enabled", False):
            visible_checked_episodes = [
                episode for episode in episodes
                if episode.get("goals", [])
                and all("visible_pixels_at_goal" in goal for goal in episode.get("goals", []))
            ]
            dropped_count = len(episodes) - len(visible_checked_episodes)
            if dropped_count > 0:
                print(f"[TG] Dropped {dropped_count} existing episodes without goal visibility metadata")
            episodes = visible_checked_episodes
        self.target_prim_counts = self._count_episode_target_prims(episodes)
        saturated_targets = sum(
            1 for count in self.target_prim_counts.values()
            if count >= self._target_prim_episode_limit()
        )
        print(
            f"[TG] Loaded {len(episodes)} existing episodes with "
            f"{len(self.target_prim_counts)} target prims ({saturated_targets} at limit)"
        )

        self._set_episode_generation_requirements(scene_id)
        self.total_samples = 0
        try:
            status_path = self._status_file(self.cfg.scene["scene_id"])
            with status_path.open("r") as f:
                status_data = json.load(f)
            self.checked_episode_attempts = int(status_data.get("checked_episode_attempts", 0) or 0)
        except Exception:
            self.checked_episode_attempts = 0
        # generate task
        total_goal_found = self.parse_scene()
        if total_goal_found == 0:
            print('[ERROR]: No goal found')
            return []
        self.generated_episodes = episodes
        if self.rule_pattern == "innout":
            self._validate_innout_target_capacity()
        print(f'[TG] Generating {self.num_episodes} episodes')
        self.vln_sim.visualize_waypoints = not self.args.headless
        self._target_first_blacklist_scene_id = self.cfg.scene["scene_id"]
        self._target_first_blacklisted_target_prims = set()
        self._generate_episodes()

    def _set_episode_generation_requirements(self, scene_id):
        scene_type, _ = self.parse_scene_id(scene_id)
        if scene_type == "vc":
            self.min_path_length = 15.0
        else:
            self.min_path_length = 5.0
        self.min_path_length = float(self._sample("min_path_length", self.min_path_length))
        default_max_path_length = 45.0 if self.rule_pattern == "vc_store" else 50.0
        self.max_path_length = self._sample("max_path_length", default_max_path_length)
        self.min_duration = 5.0
        self.timeout = self._episode_timeout_s(scene_type)
        self.max_episode_attempts = int(self._flow("max_episode_attempts", 500))

    def _load_replay_episodes(self, scene_id):
        episode_path = self.episode_output_path(scene_id)
        if not episode_path.is_file():
            raise FileNotFoundError(f"Episode file not found for replay: {episode_path}")
        with episode_path.open() as episode_file:
            episodes = [VLNEpisode(data) for data in json.load(episode_file)]
        if not episodes:
            raise RuntimeError(f"No episodes found for replay: {episode_path}")
        if any(episode.scene_id != scene_id for episode in episodes):
            raise RuntimeError(
                f"Episode file contains a different scene id: {episode_path}"
            )
        return episode_path, episodes

    def _filter_replay_episode_labels(self, scene_id, episodes):
        labels_file = getattr(self.args, "replay_episode_labels_file", None)
        if not labels_file:
            return episodes
        allowed_labels = {
            line.strip()
            for line in Path(labels_file).read_text().splitlines()
            if line.strip()
        }
        selected = [
            episode for episode in episodes if episode.episode_label in allowed_labels
        ]
        if not selected:
            raise RuntimeError(f"No episodes for {scene_id} are selected by {labels_file}")
        print(
            f"[TG] Selected {len(selected)} train-set episode(s) for {scene_id} "
            f"from {labels_file}"
        )
        return selected

    def _legacy_replay_selection(self, scene_id, episodes):
        try:
            previous_status = json.loads(self._status_file(scene_id).read_text())
        except (OSError, json.JSONDecodeError):
            previous_status = {}
        retry_ids = None
        if (
            previous_status.get("episode_data_replay_status") == "failed"
            and int(previous_status.get("episode_data_replay_total", -1))
            == len(episodes)
        ):
            retry_ids = {
                int(value)
                for value in previous_status.get("episode_data_replay_failed_ids", [])
            }
        if retry_ids is None:
            return episodes, len(episodes), []
        total = int(previous_status["episode_data_replay_total"])
        if not retry_ids:
            return [], total, list(range(total))
        retry_episodes = [
            episode for episode in episodes if int(episode.episode_id) in retry_ids
        ]
        print(f"[TG] Retrying replay episode ids: {sorted(retry_ids)}")
        return retry_episodes, total, sorted(set(range(total)) - retry_ids)

    def _initialize_replay_state(self, total, completed_ids):
        self.num_episodes = total
        self.generate_finished = False
        self.generated_episodes = []
        self.checked_episode_attempts = 0
        self.replay_episode_data_only = True
        self.replay_episode_total = total
        self.replay_episode_successes = completed_ids
        self.replay_episode_failures = []
        self.replay_episode_attempts = Counter()
        attempts = self._flow("replay_episode_max_attempts", 2)
        self.replay_episode_max_attempts = max(1, int(attempts))

    def replay_episode_data(self, scene_id):
        self._set_runtime_config(scene_id)
        self._set_generation_render_interval(50)
        self.rule_pattern = self._flow("rule_pattern", "name")
        self._set_episode_generation_requirements(scene_id)
        episode_path, episodes = self._load_replay_episodes(scene_id)
        episodes = self._filter_replay_episode_labels(scene_id, episodes)
        episodes, total, completed_ids = self._legacy_replay_selection(scene_id, episodes)
        self._initialize_replay_state(total, completed_ids)
        self.vln_sim.visualize_waypoints = not self.args.headless
        print(f"[TG] Replaying {len(episodes)} existing episodes from {episode_path}")
        if not episodes:
            self._finish_episode_data_replay()
            return
        self.check_episodes(episodes)

    def _generate_episodes(self):
        if not self._episode_generation_target_reached():
            if self._max_episode_attempts_reached():
                self._skip_scene_after_max_attempts()
                return
            max_empty_batches = max(1, int(self._flow("max_empty_sampling_batches", 1)))
            for empty_batch in range(1, max_empty_batches + 1):
                self._last_sample_failure_reason = None
                if "innout" in self.cfg.scene['scene_id'] and self._flow("episode_mix"):
                    new_episodes = self.sample_mixed_innout_episodes()
                elif "innout" in self.cfg.scene['scene_id']:
                    new_episodes = self.sample_episodes_innout(self.num_episodes)
                elif self._flow("episode_mix"):
                    new_episodes = self.sample_mixed_episodes()
                else:
                    new_episodes = self.sample_episodes(1)
                print(f'[TG] Sampled {len(new_episodes)} episodes')
                if self.generate_finished:
                    return
                if new_episodes:
                    break
                reason = getattr(self, "_last_sample_failure_reason", None) or "target_prim_limit_exhausted"
                print(
                    f"[TG] Empty sampling batch {empty_batch}/{max_empty_batches}: {reason}"
                )
            if len(new_episodes) == 0:
                print("[TG] No episodes sampled after retry budget; stopping generation")
                self.generate_finished = True
                self.save_status(
                    self.cfg.scene["scene_id"],
                    status="skipped",
                    skip_reason=reason,
                    checked_episode_attempts=self.checked_episode_attempts,
                    generated_episodes=len(self.generated_episodes),
                    target_episodes=self.num_episodes,
                )
                self.stop_generation()
                return
            self.total_samples += len(new_episodes)
            self.check_episodes(new_episodes)
        else:
            self.generate_finished = True
            self.stop_generation()
            print('[TG] Episode generation target reached!!!')

    def _episode_generation_target_reached(self):
        episode_mix = self._flow("episode_mix")
        if not episode_mix:
            return len(self.generated_episodes) >= self.num_episodes
        for task_type, target_count in episode_mix.items():
            target_count = int(target_count)
            if target_count <= 0:
                continue
            current_count = sum(ep.get("task_type") == task_type for ep in self.generated_episodes)
            if current_count < target_count:
                return False
        return True

    def _max_episode_attempts_reached(self):
        return self.checked_episode_attempts >= self.max_episode_attempts

    def _record_episode_attempt(self, stage, reason=None, skip_on_reach=False):
        self.checked_episode_attempts += 1
        status = {
            "checked_episode_attempts": self.checked_episode_attempts,
            "max_episode_attempts": self.max_episode_attempts,
            "last_attempt_stage": stage,
        }
        if reason is not None:
            status["last_attempt_reason"] = reason
        self.save_status(self.cfg.scene["scene_id"], **status)
        print(
            f"[TG] failed checked episodes: {self.checked_episode_attempts}/{self.max_episode_attempts} "
            f"stage={stage}"
        )
        if skip_on_reach and self._max_episode_attempts_reached():
            self._skip_scene_after_max_attempts()
            return False
        return True

    def _skip_scene_after_max_attempts(self):
        print(
            f"[TG] Skipping {self.cfg.scene['scene_id']} after "
            f"{self.checked_episode_attempts} failed checked episodes; "
            f"generated {len(self.generated_episodes)}/{self.num_episodes}"
        )
        self.generate_finished = True
        self.save_status(
            self.cfg.scene["scene_id"],
            status="skipped",
            skip_reason="max_episode_attempts_exceeded",
            checked_episode_attempts=self.checked_episode_attempts,
            generated_episodes=len(self.generated_episodes),
            target_episodes=self.num_episodes,
        )
        self.stop_generation()
    
    def stop_generation(self):
        if self.check_status_callback is not None:
            self.vln_sim.remove_callback('step_finished', self.check_status_callback)
            self.check_status_callback = None
            print('[TG] Generation stopped')
        self.vln_sim.clear_waypoints()
        self._set_generation_render_interval(5)

    def _timeout_summary(self, measurements):
        body_pos, _ = self.env.get_body_pose()
        closest_goal_idx = int(measurements.get("closest_goal", 0) or 0)
        goal = self.current_episode["goals"][closest_goal_idx]
        reference_path = np.asarray(goal.get("reference_path", []), dtype=float)
        target = reference_path[-1].tolist() if len(reference_path) else None
        waypoint_idx = getattr(self.vln_sim.waypoint_follower, "current_wp_idx", None)
        waypoint_count = len(getattr(self.vln_sim, "waypoints", []) or [])
        current_waypoint = None
        if waypoint_idx is not None and 0 <= waypoint_idx < waypoint_count:
            current_waypoint = self.vln_sim.waypoints[waypoint_idx]
        print(
            "[TG] timeout summary: "
            f"pos={[round(float(x), 3) for x in body_pos[:3]]}, "
            f"target={None if target is None else [round(float(x), 3) for x in target[:3]]}, "
            f"waypoint_idx={waypoint_idx}/{waypoint_count}, "
            f"current_waypoint={current_waypoint}, "
            f"distance_to_goal={measurements.get('distance_to_goal'):.2f}, "
            f"path_length={measurements.get('path_length'):.2f}, "
            f"covered_area={measurements.get('covered_area'):.2f}, "
            f"collision_rate={measurements.get('collision_rate'):.2f}"
        )

    def _innout_attempt_summary(self, reason, measurements):
        try:
            body_pos, body_quat = self.env.get_body_pose()
            plan = self._innout_plan()
            door_xy = np.asarray(plan["door_center_xy"], dtype=float)
            outward = np.asarray(plan["_outward_normal_xy"], dtype=float)
            tangent = np.asarray([-outward[1], outward[0]], dtype=float)
            door_offset = np.asarray(body_pos[:2], dtype=float) - door_xy
            euler_deg = R.from_quat(body_quat).as_euler("xyz", degrees=True)
            waypoint_idx = getattr(self.vln_sim.waypoint_follower, "current_wp_idx", None)
            waypoints = getattr(self.vln_sim, "waypoints", []) or []
            current_waypoint = None
            if waypoint_idx is not None and 0 <= waypoint_idx < len(waypoints):
                current_waypoint = [round(float(value), 3) for value in waypoints[waypoint_idx]]
            target_prim = self._episode_target_prim(self.current_episode)
            print(
                "[TG] InNOut failure summary: "
                f"reason={reason}, target={target_prim}, "
                f"pos={[round(float(value), 3) for value in body_pos[:3]]}, "
                f"door_signed_m={float(np.dot(door_offset, outward)):.3f}, "
                f"door_lateral_m={float(np.dot(door_offset, tangent)):.3f}, "
                f"tilt_deg={float(max(abs(euler_deg[0]), abs(euler_deg[1]))):.2f}, "
                f"waypoint_idx={waypoint_idx}/{len(waypoints)}, "
                f"current_waypoint={current_waypoint}, "
                f"distance_to_goal={float(measurements.get('distance_to_goal', np.nan)):.2f}, "
                f"collision_rate={float(measurements.get('collision_rate', np.nan)):.2f}, "
                f"contact_events={getattr(self, 'contact_collision_count', 0)}, "
                f"max_body_force={getattr(self, 'contact_collision_max_force_n', 0.0):.1f}N"
            )
        except Exception as exc:
            print(f"[TG] InNOut failure summary unavailable: {exc}")

    def check_episodes(self, episodes):
        """
        Check episodes asynchronously using callbacks.
        Uses a state machine pattern with episode queue.
        """
        self.episode_queue = deque(episodes)
        self._check_episode()

    def _finish_episode_data_replay(self):
        failed_ids = sorted(set(self.replay_episode_failures))
        replayed_ids = sorted(set(self.replay_episode_successes))
        replay_status = "success" if not failed_ids and len(replayed_ids) == self.replay_episode_total else "failed"
        self.save_status(
            self.cfg.scene["scene_id"],
            episode_data_replay_status=replay_status,
            episode_data_replay_total=self.replay_episode_total,
            episode_data_replay_completed=len(replayed_ids),
            episode_data_replay_failed_ids=failed_ids,
        )
        self.generate_finished = True
        if replay_status == "success":
            print(
                f"[TG] Episode data replay completed successfully: "
                f"{len(replayed_ids)}/{self.replay_episode_total}"
            )
        else:
            print(
                f"[TG] Episode data replay failed: completed={len(replayed_ids)}/"
                f"{self.replay_episode_total}, failed_ids={failed_ids}"
            )

    def _retry_or_fail_replay_episode(self, data_folder, reason):
        shutil.rmtree(data_folder, ignore_errors=True)
        episode_id = int(self.current_episode.episode_id)
        self.replay_episode_attempts[episode_id] += 1
        attempts = self.replay_episode_attempts[episode_id]
        if attempts < self.replay_episode_max_attempts:
            print(
                f"[TG] Requeue replay episode {episode_id} after {reason}: "
                f"attempt={attempts}/{self.replay_episode_max_attempts}"
            )
            self.episode_queue.append(self.current_episode.copy())
        else:
            print(
                f"[TG] Replay episode {episode_id} failed after {attempts} attempts: {reason}"
            )
            self.replay_episode_failures.append(episode_id)
        self._check_episode()

    def _create_episode_recorder(self):
        data_folder = self.episode_data_scene_folder(
            self.current_episode.scene_id
        ) / f"episode_{self.current_episode.episode_id}"
        recorder = EpisodeRecorder(
            self.vln_sim,
            data_folder,
            self.current_episode,
            self.args.scene_folder,
            clear_existing=True,
        )
        return data_folder, recorder

    def _finalize_episode_recorder(self, recorder, data_folder):
        try:
            recorder.finalize(
                astar_world_path=getattr(
                    self.vln_sim, "last_astar_world_path", None
                ),
                astar_plan_info=getattr(self.vln_sim, "last_astar_plan_info", None),
            )
        except Exception as exc:
            print(f"[TG] Recorder finalize failed: {type(exc).__name__}: {exc}")
            return False
        return True

    def _save_checked_episode(
        self,
        recorder,
        data_folder,
        replay_only,
        using_astar_execution,
    ):
        if using_astar_execution and not replay_only:
            update_episode_with_astar_path(self.current_episode, self.vln_sim)
        if not self._finalize_episode_recorder(recorder, data_folder):
            return False
        if replay_only:
            self.replay_episode_successes.append(int(self.current_episode.episode_id))
            return True
        self._clear_innout_target_failure(
            self._episode_target_prim(self.current_episode)
        )
        self.generated_episodes.append(self.current_episode)
        self.target_prim_counts = self._count_episode_target_prims(
            self.generated_episodes
        )
        episode_path = self.episode_output_path(self.cfg.scene["scene_id"])
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        save_episodes(self.generated_episodes, episode_path)
        print(f"[TG] {len(self.generated_episodes)} episodes saved to {episode_path}")
        return True

    def _handle_checked_episode_completion(
        self,
        success,
        failure_reason,
        measurements,
        recorder,
        data_folder,
        replay_only,
        using_astar_execution,
    ):
        if success and self._save_checked_episode(
            recorder, data_folder, replay_only, using_astar_execution
        ):
            return False
        failure_reason = failure_reason or "recorder_finalize_failed"
        if replay_only:
            self._retry_or_fail_replay_episode(data_folder, failure_reason)
            return True
        self._record_innout_check_failure(
            self.current_episode,
            failure_reason,
            measurements,
        )
        self._record_episode_attempt("check_episode_failed", skip_on_reach=False)
        shutil.rmtree(data_folder, ignore_errors=True)
        return False
    
    def _check_episode(self):
        """Start checking the next episode in the queue."""
        replay_only = bool(getattr(self, "replay_episode_data_only", False))
        if len(self.episode_queue) == 0 and replay_only:
            self._finish_episode_data_replay()
            return
        if len(self.episode_queue) == 0 or self._episode_generation_target_reached():
            print('[TG] checking finished')
            self._generate_episodes()
            return
        if self._max_episode_attempts_reached():
            self._skip_scene_after_max_attempts()
            return
        
        self.current_episode = self.episode_queue.popleft()
        if not replay_only:
            all_episode_ids = set([x["episode_id"] for x in self.generated_episodes])
            for episode_id in range(len(self.generated_episodes)+1):
                if episode_id not in all_episode_ids:
                    self.current_episode['episode_id'] = episode_id
                    break
        self.current_episode_start_time = time.time()
        print('======================')
        print(f'[TG] Checking episode {self.current_episode.episode_id}')
        print(f'[TG] target: {self.current_episode["objnav"]}, path length: {self.current_episode["goals"][0]["path_length"]:.2f}m')

        data_folder, recorder = self._create_episode_recorder()
        print(f'[TG] Saving data to {data_folder}')
        using_astar_execution = False
        self._reset_episode_quality_state()

        # Create new callback for this episode
        def check_status():
            if self.vln_sim.current_episode.episode_label != self.current_episode.episode_label:
                return
            self.vln_sim.waiting_for_first_action = False
            check_done = False
            success = False
            failure_reason = None
            img_saving_interval = 1
            # save data for vln
            if self.vln_sim.obs_index%img_saving_interval == 0:
                recorder.capture_frame()
            # check if episode is done
            self.env.measure_manager.update_measures()
            measurements = self.env.measure_manager.get_measurements()
            self._update_contact_collision_state()
            if (
                not replay_only
                and self.vln_sim.obs_index == 0
                and measurements["distance_to_goal"] < self.min_path_length
            ):
                print(f'[TG] episode {self.current_episode.episode_id} starting position is too close to goal')
                failure_reason = "start_too_close"
                check_done = True
            elif self._contact_collision_limit_exceeded():
                print(
                    f"[TG] episode {self.current_episode.episode_id} exceeded contact collision limit: "
                    f"events={self.contact_collision_count}, max_body_force={self.contact_collision_max_force_n:.1f}N"
                )
                failure_reason = "contact_collision_limit"
                check_done = True
            elif self._arrived_for_task_generation(measurements):
                measurements = self._mark_stop_for_task_generation_success()
                print("[TG] Measures: ", ", ".join([f"{k}={v:.2f}" for k, v in measurements.items()]))
                # check episode quality by metrics
                if measurements["success"] != 1.0:
                    print(f'[TG] episode {self.current_episode.episode_id} failed: follower reached final waypoint but success=0')
                    failure_reason = "follower_endpoint_failed"
                elif not replay_only and measurements["path_length"] < self.min_path_length:
                    print(f'[TG] episode {self.current_episode.episode_id} completed but path length is too short')
                    failure_reason = "path_too_short"
                elif quality_failure := self._episode_quality_failure(measurements, recorder):
                    print(f'[TG] episode {self.current_episode.episode_id} rejected by quality gate: {quality_failure}')
                    failure_reason = f"quality_gate:{quality_failure}"
                else:
                    print(f'[TG] episode {self.current_episode.episode_id} completed successfully')
                    success = True
                check_done = True
            elif measurements["sim_duration"] > self.timeout:
                print(f'[TG] episode {self.current_episode.episode_id} timed out')
                self._timeout_summary(measurements)
                failure_reason = "timeout"
                check_done = True
            elif "terminations" in self.vln_sim.info:
                termination_reason = self.vln_sim.info["terminations"]["termination_reason"]
                print(f'[TG] episode {self.current_episode.episode_id} terminated due to {termination_reason}')
                failure_reason = f"termination:{termination_reason}"
                check_done = True
            # episode is done, check if it is successful
            if check_done:
                self.vln_sim.remove_callback('step_finished', self.check_status_callback)
                self.check_status_callback = None
                if self._handle_checked_episode_completion(
                    success,
                    failure_reason,
                    measurements,
                    recorder,
                    data_folder,
                    replay_only,
                    using_astar_execution,
                ):
                    return
                self._check_episode()
        
        # Reset and start the episode
        print(f'[TG] Reset episode {self.current_episode.episode_id}')
        self.current_episode["episode_timeout"] = self.timeout
        self.vln_sim.reset(self.current_episode)
        self._configure_task_generation_follower()
        self.env.terminations_cfg.time_out.params["max_time"] = self.timeout
        if hasattr(self.env.terminations_cfg, "stuck"):
            self.env.terminations_cfg.stuck.params["max_time"] = 5.0
            self.env.terminations_cfg.stuck.params["max_tdist_thresholdime"] = 0.04
        print('[TG] Set reference waypoints')
        try:
            with self._scene_execution_args():
                self.vln_sim.set_ref_waypoints(self.current_episode)
                using_astar_execution = getattr(self.vln_sim, "waypoint_path_planner", None) == "astar"
                self._configure_task_generation_follower()
        except Exception as exc:
            print(f'[TG] episode {self.current_episode.episode_id} waypoint planning failed: {exc}')
            if replay_only:
                self._retry_or_fail_replay_episode(data_folder, "waypoint_planning_failed")
                return
            self._record_episode_attempt("check_episode_failed", reason="waypoint_planning_failed", skip_on_reach=False)
            shutil.rmtree(data_folder, ignore_errors=True)
            self._check_episode()
            return
        self.vln_sim.waiting_for_first_action = False
        self.check_status_callback = check_status
        self.vln_sim.add_callback('step_finished', check_status)

    def parse_scene(self):
        # find target prims
        self.prim_list = [x for x in self.manager_env.scene.stage.Traverse()]
        # [x for x in self.prim_list if str(x.GetPath()).startswith("/World/ground/terrain/Brownstone03/Geometry/Specialty_Equipment/")]
        print(f'Loaded {len(self.prim_list)} prims')
        self.goal_dict = {}
        self.goal_dict_by_task = {}
        def objaverse_goal_from_path(prim_path_str):
            match_result = re.search(r"/Objaverse/([^/]*?)$", prim_path_str)
            if not match_result:
                return None
            return "_".join(match_result.group(1).split("_")[:-1])

        def store_goal_from_path(prim_path_str):
            match_result = re.search(r"/store/([^/]*)$", prim_path_str)
            if not match_result:
                return None
            return re.sub(r"_\d+$", "", match_result.group(1))

        for goal, goal_rule in self._sample("goal_rules", {}).items():
            if self.rule_pattern == "path":
                # Convert prim path to string and normalize path separators
                goal_prim = []
                for x in self.prim_list:
                    prim_path_str = str(x.GetPrimPath())
                    if re.search(goal_rule, prim_path_str):
                        goal_prim.append(x)
                        print(f"  Matched: {prim_path_str}")
            elif self.rule_pattern == "name":
                goal_prim = [x for x in self.prim_list if re.search(goal_rule, x.GetName())]
            self.goal_dict[goal] = {
                'prim': goal_prim,
            }
            print(f'[TG] Found {len(goal_prim)} {goal}')
        if self.rule_pattern == "gr":
            for x in self.prim_list:
                prim_path_str = str(x.GetPrimPath())
                match_result = re.search(r"/(Furnitures|Animation)/([^/]*?)/(model_[^/]*)/Instance$", prim_path_str)
                if match_result:
                    goal = match_result.group(2)
                    if goal in ["other", "box", "bottle", "person"]:
                        continue
                    self.goal_dict.setdefault(goal, {"prim": []})["prim"].append(x)
            new_goal_dict = {}
            for goal, goal_item in self.goal_dict.items():
                if len(goal_item['prim'])<=8:
                    new_goal_dict[goal] = goal_item
            self.goal_dict = new_goal_dict
        elif self.rule_pattern == "vc":
            for x in self.prim_list:
                prim_path_str = str(x.GetPrimPath())
                goal = objaverse_goal_from_path(prim_path_str)
                if goal:
                    self.goal_dict.setdefault(goal, {"prim": []})["prim"].append(x)
        elif self.rule_pattern == "vc_store":
            for x in self.prim_list:
                prim_path_str = str(x.GetPrimPath())
                goal = store_goal_from_path(prim_path_str)
                if goal:
                    self.goal_dict.setdefault(goal, {"prim": []})["prim"].append(x)
        elif self.rule_pattern == "vcp":
            store_goal_dict = {}
            obj_goal_dict = {}
            for x in self.prim_list:
                prim_path_str = str(x.GetPrimPath())
                store_goal = store_goal_from_path(prim_path_str)
                if store_goal:
                    store_goal_dict.setdefault(store_goal, {"prim": []})["prim"].append(x)
                    continue
                obj_goal = objaverse_goal_from_path(prim_path_str)
                if obj_goal:
                    obj_goal_dict.setdefault(obj_goal, {"prim": []})["prim"].append(x)
            self.goal_dict_by_task = {
                "placenav": store_goal_dict,
                "objnav": obj_goal_dict,
            }
            self.goal_dict = {**store_goal_dict, **obj_goal_dict}
        elif self.rule_pattern == "innout":
            store_goal_dict = {}
            obj_goal_dict = {}
            for x in self.prim_list:
                prim_path_str = str(x.GetPrimPath())
                store_goal = store_goal_from_path(prim_path_str)
                if store_goal:
                    store_goal_dict.setdefault(store_goal, {"prim": []})["prim"].append(x)
                    continue
                obj_goal = objaverse_goal_from_path(prim_path_str)
                if obj_goal:
                    obj_goal_dict.setdefault(obj_goal, {"prim": []})["prim"].append(x)
            self.goal_dict_by_task = {
                "placenav": store_goal_dict,
                "objnav": obj_goal_dict,
            }
            self.goal_dict = {**store_goal_dict, **obj_goal_dict}
        total_goal_found = 0
        for goal, goal_item in self.goal_dict.items():
            print(f"  Found {len(goal_item['prim'])} {goal}")
            total_goal_found += len(goal_item['prim'])
        print(f'[TG] Total goal found: {total_goal_found}')
        return total_goal_found
    
    def check_navmesh(self, scene_id):
        scene_spec = self.get_scene_spec(scene_id)
        scene_config = scene_spec["scene"]
        navmesh_interface = self.navmesh_interface
        preprocess_config = scene_spec.get("navmesh_preprocess") or {}
        navmesh_interface.preprocess_config = preprocess_config
        navmesh_interface.update_settings(scene_spec.get("navmesh") or {})
        scene_folder = Path(self.args.scene_folder)
        os.makedirs(scene_folder / "navmesh", exist_ok=True)
        navmesh_path = str(scene_folder / f"navmesh/{scene_config['scene_id']}_navmesh.bin")
        expected_metadata = {
            "scene_id": scene_config["scene_id"],
            "scene_type": scene_config.get("scene_type"),
            "settings": navmesh_interface.serialized_settings(),
        }
        if scene_config.get("scene_type") == "innout":
            expected_metadata["innout_build_version"] = navmesh_utils.INNOUT_NAVMESH_BUILD_VERSION
            expected_metadata["innout_max_dist_m"] = float(
                self._sample("innout_navmesh_radius_m", 90.0)
            )
        preprocess_key = navmesh_utils.navmesh_preprocess_cache_key(scene_config.get("scene_type"), preprocess_config)
        if preprocess_key is not None:
            expected_metadata["preprocess"] = preprocess_key
        meta_path = f"{navmesh_path}.meta.json"
        navmesh_cache_valid = False
        legacy_navmesh_cache = os.path.exists(navmesh_path) and not os.path.exists(meta_path)
        if legacy_navmesh_cache:
            print("[TG] Navmesh metadata is missing; rebuilding navmesh cache.")
        elif os.path.exists(navmesh_path) and os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    navmesh_metadata = json.load(f)
                navmesh_cache_valid = all(navmesh_metadata.get(k) == v for k, v in expected_metadata.items())
            except (OSError, json.JSONDecodeError):
                navmesh_cache_valid = False
        if os.path.exists(navmesh_path) and navmesh_cache_valid:
            print("[TG] Navmesh found, loading...")
            navmesh_interface.load_navmesh(navmesh_path)
            navmesh_interface.configure_building_exclusion(
                scene_config.get("navmesh_include", ["/World/ground/terrain"]),
                self.manager_env.scene.stage,
                scene_type=scene_config.get("scene_type"),
            )
        else:
            if os.path.exists(navmesh_path):
                print("[TG] Navmesh cache metadata is stale or missing, rebuilding...")
            else:
                print("[TG] Navmesh not found, building...")
            selected_paths = scene_config.get("navmesh_include", ["/World/ground/terrain"])
            missing_paths = [path for path in selected_paths if not self.manager_env.scene.stage.GetPrimAtPath(path).IsValid()]
            if missing_paths:
                raise RuntimeError(f"Navmesh input prims not found for {scene_config['scene_id']}: {missing_paths}")
            print(f"[TG] Navmesh input prims: {selected_paths}")
            start_time = time.time()
            if scene_config.get("scene_type") == "innout":
                navmesh_interface.build_navmesh_innout(
                    selected_paths,
                    scene_config.get("navmesh_exclude", []),
                    self.manager_env.scene.stage,
                    max_dist=float(self._sample("innout_navmesh_radius_m", 90.0)),
                    preprocess_config=preprocess_config,
                )
            else:
                navmesh_interface.setup_navmesh(
                    selected_paths,
                    scene_config.get("navmesh_exclude", []),
                    self.manager_env.scene.stage,
                    scene_type=scene_config.get("scene_type"),
                    preprocess_config=preprocess_config,
                )
                navmesh_interface.build_navmesh()
            if not navmesh_interface.built:
                raise RuntimeError(f"Failed to build navmesh for {scene_config['scene_id']}")
            end_time = time.time()
            print(f"[TG] Navmesh build time: {end_time - start_time:.2f} seconds")
            navmesh_interface.save_navmesh(navmesh_path, metadata=expected_metadata)

    def _trim_store_reference_path(self, path, navmesh_interface, min_clearance=0.8):
        if self.rule_pattern != "vc_store" or len(path) <= 2:
            return path
        reference_path = path.copy()
        while len(reference_path) > 2:
            try:
                clearance = navmesh_interface.distance_to_wall(reference_path[-1])
            except Exception:
                break
            if np.isfinite(clearance) and clearance >= min_clearance:
                break
            reference_path = reference_path[:-1]
        return reference_path

    def _trim_reference_path_to_valid_endpoint(self, path, navmesh_interface, dist_to_start):
        reference_path = self._trim_store_reference_path(path, navmesh_interface)
        if not self._flow("goal_visibility_enabled", False):
            return reference_path
        if len(reference_path) <= 1:
            return np.asarray([])
        for end_idx in range(len(reference_path) - 1, 0, -1):
            candidate = reference_path[end_idx]
            if not self.is_valid_start_point(candidate, navmesh_interface):
                continue
            candidate_path = reference_path[:end_idx + 1]
            path_length = np.linalg.norm(candidate_path[1:] - candidate_path[:-1], axis=1).sum() + dist_to_start
            if path_length >= self.min_path_length:
                return candidate_path
        return np.asarray([])

    def _trajectory_planner_name(self, *, for_sampling=False):
        if for_sampling:
            sampling_planner = self._sample("sampling_trajectory_planner")
            if sampling_planner:
                return sampling_planner
            if self._flow("rule_pattern") == "vcp":
                return "navmesh"
        override = getattr(self.args, "trajectory_planner", None)
        if override:
            return override
        return self._sample("trajectory_planner", "navmesh")

    def _runtime_config_path(self):
        explicit_args = getattr(self.args, "_explicit_args", set())
        if "config" in explicit_args:
            return getattr(self.args, "config", "configs/default.yaml")
        return self.cfg.scene.get("config", getattr(self.args, "config", "configs/default.yaml"))

    def _astar_config(self, scene_id=None):
        if self.vln_sim is not None and hasattr(self.vln_sim, "_astar_config"):
            self.vln_sim.runtime_config_path = self._runtime_config_path()
            return self.vln_sim._astar_config(scene_id)
        runtime_config = load_runtime_config(
            config_path=self._runtime_config_path(),
            dotlist_overrides=getattr(self.args, "_config_dotlist_overrides", []) or [],
        )
        self.args.astar = runtime_config.astar
        return self.args.astar

    def _navmesh_preprocess_config(self):
        return self.task_config.get("navmesh_preprocess", {})

    def _get_astar_reference_planner(self, scene_id):
        scene_astar = self.cfg.astar
        astar_config = self._astar_config(scene_id)
        bev_file_name = scene_astar.get("bev_file_name", "nav_bev_map")
        bev_map_path = self.episode_data_scene_folder(scene_id) / f"{bev_file_name}.npz"
        if not bev_map_path.exists() and bev_file_name != "bev_map":
            bev_map_path = self.episode_data_scene_folder(scene_id) / "bev_map.npz"
        if not bev_map_path.exists():
            raise FileNotFoundError(f"Missing BEV map for A* planning: {bev_map_path}")
        cache_key = (
            str(bev_map_path),
            float(scene_astar.get("max_climb_height", astar_config.planner.max_climb_height_m)),
            float(scene_astar.get("robot_width", getattr(astar_config.planner, "robot_width_m", getattr(self.manager_env.cfg, "robot_width", 0.5)))),
            float(scene_astar.get("robot_height", getattr(astar_config.planner, "robot_height_m", getattr(self.manager_env.cfg, "robot_height", 0.9)))),
            float(scene_astar.get("step_size", astar_config.planner.step_size_m)),
            int(scene_astar.get("step_size_yaw", astar_config.planner.step_size_yaw_deg)),
            float(scene_astar.get("reach_threshold", astar_config.planner.reach_threshold_m)),
            float(scene_astar.get("reach_yaw", astar_config.planner.reach_yaw_deg)),
            float(scene_astar.get("clearance_m", astar_config.planner.clearance_m)),
            float(scene_astar.get("preferred_clearance_m", astar_config.planner.preferred_clearance_m)),
            float(scene_astar.get("clearance_cost", astar_config.planner.clearance_cost)),
            int(scene_astar.get("min_obstacle_area_px", astar_config.planner.min_obstacle_area_px)),
        )
        if cache_key not in self.astar_reference_planner_cache:
            self.astar_reference_planner_cache[cache_key] = BevAStarReferencePlanner(
                bev_map_path,
                max_climb_height=cache_key[1],
                robot_width=cache_key[2],
                robot_height=cache_key[3],
                step_size=cache_key[4],
                step_size_yaw=cache_key[5],
                reach_threshold=cache_key[6],
                reach_yaw=cache_key[7],
                clearance_m=cache_key[8],
                preferred_clearance_m=cache_key[9],
                clearance_cost=cache_key[10],
                min_obstacle_area_px=cache_key[11],
            )
        return self.astar_reference_planner_cache[cache_key]

    def _find_reference_path(self, start_pos, goal_pos, navmesh_interface, goal_radius=None):
        planner_name = self._trajectory_planner_name(for_sampling=True)
        self._last_reference_path_planner = planner_name
        self._last_reference_path_plan_info = {}
        if planner_name == "navmesh":
            return np.asarray(navmesh_interface.find_paths(start_pos, goal_pos), dtype=float)
        if planner_name != "astar":
            raise ValueError(f"Unsupported trajectory_planner: {planner_name}")

        guide_path = np.asarray(navmesh_interface.find_paths(start_pos, goal_pos), dtype=float)
        if len(guide_path) < 2:
            return guide_path

        if self.vln_sim is not None and hasattr(self.vln_sim, "generate_astar_path"):
            try:
                self.vln_sim.generate_astar_path(
                    self.cfg.scene["scene_id"],
                    guide_path,
                    goal_position=goal_pos,
                    goal_radius=goal_radius,
                )
            except RuntimeError as exc:
                self._last_reference_path_plan_info = {
                    "source": "vln_sim_astar",
                    "fallback": False,
                    "error": repr(exc),
                }
                return np.empty((0, 3), dtype=float)
            astar_path = np.asarray(getattr(self.vln_sim, "last_astar_world_path", None) or [], dtype=float)
            self._last_reference_path_plan_info = deepcopy(getattr(self.vln_sim, "last_astar_plan_info", {}) or {})
            if self._last_reference_path_plan_info.get("fallback"):
                return np.empty((0, 3), dtype=float)
            if len(astar_path) >= 2:
                return astar_path

        scene_astar = self.cfg.astar
        astar_config = self._astar_config(self.cfg.scene["scene_id"])
        astar_planner = self._get_astar_reference_planner(self.cfg.scene["scene_id"])
        astar_path = astar_planner.plan_reference_path(
            start_pos,
            goal_pos,
            time_budget=float(scene_astar.get("time_budget", astar_config.global_planner.time_budget_s)),
        )
        self._last_reference_path_plan_info = {"source": "task_generator_direct_astar", "fallback": False}
        return astar_path

    def _start_sampling_bbox(self):
        scene_id = self.cfg.scene["scene_id"]
        if getattr(self, "_start_sampling_bbox_scene_id", None) != scene_id:
            self._start_sampling_bbox_value = self.env.get_prim_bounding_box("/World/ground/terrain")
            self._start_sampling_bbox_scene_id = scene_id
        return self._start_sampling_bbox_value

    def get_innout_indoor_bbox(self, scene_id):
        if "innout" not in str(scene_id):
            return None
        indoor_prim = self.manager_env.scene.stage.GetPrimAtPath("/World/ground/terrain/start_result_navigation")
        if not indoor_prim.IsValid():
            return None
        return np.asarray(self.env.get_prim_bounding_box(indoor_prim.GetPrimPath()), dtype=float)

    def _innout_plan(self):
        scene_id = self.cfg.scene["scene_id"]
        if getattr(self, "_innout_plan_scene_id", None) == scene_id:
            return self._innout_plan_value

        scene_path = Path(self.cfg.scene["path"])
        if not scene_path.is_absolute():
            scene_path = Path(self.args.scene_folder) / scene_path
        plan_path = scene_path.parent / "innout_final_plan.json"
        try:
            plan = json.loads(plan_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read InNOut placement plan: {plan_path}") from exc

        required = {
            "door_center_xy",
            "door_width",
            "interior_scene_bbox_min",
            "interior_scene_bbox_max",
            "outward_normal_xy",
            "room_effective_scale",
            "room_translate",
            "room_yaw_deg",
        }
        missing = sorted(required - set(plan))
        if missing:
            raise RuntimeError(f"InNOut placement plan is missing fields {missing}: {plan_path}")
        outward = np.asarray(plan["outward_normal_xy"], dtype=float)
        outward_norm = float(np.linalg.norm(outward))
        if outward.shape != (2,) or outward_norm < 1e-6:
            raise RuntimeError(f"InNOut placement plan has an invalid outward normal: {plan_path}")
        plan["_outward_normal_xy"] = outward / outward_norm
        plan["_plan_path"] = str(plan_path)
        self._innout_plan_scene_id = scene_id
        self._innout_plan_value = plan
        return plan

    @staticmethod
    def _innout_room_local_xy(point, plan):
        yaw = np.deg2rad(float(plan["room_yaw_deg"]))
        inverse_rotation = np.asarray(
            [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]],
            dtype=float,
        )
        translated = np.asarray(point, dtype=float)[:2] - np.asarray(plan["room_translate"], dtype=float)[:2]
        return inverse_rotation @ translated / float(plan["room_effective_scale"])

    @classmethod
    def _point_in_innout_room_xy(cls, point, plan, margin_m=0.0):
        local_xy = cls._innout_room_local_xy(point, plan)
        local_min = np.asarray(plan["interior_scene_bbox_min"], dtype=float)[:2]
        local_max = np.asarray(plan["interior_scene_bbox_max"], dtype=float)[:2]
        local_margin = float(margin_m) / float(plan["room_effective_scale"])
        return bool(np.all(local_xy >= local_min + local_margin) and np.all(local_xy <= local_max - local_margin))

    @classmethod
    def _point_outside_innout_room_xy(cls, point, plan, margin_m=0.0):
        local_xy = cls._innout_room_local_xy(point, plan)
        local_min = np.asarray(plan["interior_scene_bbox_min"], dtype=float)[:2]
        local_max = np.asarray(plan["interior_scene_bbox_max"], dtype=float)[:2]
        local_margin = float(margin_m) / float(plan["room_effective_scale"])
        return bool(np.any(local_xy < local_min - local_margin) or np.any(local_xy > local_max + local_margin))

    @staticmethod
    def _innout_door_signed_distance(point, plan):
        point_xy = np.asarray(point, dtype=float)[:2]
        door_xy = np.asarray(plan["door_center_xy"], dtype=float)
        return float((point_xy - door_xy) @ plan["_outward_normal_xy"])

    @staticmethod
    def _innout_room_center_xy(plan):
        local_min = np.asarray(plan["interior_scene_bbox_min"], dtype=float)[:2]
        local_max = np.asarray(plan["interior_scene_bbox_max"], dtype=float)[:2]
        local_center = 0.5 * (local_min + local_max) * float(plan["room_effective_scale"])
        yaw = np.deg2rad(float(plan["room_yaw_deg"]))
        rotation = np.asarray(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=float,
        )
        return rotation @ local_center + np.asarray(plan["room_translate"], dtype=float)[:2]

    def _innout_storefront_plan(self):
        scene_id = self.cfg.scene["scene_id"]
        if getattr(self, "_innout_storefront_plan_scene_id", None) == scene_id:
            return self._innout_storefront_plan_value

        plan_path = Path(self._innout_plan()["_plan_path"]).with_name("innout_storefront_plan.json")
        try:
            raw_plan = json.loads(plan_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read InNOut storefront plan: {plan_path}") from exc
        placements = raw_plan.get("placements")
        if not isinstance(placements, list):
            raise RuntimeError(f"InNOut storefront plan has no placements list: {plan_path}")
        try:
            placements_by_index = {int(item["index"]): item for item in placements}
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"InNOut storefront plan has an invalid placement index: {plan_path}") from exc

        self._innout_storefront_plan_scene_id = scene_id
        self._innout_storefront_plan_value = {
            "path": str(plan_path),
            "placements_by_index": placements_by_index,
        }
        return self._innout_storefront_plan_value

    def _innout_storefront_goal_positions(self, start_pos, goal_prim):
        prim_path = str(self._goal_prim_path(goal_prim))
        match = re.search(r"_(\d+)$", prim_path)
        if not match:
            raise RuntimeError(f"Cannot map InNOut storefront prim to its placement: {prim_path}")
        storefront_plan = self._innout_storefront_plan()
        placement_index = int(match.group(1))
        placement = storefront_plan["placements_by_index"].get(placement_index)
        if placement is None:
            raise RuntimeError(
                f"InNOut storefront placement {placement_index} is missing for {prim_path}: "
                f"{storefront_plan['path']}"
            )

        outward = np.asarray(placement.get("outward_normal_xy"), dtype=float)
        outward_norm = float(np.linalg.norm(outward))
        if outward.shape != (2,) or outward_norm < 1e-6:
            raise RuntimeError(f"InNOut storefront has an invalid outward normal: {prim_path}")
        outward /= outward_norm
        tangent = np.asarray([-outward[1], outward[0]], dtype=float)
        facade_width = float(placement.get("storefront_facade_width", 0.0))
        lateral = min(
            float(self._sample("innout_storefront_max_lateral_m", 2.0)),
            float(self._sample("innout_storefront_lateral_fraction", 0.3)) * facade_width,
        )
        standoffs = self._sample("innout_storefront_standoff_distances_m", [0.5, 1.0])
        anchor = self._goal_prim_anchor(goal_prim)
        positions = []
        for standoff in standoffs:
            for lateral_offset in (0.0, -lateral, lateral):
                goal_pos = anchor.copy()
                goal_pos[:2] += outward * float(standoff) + tangent * lateral_offset
                goal_pos[2] = float(np.asarray(start_pos, dtype=float)[2])
                positions.append(goal_pos)
        return positions

    def _goal_prim_anchor(self, goal_prim):
        scene_id = self.cfg.scene["scene_id"]
        if getattr(self, "_goal_prim_anchor_scene_id", None) != scene_id:
            self._goal_prim_anchor_scene_id = scene_id
            self._goal_prim_anchor_cache = {}
            self._goal_prim_anchor_xform_cache = UsdGeom.XformCache()
        prim_path = str(self._goal_prim_path(goal_prim))
        if prim_path not in self._goal_prim_anchor_cache:
            prim = self.manager_env.scene.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Goal prim is missing: {prim_path}")
            transform = self._goal_prim_anchor_xform_cache.GetLocalToWorldTransform(prim)
            self._goal_prim_anchor_cache[prim_path] = np.asarray(transform.ExtractTranslation(), dtype=float)
        return self._goal_prim_anchor_cache[prim_path]

    def _filter_innout_goal_dict(self, goal_dict, plan):
        door_xy = np.asarray(plan["door_center_xy"], dtype=float)
        radius = float(self._sample("innout_goal_filter_radius_m", 50.0))
        outside_margin = float(self._sample("goal_outside_indoor_margin_m", 0.0))
        filtered = {}
        for goal_name, goal_item in goal_dict.items():
            exterior_prims = []
            for goal_prim in goal_item.get("prim", []):
                anchor = self._goal_prim_anchor(goal_prim)
                if np.linalg.norm(anchor[:2] - door_xy) > radius:
                    continue
                if not self._point_outside_innout_room_xy(anchor, plan, margin_m=outside_margin):
                    continue
                exterior_prims.append(goal_prim)
            if exterior_prims:
                filtered[goal_name] = {**goal_item, "prim": exterior_prims}
        return filtered

    def _validate_innout_target_capacity(self):
        plan = self._innout_plan()
        episode_mix = self._flow("episode_mix") or {"objnav": self.num_episodes}
        current_mix = Counter(episode.get("task_type") for episode in self.generated_episodes)
        limit = self._target_prim_episode_limit()
        for task_type, requested in episode_mix.items():
            remaining = max(0, int(requested) - current_mix.get(task_type, 0))
            if remaining == 0:
                continue
            filtered = self._filter_innout_goal_dict(
                self.goal_dict_by_task.get(task_type, {}),
                plan,
            )
            prim_paths = {
                str(self._goal_prim_path(prim))
                for item in filtered.values()
                for prim in item.get("prim", [])
            }
            capacity = sum(max(0, limit - self.target_prim_counts.get(path, 0)) for path in prim_paths)
            print(
                f"[TG] InNOut {task_type} capacity: targets={len(prim_paths)} "
                f"remaining_capacity={capacity} required={remaining} limit={limit}"
            )
            if capacity < remaining:
                raise RuntimeError(
                    f"InNOut {task_type} target capacity {capacity} is below the required {remaining}; "
                    f"found {len(prim_paths)} eligible targets within the doorway-centered radius"
                )

    @staticmethod
    def _path_crosses_innout_door(
        path,
        plan,
        lateral_margin_m=0.1,
        side_margin_m=0.1,
        min_forward_dot=0.0,
        max_lateral_offset_m=None,
    ):
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return False
        door_xy = np.asarray(plan["door_center_xy"], dtype=float)
        outward = plan["_outward_normal_xy"]
        tangent = np.asarray([-outward[1], outward[0]], dtype=float)
        signed = (path[:, :2] - door_xy) @ outward
        if signed[0] > -float(side_margin_m) or np.max(signed) < float(side_margin_m):
            return False
        half_width = 0.5 * float(plan["door_width"]) - float(lateral_margin_m)
        if max_lateral_offset_m is not None:
            half_width = min(half_width, float(max_lateral_offset_m))
        if half_width <= 0.0:
            return False
        for start, end, start_signed, end_signed in zip(
            path[:-1],
            path[1:],
            signed[:-1],
            signed[1:],
            strict=True,
        ):
            if start_signed > 0.0 or end_signed < 0.0:
                continue
            denominator = end_signed - start_signed
            if denominator <= 1e-9:
                continue
            segment_xy = end[:2] - start[:2]
            segment_norm = float(np.linalg.norm(segment_xy))
            if segment_norm <= 1e-9:
                continue
            if float(segment_xy @ outward / segment_norm) < float(min_forward_dot):
                continue
            fraction = -start_signed / denominator
            crossing_xy = start[:2] + fraction * (end[:2] - start[:2])
            if abs(float((crossing_xy - door_xy) @ tangent)) <= half_width:
                return True
        return False

    def start_clearance_default(self, scene_id, robot_radius=None, mode="task_generation"):
        if mode == "ui":
            return 0.60 if self._flow("rule_pattern") == "vc_store" else 0.0
        return float(self._sample("start_min_wall_distance", 0.0))

    def navmesh_clearance(self, point, navmesh_interface=None):
        navmesh_interface = navmesh_interface or self.navmesh_interface
        try:
            return float(navmesh_interface.distance_to_wall(np.asarray(point, dtype=float).tolist()))
        except Exception:
            return float("-inf")

    @staticmethod
    def _point_in_bbox(point, bbox):
        if bbox is None:
            return True
        point = np.asarray(point, dtype=float)
        bbox = np.asarray(bbox, dtype=float)
        return bool(
            bbox[0] <= point[0] <= bbox[3]
            and bbox[1] <= point[1] <= bbox[4]
            and bbox[2] <= point[2] <= bbox[5]
        )

    def is_valid_start_point(self, point, navmesh_interface, constraints=None, start_min_wall_distance=None):
        constraints = constraints or {}
        if start_min_wall_distance is None:
            start_min_wall_distance = self._sample("start_min_wall_distance", 0.0)
        min_clearance = max(
            float(start_min_wall_distance),
            float(constraints.get("start_clearance_m", 0.0) or 0.0),
        )
        if min_clearance > 0.0:
            clearance = self.navmesh_clearance(point, navmesh_interface)
            if not np.isfinite(clearance) or clearance < min_clearance:
                return False

        if not self._point_in_bbox(point, constraints.get("indoor_bbox")):
            return False

        innout_plan = constraints.get("innout_plan")
        if innout_plan is not None:
            room_margin = float(constraints.get("innout_room_margin_m", 0.0))
            if not self._point_in_innout_room_xy(point, innout_plan, margin_m=room_margin):
                return False
            door_margin = float(constraints.get("innout_door_interior_margin_m", 0.0))
            if self._innout_door_signed_distance(point, innout_plan) > -door_margin:
                return False

        z_range = constraints.get("z_range")
        if z_range is not None and not (float(z_range[0]) <= point[2] <= float(z_range[1])):
            return False

        bbox_margin = float(self._sample("start_bbox_margin", 0.0))
        if bbox_margin > 0.0:
            min_x, min_y, _, max_x, max_y, _ = self._start_sampling_bbox()
            if not (min_x + bbox_margin <= point[0] <= max_x - bbox_margin):
                return False
            if not (min_y + bbox_margin <= point[1] <= max_y - bbox_margin):
                return False
        return True

    def sample_valid_start(self, navmesh_interface, constraints=None, start_min_wall_distance=None):
        constraints = constraints or {}
        if start_min_wall_distance is None:
            start_min_wall_distance = self._sample("start_min_wall_distance", 0.0)
        batch_size = int(constraints.get("start_sample_batch_size", self._sample("start_sample_batch_size", 1)))
        attempts = int(constraints.get("start_sample_attempts", self._sample("start_sample_attempts", 1)))
        batch_size = max(1, batch_size)
        attempts = max(1, attempts)
        for _ in range(attempts):
            random_points = navmesh_interface.sample_random_points(batch_size)
            if random_points is None:
                return None
            indices = np.random.permutation(len(random_points))
            for idx in indices:
                point = np.asarray(random_points[idx], dtype=float)
                if self.is_valid_start_point(
                    point,
                    navmesh_interface,
                    constraints,
                    start_min_wall_distance=start_min_wall_distance,
                ):
                    return point
        return None

    def sample_valid_starts_near_point(
        self,
        navmesh_interface,
        center,
        radius_m=50.0,
        num_points=1,
        constraints=None,
        start_min_wall_distance=None,
    ):
        constraints = constraints or {}
        center = np.asarray(center, dtype=float)
        radius_m = float(radius_m)
        num_points = max(1, int(num_points))
        if start_min_wall_distance is None:
            start_min_wall_distance = self._sample("start_min_wall_distance", 0.0)
        if (
            not getattr(navmesh_interface, "built", False)
            or getattr(navmesh_interface, "navmesh_v", np.empty((0, 3))).shape[0] == 0
        ):
            return []

        vertices = np.asarray(navmesh_interface.navmesh_v, dtype=float)
        triangles = np.asarray(navmesh_interface.navmesh_t, dtype=int)
        tri_vertices = vertices[triangles]
        tri_min = tri_vertices[:, :, :2].min(axis=1)
        tri_max = tri_vertices[:, :, :2].max(axis=1)
        center_xy = center[:2]
        dx = np.maximum(np.maximum(tri_min[:, 0] - center_xy[0], 0.0), center_xy[0] - tri_max[:, 0])
        dy = np.maximum(np.maximum(tri_min[:, 1] - center_xy[1], 0.0), center_xy[1] - tri_max[:, 1])
        eligible = np.flatnonzero((dx * dx + dy * dy) <= radius_m * radius_m)
        if len(eligible) == 0:
            return []

        eligible_vertices = tri_vertices[eligible]
        areas = np.linalg.norm(
            np.cross(
                eligible_vertices[:, 1] - eligible_vertices[:, 0],
                eligible_vertices[:, 2] - eligible_vertices[:, 0],
            ),
            axis=1,
        )
        if not np.any(areas > 0.0):
            return []
        probabilities = areas / areas.sum()

        points = []
        max_batches = int(constraints.get("near_point_sample_attempts", 8))
        for _ in range(max_batches):
            remaining = num_points - len(points)
            if remaining <= 0:
                break
            batch_count = max(remaining * 6, 64)
            tri_indices = np.random.choice(len(eligible_vertices), size=batch_count, p=probabilities)
            selected = eligible_vertices[tri_indices]
            r1 = np.sqrt(np.random.random(batch_count))[:, None]
            r2 = np.random.random(batch_count)[:, None]
            samples = (
                (1.0 - r1) * selected[:, 0]
                + r1 * (1.0 - r2) * selected[:, 1]
                + r1 * r2 * selected[:, 2]
            )
            if (
                constraints.get("exclude_building_interiors", True)
                and len(getattr(navmesh_interface, "building_exclusion_bboxes", [])) > 0
            ):
                samples = samples[~navmesh_interface._point_in_excluded_building(samples)]
            for point in samples:
                if np.linalg.norm(point[:2] - center_xy) > radius_m:
                    continue
                if self.is_valid_start_point(
                    point,
                    navmesh_interface,
                    constraints,
                    start_min_wall_distance=start_min_wall_distance,
                ):
                    points.append(np.asarray(point, dtype=float))
                    if len(points) >= num_points:
                        break
        return points

    def _goal_endpoint_tolerance(self, obj_radius, task_type):
        if task_type == "placenav":
            return float(self._sample("placenav_goal_tolerance", 1.0))
        return float(obj_radius) + 1.0

    def _reference_paths_to_goal_observation(
        self,
        start_pos,
        goal_pos,
        obj_radius,
        task_type,
        navmesh_interface,
        reference_constraints=None,
    ):
        reference_paths = []
        path = self._find_reference_path(start_pos, goal_pos, navmesh_interface, goal_radius=obj_radius)
        constraints = {
            "navmesh_interface": navmesh_interface,
            "goal_radius": obj_radius,
            "task_type": task_type,
            "endpoint_tolerance_m": self._goal_endpoint_tolerance(obj_radius, task_type),
            "goal_clearance_m": float(self._sample("goal_clearance_m", 0.0) or 0.0),
        }
        if reference_constraints:
            constraints.update(reference_constraints)
        validation = self.validate_reference_path(
            start_pos,
            goal_pos,
            path,
            constraints,
        )
        if validation["accepted"]:
            reference_paths.append((
                validation["reference_path"],
                float(validation["reference_path_length_m"]),
                validation,
            ))
        else:
            reason = validation.get("reject_reason") or "unknown"
            plan_info = getattr(self, "_last_reference_path_plan_info", None) or {}
            plan_error = str(plan_info.get("error", ""))
            if reason == "no_reference_path" and plan_error:
                if "global_grid2d_collision" in plan_error:
                    reason = "no_reference_path:grid2d_collision"
                elif "global_grid2d_failed" in plan_error:
                    reason = "no_reference_path:grid2d_failed"
                elif "clearance check" in plan_error:
                    reason = "no_reference_path:path_clearance"
                elif "full footprint collision" in plan_error:
                    reason = "no_reference_path:footprint_collision"
                elif "final target connection is unsafe" in plan_error:
                    reason = "no_reference_path:unsafe_target_connection"
                elif "planning failed" in plan_error:
                    reason = "no_reference_path:planning_failed"
                elif "fallback rejected" in plan_error:
                    reason = "no_reference_path:fallback_rejected"
            self.reference_path_reject_counts[reason] += 1
            total_rejections = sum(self.reference_path_reject_counts.values())
            if total_rejections <= 10 or total_rejections % 100 == 0:
                print(
                    f"[TG] Reference path rejects: total={total_rejections} "
                    f"latest={reason} counts={dict(self.reference_path_reject_counts)}"
                )
        return reference_paths

    def _goal_prim_path(self, goal_prim):
        if hasattr(goal_prim, "GetPrimPath"):
            return goal_prim.GetPrimPath()
        if hasattr(goal_prim, "GetPath"):
            return goal_prim.GetPath()
        return goal_prim

    def _target_prim_episode_limit(self):
        return int(self._sample("target_prim_max_episodes", 3))

    def _target_prim_episode_quotas(self):
        quota_path = self._flow("target_prim_quota_path")
        if not quota_path:
            return None
        scene_id = self.cfg.scene.get("scene_id")
        cache_key = (scene_id, str(quota_path))
        if getattr(self, "_target_prim_quota_cache_key", None) == cache_key:
            return self._target_prim_quota_cache
        path = Path(str(quota_path))
        if not path.is_absolute():
            path = Path(self.args.tg_config_path).resolve().parent / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        quotas = payload.get(scene_id, {})
        if not isinstance(quotas, dict):
            raise RuntimeError(f"Invalid target prim quota entry for {scene_id}: {path}")
        self._target_prim_quota_cache_key = cache_key
        self._target_prim_quota_cache = {str(prim): int(limit) for prim, limit in quotas.items()}
        print(f"[TG] Loaded {len(quotas)} target prim episode quotas for {scene_id}: {path}")
        return self._target_prim_quota_cache

    @staticmethod
    def _episode_target_prim(episode):
        goals = episode.get("goals") or []
        if not goals:
            return None
        goal_index = episode.get("executed_goal_idx", episode.get("closest_goal_idx", 0))
        try:
            goal_index = int(goal_index)
        except (TypeError, ValueError):
            goal_index = 0
        if goal_index < 0 or goal_index >= len(goals):
            goal_index = 0
        return goals[goal_index].get("instance")

    def _count_episode_target_prims(self, episodes):
        counts = Counter()
        for episode in episodes:
            target_prim = self._episode_target_prim(episode)
            if target_prim:
                counts[target_prim] += 1
        return counts

    def _target_prim_available(self, target_prim, counts):
        quotas = self._target_prim_episode_quotas()
        if quotas is not None:
            limit = quotas.get(str(target_prim))
            return limit is not None and counts.get(str(target_prim), 0) < limit
        return counts.get(str(target_prim), 0) < self._target_prim_episode_limit()

    def _innout_failure_counts(self):
        scene_id = self.cfg.scene.get("scene_id")
        if getattr(self, "_innout_failure_scene_id", None) != scene_id:
            self._innout_failure_scene_id = scene_id
            self._innout_target_failures = Counter()
        return self._innout_target_failures

    def _record_innout_target_failure(self, target_prim, reason):
        if not target_prim:
            return
        prim_path = str(self._goal_prim_path(target_prim))
        failures = self._innout_failure_counts()
        failures[prim_path] += 1
        print(
            f"[TG] InNOut target failure: prim={prim_path} "
            f"score={failures[prim_path]} reason={reason}"
        )

    def _clear_innout_target_failure(self, target_prim):
        if self.rule_pattern != "innout" or not target_prim:
            return
        prim_path = str(self._goal_prim_path(target_prim))
        failures = self._innout_failure_counts()
        if failures.pop(prim_path, None) is not None:
            print(f"[TG] InNOut target recovered: prim={prim_path}")

    def _record_innout_check_failure(self, episode, reason, measurements):
        if self.rule_pattern != "innout":
            return
        target_prim = self._episode_target_prim(episode)
        self._record_innout_target_failure(target_prim, reason)
        self._innout_attempt_summary(reason, measurements)

    def _choose_innout_target(self, goal_dict, target_prim_counts):
        failures = self._innout_failure_counts()
        candidates = []
        for goal_name, goal_item in goal_dict.items():
            if not self._goal_name_allowed(goal_name):
                continue
            for goal_prim in goal_item.get("prim", []):
                prim_path = str(self._goal_prim_path(goal_prim))
                if self._target_prim_available(prim_path, target_prim_counts):
                    candidates.append((goal_name, goal_prim, prim_path))
        if not candidates:
            return None
        best_score = min(
            (failures[prim_path], target_prim_counts[prim_path])
            for _, _, prim_path in candidates
        )
        preferred = [
            candidate
            for candidate in candidates
            if (failures[candidate[2]], target_prim_counts[candidate[2]]) == best_score
        ]
        return preferred[int(np.random.randint(len(preferred)))]

    def _goal_item_has_available_prim(self, goal_item, counts):
        return any(
            self._target_prim_available(self._goal_prim_path(goal_prim), counts)
            for goal_prim in goal_item.get("prim", [])
        )

    def _goal_name_allowed(self, goal_name):
        exclude_pattern = self._sample("target_goal_name_exclude_regex")
        if exclude_pattern and re.search(str(exclude_pattern), str(goal_name)):
            return False
        return True

    def _filter_goals_by_target_prim_limit(self, goals, counts):
        return [
            goal for goal in goals
            if self._target_prim_available(goal.get("instance"), counts)
        ]

    def _goal_task_type(self, goal_name, goal_prim, default_task_type="objnav"):
        prim_path = str(self._goal_prim_path(goal_prim))
        for task_type in ("placenav", "objnav"):
            task_goal = self.goal_dict_by_task.get(task_type, {}).get(goal_name)
            if task_goal is None:
                continue
            for candidate_prim in task_goal.get("prim", []):
                if str(self._goal_prim_path(candidate_prim)) == prim_path:
                    return task_type
        return default_task_type

    def collect_goal_candidates(self, scene_id, task_type=None, filter_innout=True):
        total_goal_found = self.parse_scene()
        if total_goal_found == 0:
            raise RuntimeError("No goal object found in current scene")

        if task_type is None:
            goal_dict = self.goal_dict
        else:
            goal_dict = self.goal_dict_by_task.get(task_type, self.goal_dict)

        indoor_bbox = self.get_innout_indoor_bbox(scene_id) if filter_innout else None
        indoor_center_xy = None
        goal_filter_radius = None
        if indoor_bbox is not None:
            indoor_center_xy = (indoor_bbox[3:5] + indoor_bbox[:2]) / 2.0
            goal_filter_radius = 150.0 if self._flow("rule_pattern") == "vc_store" else 50.0

        candidates = []
        for goal_name, goal_item in goal_dict.items():
            for goal_prim in goal_item.get("prim", []):
                prim_path = str(self._goal_prim_path(goal_prim))
                prim = self.manager_env.scene.stage.GetPrimAtPath(prim_path)
                if not prim_path or not prim.IsValid():
                    continue
                try:
                    goal_pos = np.asarray(self.env.get_prim_position(prim_path), dtype=float)
                    obj_radius = float(self.env.get_prim_radius(prim_path))
                    bbox = [float(value) for value in self.env.get_prim_bounding_box(prim_path)]
                except Exception:
                    continue
                if indoor_center_xy is not None and np.linalg.norm(goal_pos[:2] - indoor_center_xy) >= goal_filter_radius:
                    continue
                candidate_task_type = task_type or self._goal_task_type(goal_name, goal_prim)
                candidates.append({
                    "goal_name": goal_name,
                    "goal_prim": goal_prim,
                    "goal_instance": prim_path,
                    "goal_position": goal_pos,
                    "goal_radius": obj_radius,
                    "goal_bbox": bbox,
                    "task_type": candidate_task_type,
                })
        if not candidates:
            raise RuntimeError("No valid goal object prims found in current scene")
        return candidates

    @staticmethod
    def _reference_path_length(reference_path):
        reference_path = np.asarray(reference_path, dtype=float)
        if len(reference_path) < 2:
            return 0.0
        return float(np.linalg.norm(reference_path[1:] - reference_path[:-1], axis=1).sum())

    @staticmethod
    def _xy_distance(a, b):
        return float(np.linalg.norm(np.asarray(a, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]))

    @staticmethod
    def _z_gap(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape[0] < 3 or b.shape[0] < 3:
            return 0.0
        return float(abs(a[2] - b[2]))

    def validate_reference_path(self, start_pos, goal_pos, reference_path, constraints=None):
        constraints = constraints or {}
        navmesh_interface = constraints.get("navmesh_interface", self.navmesh_interface)
        reference_path = np.asarray(reference_path, dtype=float)
        start_pos = np.asarray(start_pos, dtype=float)
        goal_pos = np.asarray(goal_pos, dtype=float)
        obj_radius = float(constraints.get("goal_radius", 0.0) or 0.0)
        task_type = constraints.get("task_type", "objnav")
        min_path_length = float(constraints.get("min_path_length_m", self.min_path_length))
        max_path_length = float(constraints.get("max_path_length_m", self.max_path_length))
        endpoint_tolerance = float(constraints.get("endpoint_tolerance_m", self._goal_endpoint_tolerance(obj_radius, task_type)))

        metadata = {
            "accepted": False,
            "reject_reason": None,
            "start_path_gap_m": float("inf"),
            "goal_path_gap_m": float("inf"),
            "goal_path_gap_3d_m": float("inf"),
            "goal_path_gap_z_m": float("inf"),
            "reference_path_length_m": 0.0,
        }
        if len(reference_path) < 2:
            metadata["reject_reason"] = "no_reference_path"
            return metadata

        dist_to_start = float(np.linalg.norm(start_pos - reference_path[0]))
        dist_to_end = self._xy_distance(goal_pos, reference_path[-1])
        dist_to_end_3d = float(np.linalg.norm(goal_pos - reference_path[-1]))
        metadata["start_path_gap_m"] = dist_to_start
        metadata["goal_path_gap_m"] = dist_to_end
        metadata["goal_path_gap_3d_m"] = dist_to_end_3d
        metadata["goal_path_gap_z_m"] = self._z_gap(goal_pos, reference_path[-1])
        if dist_to_start > float(constraints.get("start_path_tolerance_m", 1.0)):
            metadata["reject_reason"] = "start_path_gap"
            return metadata
        if dist_to_end > endpoint_tolerance:
            metadata["reject_reason"] = "goal_path_gap"
            return metadata

        path_length = self._reference_path_length(reference_path) + dist_to_start
        if path_length < min_path_length:
            metadata["reject_reason"] = "too_short"
            metadata["reference_path_length_m"] = path_length
            return metadata
        if path_length > max_path_length:
            metadata["reject_reason"] = "too_long"
            metadata["reference_path_length_m"] = path_length
            return metadata

        relax_endpoint_validity = bool(constraints.get(
            "relax_endpoint_validity",
            self._sample("relax_endpoint_validity", True),
        ))
        trimmed_path = self._trim_reference_path_to_valid_endpoint(reference_path, navmesh_interface, dist_to_start)
        if len(trimmed_path) < 2:
            if not relax_endpoint_validity:
                metadata["reject_reason"] = "invalid_trimmed_endpoint"
                return metadata
            trimmed_path = reference_path
            metadata["endpoint_validity_relaxed"] = True
        trimmed_goal_gap = self._xy_distance(goal_pos, trimmed_path[-1])
        metadata["goal_path_gap_m"] = trimmed_goal_gap
        metadata["goal_path_gap_3d_m"] = float(np.linalg.norm(goal_pos - trimmed_path[-1]))
        metadata["goal_path_gap_z_m"] = self._z_gap(goal_pos, trimmed_path[-1])
        if trimmed_goal_gap > endpoint_tolerance:
            metadata["reject_reason"] = "trimmed_goal_path_gap"
            return metadata
        trimmed_length = self._reference_path_length(trimmed_path) + dist_to_start
        if trimmed_length < min_path_length:
            metadata["reject_reason"] = "trimmed_too_short"
            metadata["reference_path_length_m"] = trimmed_length
            return metadata

        goal_clearance = float("nan")
        goal_clearance_min = float(constraints.get("goal_clearance_m", 0.0) or 0.0)
        if goal_clearance_min > 0.0:
            goal_clearance = self.navmesh_clearance(trimmed_path[-1], navmesh_interface)
            if not np.isfinite(goal_clearance) or goal_clearance < goal_clearance_min:
                metadata["reject_reason"] = "goal_clearance"
                metadata["goal_clearance_m"] = goal_clearance
                return metadata

        metadata.update({
            "accepted": True,
            "reference_path": trimmed_path,
            "reference_path_length_m": trimmed_length,
            "goal_clearance_m": goal_clearance,
        })
        return metadata

    @staticmethod
    def _nearest_goal_point_to_start(start_pos, goal_pos, bbox=None, standoff_m=0.0):
        start_pos = np.asarray(start_pos, dtype=float)
        goal_pos = np.asarray(goal_pos, dtype=float)
        goal_point = goal_pos.copy()
        if bbox is None:
            if start_pos.shape[0] >= 3 and goal_point.shape[0] >= 3:
                goal_point[2] = start_pos[2]
            return goal_point
        bbox = np.asarray(bbox, dtype=float)
        if bbox.shape[0] < 6 or not np.all(np.isfinite(bbox)):
            if start_pos.shape[0] >= 3 and goal_point.shape[0] >= 3:
                goal_point[2] = start_pos[2]
            return goal_point
        goal_point[0] = float(np.clip(start_pos[0], bbox[0], bbox[3]))
        goal_point[1] = float(np.clip(start_pos[1], bbox[1], bbox[4]))
        standoff_m = float(standoff_m or 0.0)
        if standoff_m > 0.0:
            center_xy = np.array([
                0.5 * (bbox[0] + bbox[3]),
                0.5 * (bbox[1] + bbox[4]),
            ], dtype=float)
            direction = start_pos[:2] - goal_point[:2]
            if np.linalg.norm(direction) < 1e-6:
                direction = start_pos[:2] - center_xy
            norm = float(np.linalg.norm(direction))
            if norm >= 1e-6:
                goal_point[:2] = goal_point[:2] + direction / norm * standoff_m
        if start_pos.shape[0] >= 3 and goal_point.shape[0] >= 3:
            goal_point[2] = float(start_pos[2])
        return goal_point

    def _build_goals_for_prim(
        self,
        start_pos,
        goal_prim,
        task_type,
        navmesh_interface,
        reference_constraints=None,
        goal_position=None,
    ):
        prim_path = self._goal_prim_path(goal_prim)
        target_center = np.asarray(self.env.get_prim_position(prim_path), dtype=float)
        obj_radius = self.env.get_prim_radius(prim_path)
        bbox = [float(value) for value in self.env.get_prim_bounding_box(prim_path)]
        if goal_position is None:
            goal_pos = self._nearest_goal_point_to_start(
                start_pos,
                target_center,
                bbox,
                standoff_m=float(self._sample("objnav_goal_standoff_m", 0.0) or 0.0),
            )
        else:
            goal_pos = np.asarray(goal_position, dtype=float).copy()
        goals = []
        for reference_path, reference_path_length, validation in self._reference_paths_to_goal_observation(
            start_pos,
            goal_pos,
                obj_radius,
                task_type,
                navmesh_interface,
                reference_constraints=reference_constraints,
        ):
            goal = {
                "instance": str(prim_path),
                "type": "place" if task_type == "placenav" else "object",
                "location": goal_pos.tolist(),
                "target_center": target_center.tolist(),
                "bbox": bbox,
                "radius": obj_radius,
                "path_length": reference_path_length,
                "reference_path": reference_path.tolist(),
                "goal_path_gap_m": float(validation.get("goal_path_gap_m", 0.0)),
                "goal_path_gap_3d_m": float(validation.get("goal_path_gap_3d_m", 0.0)),
                "goal_path_gap_z_m": float(validation.get("goal_path_gap_z_m", 0.0)),
            }
            if getattr(self, "_last_reference_path_planner", None) == "astar":
                goal["reference_path_planner"] = "astar"
                plan_info = getattr(self, "_last_reference_path_plan_info", None)
                if plan_info:
                    goal["astar_plan_info"] = deepcopy(plan_info)
            goals.append(goal)
        return goals

    def _cached_goal_prim_position(self, goal_prim):
        scene_id = self.cfg.scene["scene_id"]
        if getattr(self, "_goal_prim_position_cache_scene_id", None) != scene_id:
            self._goal_prim_position_cache_scene_id = scene_id
            self._goal_prim_position_cache = {}
        prim_path = str(self._goal_prim_path(goal_prim))
        cache = self._goal_prim_position_cache
        if prim_path not in cache:
            cache[prim_path] = np.asarray(self.env.get_prim_position(prim_path), dtype=float)
        return cache[prim_path]

    def _cached_goal_prim_bbox_center(self, goal_prim):
        scene_id = self.cfg.scene["scene_id"]
        if getattr(self, "_goal_prim_bbox_center_cache_scene_id", None) != scene_id:
            self._goal_prim_bbox_center_cache_scene_id = scene_id
            self._goal_prim_bbox_center_cache = {}
        prim_path = str(self._goal_prim_path(goal_prim))
        cache = self._goal_prim_bbox_center_cache
        if prim_path not in cache:
            bbox = np.asarray(self.env.get_prim_bounding_box(prim_path), dtype=float)
            if bbox.shape[0] >= 6 and np.isfinite(bbox).all():
                cache[prim_path] = 0.5 * (bbox[:3] + bbox[3:6])
            else:
                cache[prim_path] = self._cached_goal_prim_position(goal_prim)
        return cache[prim_path]

    def _sample_goal_object_candidate_plan(
        self,
        scene_id,
        start_pos,
        goal_candidates,
        goal_checks_per_start,
        constraints,
        reject_counts,
    ):
        navmesh_interface = constraints.get("navmesh_interface", self.navmesh_interface)
        order = np.random.permutation(len(goal_candidates))
        for idx in order[:goal_checks_per_start]:
            candidate = goal_candidates[int(idx)]
            candidate_task_type = candidate.get("task_type", constraints.get("task_type") or "objnav")
            candidate_goals = self._build_goals_for_prim(
                start_pos,
                candidate["goal_prim"],
                candidate_task_type,
                navmesh_interface,
            )
            goals = self._filter_visible_goals(candidate_goals, navmesh_interface)
            if not goals:
                reject_counts["no_valid_goal_path"] = reject_counts.get("no_valid_goal_path", 0) + 1
                for goal in candidate_goals:
                    reason = goal.get("visibility_reject_reason")
                    if reason:
                        key = f"goal_visibility:{reason}"
                        reject_counts[key] = reject_counts.get(key, 0) + 1
                continue
            plan = self._goal_object_plan_from_goal(
                scene_id,
                start_pos,
                candidate,
                candidate_task_type,
                min(goals, key=lambda item: item["path_length"]),
                constraints,
                reject_counts,
            )
            if plan is not None:
                return plan
        return None

    def _goal_object_plan_from_goal(
        self,
        scene_id,
        start_pos,
        candidate,
        candidate_task_type,
        goal,
        constraints,
        reject_counts,
    ):
        navmesh_interface = constraints.get("navmesh_interface", self.navmesh_interface)
        reference_path = np.asarray(goal["reference_path"], dtype=float)
        validation = self.validate_reference_path(
            start_pos,
            np.asarray(goal["location"], dtype=float),
            reference_path,
            {
                **constraints,
                "navmesh_interface": navmesh_interface,
                "goal_radius": float(goal["radius"]),
                "task_type": candidate_task_type,
            },
        )
        if not validation["accepted"]:
            reason = validation.get("reject_reason", "invalid_reference_path")
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            return None

        reference_path = np.asarray(validation["reference_path"], dtype=float)
        settle_check = constraints.get("settle_check")
        settle_info = {"checked": False}
        if settle_check is not None:
            settles, settle_info = settle_check(start_pos, reference_path)
            if not settles:
                reject_counts["unstable_start"] = reject_counts.get("unstable_start", 0) + 1
                return None

        start_clearance = self.navmesh_clearance(start_pos, navmesh_interface)
        goal_clearance = validation.get("goal_clearance_m")
        if goal_clearance is None or not np.isfinite(goal_clearance):
            goal_clearance = self.navmesh_clearance(reference_path[-1], navmesh_interface)
        plan = {
            "sampled": True,
            "scene_id": scene_id,
            "sample_mode": constraints.get("sample_mode", "goal_object"),
            "start_position": np.asarray(start_pos, dtype=float).tolist(),
            "goal_name": candidate["goal_name"],
            "goal_instance": goal["instance"],
            "goal_position": np.asarray(goal["location"], dtype=float).tolist(),
            "goal_radius": float(goal["radius"]),
            "task_type": candidate_task_type,
            "reference_path": reference_path.tolist(),
            "reference_path_length_m": float(validation["reference_path_length_m"]),
            "start_clearance_m": start_clearance,
            "goal_clearance_m": float(goal_clearance),
            "start_path_gap_m": float(validation["start_path_gap_m"]),
            "goal_path_gap_m": float(validation["goal_path_gap_m"]),
            "reject_counts": reject_counts,
            "settle": settle_info,
        }
        if "reference_path_planner" in goal:
            plan["reference_path_planner"] = goal["reference_path_planner"]
        if "astar_plan_info" in goal:
            plan["astar_plan_info"] = deepcopy(goal["astar_plan_info"])
        return plan

    @contextlib.contextmanager
    def _goal_object_sampling_overrides(self, constraints):
        missing = object()
        if "min_path_length_m" in constraints:
            previous_min_path_length = getattr(self, "min_path_length", missing)
            self.min_path_length = float(constraints["min_path_length_m"])
        else:
            previous_min_path_length = None
        if "max_path_length_m" in constraints:
            previous_max_path_length = getattr(self, "max_path_length", missing)
            self.max_path_length = float(constraints["max_path_length_m"])
        else:
            previous_max_path_length = None

        previous_trajectory_planner = getattr(self.args, "trajectory_planner", None)
        if "trajectory_planner" in constraints:
            self.args.trajectory_planner = constraints["trajectory_planner"]

        try:
            yield
        finally:
            if previous_min_path_length is not None:
                if previous_min_path_length is missing:
                    delattr(self, "min_path_length")
                else:
                    self.min_path_length = previous_min_path_length
            if previous_max_path_length is not None:
                if previous_max_path_length is missing:
                    delattr(self, "max_path_length")
                else:
                    self.max_path_length = previous_max_path_length
            if "trajectory_planner" in constraints:
                self.args.trajectory_planner = previous_trajectory_planner

    def _goal_object_sampling_candidates(self, scene_id, constraints):
        goal_candidates = constraints.get("goal_candidates")
        if goal_candidates is not None:
            return list(goal_candidates)
        return self.collect_goal_candidates(
            scene_id,
            task_type=constraints.get("task_type"),
            filter_innout=bool(constraints.get("filter_innout", True)),
        )

    def _goal_object_start_constraints(self, scene_id, constraints):
        start_constraints = {
            "start_clearance_m": constraints.get("start_clearance_m", 0.0),
            "start_min_wall_distance": constraints.get(
                "start_min_wall_distance",
                self._sample("start_min_wall_distance", 0.0),
            ),
            "indoor_bbox": constraints.get("indoor_bbox"),
            "start_sample_attempts": constraints.get("start_sample_attempts", self._sample("start_sample_attempts", 1)),
            "start_sample_batch_size": constraints.get("start_sample_batch_size", self._sample("start_sample_batch_size", 1)),
        }
        if start_constraints["indoor_bbox"] is None and bool(constraints.get("filter_innout", True)):
            start_constraints["indoor_bbox"] = self.get_innout_indoor_bbox(scene_id)
        return start_constraints

    def sample_goal_object_plan(self, scene_id, constraints=None):
        constraints = dict(constraints or {})
        if self._start_settle_enabled():
            constraints.setdefault("settle_check", self._task_generation_start_settles)
        navmesh_interface = constraints.get("navmesh_interface", self.navmesh_interface)
        max_attempts = int(constraints.get("max_attempts", 100))
        goal_checks_per_start = max(1, int(constraints.get("goal_checks_per_start", 64)))
        sample_mode = constraints.get("sample_mode", "goal_object")
        reject_counts = constraints.setdefault("reject_counts", {})

        with self._goal_object_sampling_overrides(constraints):
            goal_candidates = self._goal_object_sampling_candidates(scene_id, constraints)
            start_constraints = self._goal_object_start_constraints(scene_id, constraints)
            start_min_wall_distance = float(start_constraints.get("start_min_wall_distance", 0.0))
            for _ in range(max_attempts):
                start_pos = self.sample_valid_start(
                    navmesh_interface,
                    start_constraints,
                    start_min_wall_distance=start_min_wall_distance,
                )
                if start_pos is None:
                    reject_counts["no_valid_start"] = reject_counts.get("no_valid_start", 0) + 1
                    continue
                plan = self._sample_goal_object_candidate_plan(
                    scene_id,
                    start_pos,
                    goal_candidates,
                    goal_checks_per_start,
                    {**constraints, "navmesh_interface": navmesh_interface, "sample_mode": sample_mode},
                    reject_counts,
                )
                if plan is not None:
                    return plan
        return {
            "sampled": False,
            "scene_id": scene_id,
            "sample_mode": sample_mode,
            "reject_counts": reject_counts,
            "error": f"Failed after {max_attempts} attempts",
        }

    def _goal_visibility_aim_position(self, goal, endpoint=None):
        if goal.get("type") != "place" or "bbox" not in goal:
            return np.asarray(goal["location"], dtype=np.float32)
        bbox = np.asarray(goal["bbox"], dtype=np.float32)
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        if endpoint is None:
            endpoint = goal.get("visibility_endpoint_position", goal["reference_path"][-1])
        endpoint = np.asarray(endpoint, dtype=np.float32)
        aim_x = float(np.clip(endpoint[0], min_x, max_x))
        aim_y = float(np.clip(endpoint[1], min_y, max_y))
        inside_x = min_x < endpoint[0] < max_x
        inside_y = min_y < endpoint[1] < max_y
        if inside_x and inside_y:
            side_distances = [
                (abs(endpoint[0] - min_x), "min_x"),
                (abs(endpoint[0] - max_x), "max_x"),
                (abs(endpoint[1] - min_y), "min_y"),
                (abs(endpoint[1] - max_y), "max_y"),
            ]
            _, side = min(side_distances, key=lambda item: item[0])
            if side == "min_x":
                aim_x = float(min_x)
            elif side == "max_x":
                aim_x = float(max_x)
            elif side == "min_y":
                aim_y = float(min_y)
            else:
                aim_y = float(max_y)
        elif inside_x:
            aim_y = float(min_y if endpoint[1] < min_y else max_y)
        elif inside_y:
            aim_x = float(min_x if endpoint[0] < min_x else max_x)
        aim_z = float((min_z + max_z) * 0.5)
        return np.asarray([aim_x, aim_y, aim_z], dtype=np.float32)

    def _is_black_far_visibility_view(self, rendered):
        render_stats = (rendered or {}).get("render_stats", {})
        rgb_mean = render_stats.get("rgb_mean")
        depth_center = render_stats.get("depth_center_median")
        if rgb_mean is None or depth_center is None:
            return False
        return (
            rgb_mean < float(self._visibility_check("black_rgb_mean", 1.0))
            and depth_center > float(self._visibility_check("far_depth_center", 100.0))
        )

    def _goal_visibility_camera_path(self):
        pov_path = self.env.get_camera_prim_path("pov_camera")
        return str(Path(pov_path).with_name("GoalVisibilityCamera"))

    def _goal_visibility_camera(self, include_rgb=False, include_depth=False):
        pov_cfg = self.manager_env.scene["pov_camera"].cfg
        data_types = ["instance_segmentation_fast"]
        if include_rgb:
            data_types.insert(0, "rgb")
        if include_depth:
            data_types.insert(1 if include_rgb else 0, "distance_to_image_plane")
        cache_key = (self.cfg.scene["scene_id"], bool(include_rgb), bool(include_depth))
        camera_path = self._goal_visibility_camera_path()
        if getattr(self, "_goal_visibility_camera_key", None) == cache_key and hasattr(self, "goal_visibility_camera"):
            return self.goal_visibility_camera

        spawn_cfg = pov_cfg.spawn
        clipping_min, clipping_max = spawn_cfg.clipping_range
        self.goal_visibility_camera = self.env.create_camera(
            prim_path=camera_path,
            perspective=True,
            pos=pov_cfg.offset.pos,
            quat_opengl=pov_cfg.offset.rot,
            offset_convention=pov_cfg.offset.convention,
            focal_length=spawn_cfg.focal_length,
            horizontal_aperture=spawn_cfg.horizontal_aperture,
            clipping_range_min=clipping_min,
            clipping_range_max=clipping_max,
            width=pov_cfg.width,
            height=pov_cfg.height,
            data_types=data_types,
            colorize_instance_segmentation=False,
        )
        self._goal_visibility_camera_key = cache_key
        return self.goal_visibility_camera

    def _render_goal_instance_segmentation(self, include_rgb=False, include_depth=False):
        sensor = self._goal_visibility_camera(include_rgb=include_rgb, include_depth=include_depth)
        sim = self.manager_env.sim
        sim.forward()
        sim.render()
        with torch.inference_mode():
            sensor._is_outdated[:] = True
        sensor.update(0.0, force_recompute=True)
        segmentation = sensor.data.output.get("instance_segmentation_fast")
        if segmentation is None:
            return None
        info = sensor.data.info.get("instance_segmentation_fast", {})
        rendered = {
            "segmentation": segmentation[0, :, :, 0].detach().cpu().numpy(),
            "info": info,
        }
        if include_rgb and sensor.data.output.get("rgb") is not None:
            rendered["rgb"] = sensor.data.output["rgb"][0].detach().cpu().numpy()
        if include_depth and sensor.data.output.get("distance_to_image_plane") is not None:
            rendered["depth"] = sensor.data.output["distance_to_image_plane"][0].detach().cpu().numpy()
        return rendered

    def _metadata_matches_target(self, metadata, prim_path, semantic_label=None):
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if self._metadata_matches_target(key, prim_path, semantic_label):
                    return True
                if self._metadata_matches_target(value, prim_path, semantic_label):
                    return True
            return False
        if isinstance(metadata, (list, tuple)):
            return any(self._metadata_matches_target(value, prim_path, semantic_label) for value in metadata)
        text = str(metadata)
        if semantic_label and semantic_label in text:
            return True
        if not prim_path:
            return False
        prim_path = str(prim_path).rstrip("/")
        if text == prim_path or text.startswith(f"{prim_path}/"):
            return True
        if text.startswith("/") and prim_path.startswith(f"{text.rstrip('/')}/"):
            return True
        metadata_path_suffix = r"(?=$|[/\s'\",}\]])"
        path_pattern = rf"{re.escape(prim_path)}{metadata_path_suffix}"
        if re.search(path_pattern, text):
            return True
        terrain_prefix = "/World/ground/terrain"
        if prim_path.startswith(terrain_prefix):
            suffix = prim_path[len(terrain_prefix):]
            if suffix:
                suffix_pattern = rf"{re.escape(suffix)}{metadata_path_suffix}"
                return re.search(suffix_pattern, text) is not None
        return False

    def _goal_visibility_label(self, prim_path):
        digest = hashlib.md5(str(prim_path).encode("utf-8")).hexdigest()[:16]
        return f"navverse_goal_{digest}"

    def _apply_goal_visibility_label(self, prim_path, semantic_label):
        prim = self.manager_env.scene.stage.GetPrimAtPath(prim_path)
        if not prim:
            return
        for child in Usd.PrimRange(prim):
            if not child.IsA(UsdGeom.Imageable):
                continue
            sem = Semantics.SemanticsAPI.Apply(child, "NavVerseGoal")
            sem.CreateSemanticTypeAttr().Set("class")
            sem.CreateSemanticDataAttr().Set(semantic_label)

    def _goal_target_instance_ids(self, info, prim_path, semantic_label=None):
        target_ids = []
        id_metadata = {}
        for key in ("idToSemantics", "idToLabels"):
            if isinstance(info.get(key), dict):
                id_metadata.update(info[key])
        if not id_metadata:
            id_metadata = info
        for raw_id, metadata in id_metadata.items():
            try:
                instance_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if self._metadata_matches_target(metadata, prim_path, semantic_label):
                target_ids.append(instance_id)
        return target_ids

    def _reset_robot_joints_for_visibility(self):
        robot = self.manager_env.scene["robot"]
        with torch.inference_mode():
            if hasattr(robot.data, "default_joint_pos"):
                joint_pos = robot.data.default_joint_pos.to(device=robot.device, dtype=robot.data.joint_pos.dtype).clone()
                joint_vel = torch.zeros_like(robot.data.joint_vel)
                robot.write_joint_position_to_sim(joint_pos)
                robot.write_joint_velocity_to_sim(joint_vel)
                if hasattr(robot, "set_joint_position_target"):
                    robot.set_joint_position_target(joint_pos)
                if hasattr(robot, "set_joint_velocity_target"):
                    robot.set_joint_velocity_target(joint_vel)
            robot.write_data_to_sim()

    def _robot_pose_is_stable_for_visibility(self, endpoint):
        robot = self.manager_env.scene["robot"]
        root_state = robot.data.root_state_w[0].detach().cpu().numpy()
        quat_wxyz = root_state[3:7]
        if not np.isfinite(root_state[:7]).all():
            return False, "non_finite_robot_pose"
        up = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).apply([0.0, 0.0, 1.0])
        max_tilt = float(self._visibility_check("max_tilt_deg", 45.0))
        tilt = np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))
        if tilt > max_tilt:
            return False, f"bad_orientation:{tilt:.1f}deg"
        expected_z = float(endpoint[2]) + float(
            self._check("robot_spawn_height", getattr(self.manager_env.cfg, "robot_spawn_height", 0.6))
        )
        max_z_drop = float(self._visibility_check("max_z_error", 0.75))
        if root_state[2] < expected_z - max_z_drop:
            return False, f"z_drop:{root_state[2]:.2f}<{expected_z:.2f}"
        return True, None

    def _settle_robot_for_visibility(self, endpoint):
        self._reset_settle_state()
        zero_cmd = torch.zeros(self.vln_sim.command_dim, device=self.manager_env.device)
        settle_steps = int(self._visibility_check("settle_steps", 0))
        settle_steps = max(0, settle_steps)
        if settle_steps > 0:
            with torch.inference_mode():
                self.env._reset_low_level_policy_state()
        for _ in range(settle_steps):
            env_action = self.env._command_to_env_action(zero_cmd)
            obs, _, done, _ = self.env._step_low_level_env(env_action)
            self.env.obs = obs
            if torch.as_tensor(done).any().item():
                return False, "low_level_done"
        reset_buf = self.env.termination_manager.compute()
        if torch.as_tensor(reset_buf).any().item():
            terms = self.env.termination_manager.get_active_iterable_terms(self.vln_sim.robot_index)
            reason = max(terms, key=lambda item: item[1])[0] if terms else "termination"
            return False, reason
        return self._robot_pose_is_stable_for_visibility(endpoint)

    def _reset_settle_state(self):
        if hasattr(self.env, "set_stop_called"):
            self.env.set_stop_called(self.vln_sim.robot_index, False)
        self.env.reset_termination_states([self.vln_sim.robot_index])
        self.env.termination_manager.reset()
        try:
            self.env.measure_manager.reset_measures()
        except Exception:
            pass
        self.vln_sim.done = None
        if isinstance(getattr(self.vln_sim, "info", None), dict):
            self.vln_sim.info.pop("terminations", None)

    def _robot_tilt_deg(self):
        robot = self.manager_env.scene["robot"]
        quat_wxyz = robot.data.root_quat_w[0].detach().cpu().numpy().astype(float)
        up = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).apply([0.0, 0.0, 1.0])
        return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def _robot_yaw_rad(self):
        robot = self.manager_env.scene["robot"]
        quat_wxyz = robot.data.root_quat_w[0].detach().cpu().numpy().astype(float)
        return float(R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_euler("ZYX")[0])

    def _zero_action_settle_step(self, zero_cmd):
        env_action = self.env._command_to_env_action(zero_cmd)
        obs, _, done, info = self.env._step_low_level_env(env_action)
        self.env.obs = obs
        self.vln_sim.obs = obs
        if isinstance(info, dict):
            self.vln_sim.info = info
        if torch.as_tensor(done).any().item():
            reason = None
            if isinstance(info, dict) and "terminations" in info:
                reason = info["terminations"].get("termination_reason")
            return reason or "low_level_done"
        reset_buf = self.env.termination_manager.compute()
        if torch.as_tensor(reset_buf).any().item():
            terms = self.env.termination_manager.get_active_iterable_terms(self.vln_sim.robot_index)
            return max(terms, key=lambda item: item[1])[0] if terms else "termination"
        return None

    def _start_settle_motion_probe(self, start_pos, yaw_angle, command, step_count, label):
        robot = self.manager_env.scene["robot"]
        quat_xyzw = R.from_euler("z", float(yaw_angle)).as_quat().tolist()
        temp_episode = {
            "start_position": np.asarray(start_pos, dtype=float).tolist(),
            "start_rotation": [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        }
        set_robot_pose(self.manager_env.cfg, temp_episode, robot)
        self._reset_robot_joints_for_visibility()
        self._reset_settle_state()
        obs, infos = self.env.env.reset()
        self.env.obs = obs
        self.vln_sim.obs = obs
        if isinstance(infos, dict):
            self.vln_sim.info = infos

        before = robot.data.root_state_w[0, :3].detach().cpu().numpy().astype(float)
        before_yaw = self._robot_yaw_rad()
        max_body_force = 0.0
        contact_threshold = self._start_settle_check("contact_force_threshold_n")
        if contact_threshold is None:
            contact_threshold = self._contact_check("force_threshold_n", 5.0)
        contact_threshold = float(contact_threshold)
        max_tilt = float(self._start_settle_check("max_tilt_deg", 25.0))
        warmup_steps = self._start_settle_check("probe_warmup_steps")
        if warmup_steps is None:
            warmup_steps = getattr(self.manager_env.cfg, "robot_warmup_steps", 0)
        warmup_steps = int(warmup_steps)
        with torch.inference_mode():
            self.env._reset_low_level_policy_state()
            zero_cmd = torch.zeros(self.vln_sim.command_dim, device=self.manager_env.device)
            for _ in range(max(0, warmup_steps)):
                reason = self._zero_action_settle_step(zero_cmd)
                if reason in {"bad_orientation", "terrain_out_of_bounds", "illegal_contact", "low_level_done"}:
                    return False, {"probe": label, "termination_reason": reason, "phase": "warmup"}
            for _ in range(max(0, int(step_count))):
                reason = self._zero_action_settle_step(command)
                force_n = self._contact_body_force_norm()
                if np.isfinite(force_n):
                    max_body_force = max(max_body_force, force_n)
                if reason in {"bad_orientation", "terrain_out_of_bounds", "illegal_contact", "low_level_done"}:
                    return False, {
                        "probe": label,
                        "termination_reason": reason,
                        "max_body_force_n": float(max_body_force),
                    }
                if np.isfinite(force_n) and force_n >= contact_threshold:
                    return False, {
                        "probe": label,
                        "termination_reason": "body_contact",
                        "body_force_n": float(force_n),
                        "max_body_force_n": float(max_body_force),
                    }
                tilt_deg = self._robot_tilt_deg()
                if tilt_deg > max_tilt:
                    return False, {
                        "probe": label,
                        "termination_reason": "bad_orientation",
                        "tilt_deg": float(tilt_deg),
                        "max_body_force_n": float(max_body_force),
                    }

        after = robot.data.root_state_w[0, :3].detach().cpu().numpy().astype(float)
        displacement = float(np.linalg.norm(after[:2] - before[:2])) if np.isfinite(after).all() else float("inf")
        after_yaw = self._robot_yaw_rad()
        yaw_delta = float(np.degrees(np.arctan2(np.sin(after_yaw - before_yaw), np.cos(after_yaw - before_yaw))))
        probe_info = {
            "probe": label,
            "warmup_frames": int(max(0, warmup_steps)),
            "frames": int(step_count),
            "xy_displacement_m": displacement,
            "yaw_delta_deg": yaw_delta,
            "tilt_deg": self._robot_tilt_deg(),
            "max_body_force_n": float(max_body_force),
        }
        if label.startswith("turn_"):
            min_turn_delta = float(self._start_settle_check("turn_min_delta_deg", 60.0))
            if abs(yaw_delta) < min_turn_delta:
                probe_info["termination_reason"] = "turn_probe_short"
                return False, probe_info
        if label == "forward":
            min_forward_displacement = float(self._start_settle_check("forward_min_displacement_m", 0.5))
            if displacement < min_forward_displacement:
                probe_info["termination_reason"] = "forward_probe_short"
                return False, probe_info
        return True, probe_info

    def _task_generation_start_settles(self, start_pos, reference_path, yaw_angle=None):
        if not self._start_settle_enabled():
            return True, {"checked": False}
        settle_steps = int(self._start_settle_check("check_steps", 0))
        if settle_steps <= 0:
            return True, {"checked": False}

        reference_path = np.asarray(reference_path, dtype=float)
        if yaw_angle is None:
            yaw_angle = calc_yaw(start_pos[:2], reference_path[min(1, len(reference_path) - 1)][:2])

        robot_state = self.vln_sim._snapshot_robot_sim_state()
        robot = self.manager_env.scene["robot"]
        zero_cmd = torch.zeros(self.vln_sim.command_dim, device=self.manager_env.device)
        try:
            quat_xyzw = R.from_euler("z", float(yaw_angle)).as_quat().tolist()
            temp_episode = {
                "start_position": np.asarray(start_pos, dtype=float).tolist(),
                "start_rotation": [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                "robot_spawn_height": float(
                    self._check("robot_spawn_height", getattr(self.manager_env.cfg, "robot_spawn_height", 0.6))
                ),
            }
            set_robot_pose(self.manager_env.cfg, temp_episode, robot)
            self._reset_robot_joints_for_visibility()
            self._reset_settle_state()

            before = robot.data.root_state_w[0, :3].detach().cpu().numpy().astype(float)
            max_body_force = 0.0
            contact_threshold = self._start_settle_check("contact_force_threshold_n")
            if contact_threshold is None:
                contact_threshold = self._contact_check("force_threshold_n", 5.0)
            contact_threshold = float(contact_threshold)
            with torch.inference_mode():
                self.env._reset_low_level_policy_state()
                for _ in range(settle_steps):
                    reason = self._zero_action_settle_step(zero_cmd)
                    force_n = self._contact_body_force_norm()
                    if np.isfinite(force_n):
                        max_body_force = max(max_body_force, force_n)
                    if reason in {"bad_orientation", "terrain_out_of_bounds", "illegal_contact", "low_level_done"}:
                        return False, {"checked": True, "termination_reason": reason}
                    if np.isfinite(force_n) and force_n >= contact_threshold:
                        return False, {
                            "checked": True,
                            "termination_reason": "body_contact",
                            "body_force_n": float(force_n),
                        }

            after = robot.data.root_state_w[0, :3].detach().cpu().numpy().astype(float)
            xy_drift = float(np.linalg.norm(after[:2] - before[:2])) if np.isfinite(after).all() else float("inf")
            z_drop = float(before[2] - after[2]) if np.isfinite(after).all() else float("inf")
            tilt_deg = self._robot_tilt_deg()
            max_xy_drift = float(self._start_settle_check("max_xy_drift_m", 0.20))
            max_z_drop = float(self._start_settle_check("max_z_drop_m", 0.25))
            max_tilt = float(self._start_settle_check("max_tilt_deg", 25.0))
            ok = xy_drift <= max_xy_drift and z_drop <= max_z_drop and tilt_deg <= max_tilt
            settle_info = {
                "checked": True,
                "frames": settle_steps,
                "xy_drift_m": xy_drift,
                "z_drop_m": z_drop,
                "tilt_deg": tilt_deg,
                "max_body_force_n": float(max_body_force),
            }
            if not ok:
                failed_checks = []
                if xy_drift > max_xy_drift:
                    failed_checks.append("xy_drift")
                if z_drop > max_z_drop:
                    failed_checks.append("z_drop")
                if tilt_deg > max_tilt:
                    failed_checks.append("tilt")
                settle_info["termination_reason"] = "settle:" + ",".join(failed_checks or ["unknown"])
                return False, settle_info

            if bool(self._start_settle_check("motion_probe", True)):
                step_dt = float(getattr(self.env, "step_dt", 0.05) or 0.05)
                yaw_rate = float(self._start_settle_check("turn_rate_rad_s", 0.6))
                forward_speed = float(self._start_settle_check("forward_speed_m_s", 0.3))
                turn_angle = float(self._start_settle_check("turn_angle_deg", 90.0))
                forward_distance = float(self._start_settle_check("forward_distance_m", 1.0))
                turn_steps = int(np.ceil(np.deg2rad(turn_angle) / max(abs(yaw_rate) * step_dt, 1e-6)))
                forward_steps = int(np.ceil(forward_distance / max(abs(forward_speed) * step_dt, 1e-6)))
                probes = [
                    ("turn_left", torch.tensor([0.0, 0.0, yaw_rate], device=self.manager_env.device)),
                    ("turn_right", torch.tensor([0.0, 0.0, -yaw_rate], device=self.manager_env.device)),
                    ("forward", torch.tensor([forward_speed, 0.0, 0.0], device=self.manager_env.device)),
                ]
                probe_results = []
                for label, command in probes:
                    steps = turn_steps if label.startswith("turn_") else forward_steps
                    probe_ok, probe_info = self._start_settle_motion_probe(start_pos, yaw_angle, command, steps, label)
                    probe_results.append(probe_info)
                    if not probe_ok:
                        settle_info["motion_probes"] = probe_results
                        settle_info["termination_reason"] = f"motion_probe_{probe_info.get('termination_reason', 'failed')}"
                        settle_info["failed_probe"] = label
                        return False, settle_info
                settle_info["motion_probes"] = probe_results
            return True, settle_info
        finally:
            self.vln_sim._restore_robot_sim_state(robot_state)
            self._reset_settle_state()

    def _move_robot_to_goal_endpoint(self, goal, navmesh_interface):
        self._last_goal_visibility_reject_reason = None
        reference_path = np.asarray(goal["reference_path"], dtype=np.float32)
        if len(reference_path) == 0:
            self._last_goal_visibility_reject_reason = "empty_reference_path"
            return False
        endpoint = reference_path[-1]

        goal["visibility_endpoint_position"] = endpoint.tolist()
        aim_pos = self._goal_visibility_aim_position(goal, endpoint)
        goal["visibility_aim_position"] = aim_pos.tolist()
        yaw_angle = calc_yaw(endpoint[:2], aim_pos[:2])
        goal["visibility_yaw_angle"] = float(yaw_angle)
        quat_xyzw = R.from_euler("z", yaw_angle).as_quat().tolist()
        temp_episode = {
            "start_position": endpoint.tolist(),
            "start_rotation": [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        }
        set_robot_pose(self.manager_env.cfg, temp_episode, self.manager_env.scene["robot"])
        self._reset_robot_joints_for_visibility()
        ok, reason = self._settle_robot_for_visibility(endpoint)
        if not ok:
            self._last_goal_visibility_reject_reason = reason
        return ok

    def _goal_visibility_candidate_indices(self, reference_path):
        max_candidates = int(self._visibility_check("endpoint_candidates", 12))
        stride = int(self._visibility_check("endpoint_stride", 2))
        max_candidates = max(1, max_candidates)
        stride = max(1, stride)
        return list(range(len(reference_path) - 1, 0, -stride))[:max_candidates]

    def _goal_visibility_render_stats(self, goal, rendered):
        render_stats = {}
        if rendered is None:
            return render_stats
        if rendered.get("rgb") is not None:
            render_stats["rgb_mean"] = float(np.asarray(rendered["rgb"]).mean())
        if rendered.get("depth") is not None:
            depth = np.asarray(rendered["depth"]).squeeze()
            valid = np.isfinite(depth) & (depth > 0.0)
            render_stats["depth_valid_ratio"] = float(valid.mean())
            if valid.any():
                valid_depth = depth[valid]
                render_stats["depth_min"] = float(np.min(valid_depth))
                render_stats["depth_p01"] = float(np.percentile(valid_depth, 1))
                render_stats["depth_p05"] = float(np.percentile(valid_depth, 5))
                render_stats["depth_median"] = float(np.median(valid_depth))
                h, w = depth.shape[:2]
                y0, y1 = int(h * 0.4), int(h * 0.6)
                x0, x1 = int(w * 0.4), int(w * 0.6)
                center = depth[y0:y1, x0:x1]
                center_valid = center[np.isfinite(center) & (center > 0.0)]
                if center_valid.size:
                    render_stats["depth_center_median"] = float(np.median(center_valid))
        camera_pos, _ = self.env.get_cam_pose()
        target_pos = np.asarray(goal.get("visibility_aim_position", goal["location"]), dtype=np.float32)
        endpoint = np.asarray(goal.get("visibility_endpoint_position", goal["reference_path"][-1]), dtype=np.float32)
        to_target = target_pos - camera_pos
        horizontal = np.array([to_target[0], to_target[1], 0.0], dtype=np.float32)
        horizontal_norm = float(np.linalg.norm(horizontal))
        yaw_angle = float(goal.get("visibility_yaw_angle", calc_yaw(endpoint[:2], target_pos[:2])))
        forward_xy = np.array([np.cos(yaw_angle), np.sin(yaw_angle), 0.0], dtype=np.float32)
        render_stats.update({
            "camera_position": [float(x) for x in camera_pos],
            "endpoint_position": [float(x) for x in endpoint],
            "target_position": [float(x) for x in target_pos],
            "camera_target_distance": float(np.linalg.norm(to_target)),
            "camera_target_xy_distance": horizontal_norm,
            "camera_height_over_endpoint": float(camera_pos[2] - endpoint[2]),
            "target_height_delta_from_camera": float(target_pos[2] - camera_pos[2]),
        })
        if horizontal_norm > 1e-6:
            cos_angle = float(np.clip(np.dot(horizontal / horizontal_norm, forward_xy), -1.0, 1.0))
            render_stats["target_angle_xy_deg"] = float(np.degrees(np.arccos(cos_angle)))
        return render_stats

    def _render_goal_visibility_at_current_pose(self, goal, include_rgb=False, include_depth=False):
        semantic_label = self._goal_visibility_label(goal["instance"])
        goal["visibility_label"] = semantic_label
        self._apply_goal_visibility_label(goal["instance"], semantic_label)
        rendered = self._render_goal_instance_segmentation(include_rgb=include_rgb, include_depth=include_depth)
        if rendered is None:
            return 0, None, [], "render_failed"
        rendered["render_stats"] = self._goal_visibility_render_stats(goal, rendered)
        target_ids = self._goal_target_instance_ids(rendered["info"], goal["instance"], semantic_label)
        visible_pixels = int(np.isin(rendered["segmentation"], target_ids).sum()) if target_ids else 0
        if not target_ids:
            return visible_pixels, rendered, target_ids, "target_instance_not_in_segmentation"
        return visible_pixels, rendered, target_ids, None

    def _rendered_rgb_mean(self, rendered):
        if rendered is None or rendered.get("rgb") is None:
            return None
        return float(np.asarray(rendered["rgb"]).mean())

    def _goal_visibility_from_endpoint_candidates(self, goal, navmesh_interface, min_pixels=None, include_rgb=False, include_depth=False):
        if not self._flow("goal_visibility_enabled", False):
            visible_pixels = int(self._visibility_check("min_pixels", 150))
            return visible_pixels, None, [], None
        reference_path = np.asarray(goal["reference_path"], dtype=np.float32)
        best = {
            "visible_pixels": 0,
            "rendered": None,
            "target_ids": [],
            "reject_reason": "no_valid_visibility_endpoint",
            "reference_path": None,
            "endpoint_index": None,
        }
        for endpoint_index in self._goal_visibility_candidate_indices(reference_path):
            candidate_path = reference_path[:endpoint_index + 1]
            if len(candidate_path) < 2:
                continue
            candidate_goal = dict(goal)
            candidate_goal["reference_path"] = candidate_path.tolist()
            try:
                if not self._move_robot_to_goal_endpoint(candidate_goal, navmesh_interface):
                    reason = getattr(self, "_last_goal_visibility_reject_reason", "endpoint_invalid")
                    if best["visible_pixels"] == 0:
                        best["reject_reason"] = reason
                    continue
                visible_pixels, rendered, target_ids, reason = self._render_goal_visibility_at_current_pose(
                    candidate_goal, include_rgb=include_rgb, include_depth=include_depth
                )
            finally:
                self._reset_settle_state()
            if visible_pixels == 0 and self._is_black_far_visibility_view(rendered):
                best["reject_reason"] = "black_far_background"
                continue
            if target_ids and visible_pixels < int(min_pixels or 0):
                reason = "target_visible_area_too_small"
            rgb_mean = self._rendered_rgb_mean(rendered)
            best_rgb_mean = self._rendered_rgb_mean(best["rendered"])
            brighter_tie_break = (
                include_rgb
                and visible_pixels == best["visible_pixels"]
                and rgb_mean is not None
                and (best_rgb_mean is None or rgb_mean > best_rgb_mean)
            )
            if visible_pixels > best["visible_pixels"] or best["rendered"] is None or brighter_tie_break:
                best = {
                    "visible_pixels": int(visible_pixels),
                    "rendered": rendered,
                    "target_ids": target_ids,
                    "reject_reason": reason,
                    "visibility_endpoint_position": candidate_goal.get("visibility_endpoint_position"),
                    "visibility_aim_position": candidate_goal.get("visibility_aim_position"),
                    "visibility_yaw_angle": candidate_goal.get("visibility_yaw_angle"),
                    "endpoint_index": endpoint_index,
                }
            if min_pixels is not None and visible_pixels >= min_pixels:
                break
        if best["endpoint_index"] is not None:
            goal["visible_endpoint_index"] = best["endpoint_index"]
            for key in ("visibility_endpoint_position", "visibility_aim_position", "visibility_yaw_angle"):
                if best.get(key) is not None:
                    goal[key] = best[key]
        return best["visible_pixels"], best["rendered"], best["target_ids"], best["reject_reason"]

    def _goal_visible_pixels_from_endpoint(self, goal, navmesh_interface):
        min_pixels = int(self._visibility_check("min_pixels", 150))
        visible_pixels, _, target_ids, reject_reason = self._goal_visibility_from_endpoint_candidates(
            goal,
            navmesh_interface,
            min_pixels=min_pixels,
        )
        goal["visibility_reject_reason"] = reject_reason
        goal["visibility_target_ids"] = target_ids
        return visible_pixels

    def _filter_visible_goals(self, goals, navmesh_interface):
        if not self._flow("goal_visibility_enabled", False):
            return goals
        visible_goals = []
        visible_pixel_counts = []
        reject_reasons = Counter()
        min_pixels = int(self._visibility_check("min_pixels", 150))
        for goal in sorted(goals, key=lambda item: item["path_length"]):
            visible_pixels = self._goal_visible_pixels_from_endpoint(goal, navmesh_interface)
            visible_pixel_counts.append(int(visible_pixels))
            if visible_pixels >= min_pixels:
                goal["visible_pixels_at_goal"] = int(visible_pixels)
                visible_goals.append(goal)
            else:
                reject_reasons[goal.get("visibility_reject_reason") or "visibility_failed"] += 1
        if goals:
            print(
                f"[TG] Goal visibility kept {len(visible_goals)}/{len(goals)} candidates "
                f"(min_pixels_threshold={min_pixels}, max_pixels={max(visible_pixel_counts)}, "
                f"reject_reasons={dict(reject_reasons)})"
            )
        return visible_goals

    def sample_mixed_episodes(self):
        self.check_navmesh(self.cfg.scene['scene_id'])
        episode_mix = self._flow("episode_mix", {})
        placenav_count = int(episode_mix.get("placenav", 0))
        objnav_count = int(episode_mix.get("objnav", 0))
        current_placenav = sum(ep.get("task_type") == "placenav" for ep in self.generated_episodes)
        current_objnav = sum(ep.get("task_type") == "objnav" for ep in self.generated_episodes)
        placenav_remaining = max(0, placenav_count - current_placenav)
        objnav_remaining = max(0, objnav_count - current_objnav)
        placenav_goals = self.goal_dict_by_task.get("placenav", {})
        objnav_goals = self.goal_dict_by_task.get("objnav", {})
        if placenav_remaining and not placenav_goals:
            raise RuntimeError("No placenav store goals found")
        if objnav_remaining and not objnav_goals:
            raise RuntimeError("No objnav goals found")
        placenav_batch = 1 if placenav_remaining > 0 else 0
        objnav_batch = 1 if placenav_batch == 0 and objnav_remaining > 0 else 0
        print(
            f"[TG] Mixed episode remaining: placenav={placenav_remaining}, objnav={objnav_remaining}; "
            f"sampling batch placenav={placenav_batch}, objnav={objnav_batch}"
        )
        return [
            *self.sample_episodes(placenav_batch, goal_dict=placenav_goals, task_type="placenav", check_navmesh=False),
            *self.sample_episodes(objnav_batch, goal_dict=objnav_goals, task_type="objnav", check_navmesh=False),
        ]

    def sample_episodes(self, num_episodes, goal_dict=None, task_type="objnav", check_navmesh=True):
        if check_navmesh:
            self.check_navmesh(self.cfg.scene['scene_id'])
        navmesh_interface = self.navmesh_interface
        goal_dict = goal_dict if goal_dict is not None else self.goal_dict
        if task_type == "placenav" and bool((self._sample("placenav_distractor_route", {}) or {}).get("enabled", False)):
            return self.sample_placenav_distractor_episodes(
                num_episodes,
                goal_dict=goal_dict,
                navmesh_interface=navmesh_interface,
            )

        generated_episodes = []
        target_prim_counts = Counter(self.target_prim_counts)
        scene_id = self.cfg.scene["scene_id"]
        if getattr(self, "_target_first_blacklist_scene_id", None) != scene_id:
            self._target_first_blacklist_scene_id = scene_id
            self._target_first_blacklisted_target_prims = set()
        blacklisted_target_prims = self._target_first_blacklisted_target_prims
        pbar = tqdm(total=num_episodes, desc="Generating episodes")
        while len(generated_episodes) < num_episodes and not self.generate_finished:
            if self._max_episode_attempts_reached():
                self._skip_scene_after_max_attempts()
                break
            available_goal_names = [
                goal_name for goal_name, goal_item in goal_dict.items()
                if self._goal_item_has_available_prim(goal_item, target_prim_counts)
                and self._goal_name_allowed(goal_name)
            ]
            if not available_goal_names:
                print("[TG] No target prims remain below repeat limit")
                break
            path_found = False
            retry_count = 0
            goal_name = None
            target_first_retry_count = int(self._sample("target_first_retry_count", 100))
            while not path_found and retry_count < target_first_retry_count and not self.generate_finished:
                if self._max_episode_attempts_reached():
                    self._skip_scene_after_max_attempts()
                    break
                last_reject_description = None
                target_candidates = [
                    (name, goal_prim)
                    for name in available_goal_names
                    for goal_prim in goal_dict[name].get("prim", [])
                    if (
                        self._target_prim_available(self._goal_prim_path(goal_prim), target_prim_counts)
                        and str(self._goal_prim_path(goal_prim)) not in blacklisted_target_prims
                    )
                ]
                if not target_candidates:
                    pbar.set_description("[TG] No target prims remain below repeat limit")
                    pbar.close()
                    return generated_episodes
                goal_name, goal_prim = target_candidates[int(np.random.randint(len(target_candidates)))]
                prim_path = str(self._goal_prim_path(goal_prim))
                goal_pos = self._cached_goal_prim_position(goal_prim)
                max_goal_surface_z_gap = float(
                    self._sample("target_first_max_goal_surface_z_gap_m", 3.0)
                )
                goal_bbox = np.asarray(self.env.get_prim_bounding_box(prim_path), dtype=float)
                start_constraints = {
                    "z_range": (
                        float(goal_bbox[2]) - max_goal_surface_z_gap,
                        float(goal_bbox[5]) + max_goal_surface_z_gap,
                    )
                }
                radius_m = float(self._sample("target_search_radius_m", 50.0))
                local_start_count = int(self._sample("target_first_start_points", 100))
                blacklist_start_count = max(1, int(self._sample("target_blacklist_start_points", 100)))
                start_candidates = self.sample_valid_starts_near_point(
                    navmesh_interface,
                    goal_pos,
                    radius_m=radius_m,
                    num_points=local_start_count,
                    constraints=start_constraints,
                    start_min_wall_distance=float(self._sample("start_min_wall_distance", 0.0)),
                )
                if not start_candidates:
                    blacklisted_target_prims.add(prim_path)
                    retry_count += 1
                    pbar.set_description(
                        f"[TG] Blacklisted target {goal_name}: no valid starts within {radius_m:.0f}m"
                    )
                    continue
                attempted_starts = 0
                reference_failures = 0
                reference_failure_blacklist = int(self._sample("target_blacklist_reference_failures", 0) or 0)
                min_start_goal_xy_distance = float(
                    self._sample("target_first_min_start_goal_xy_distance", self.min_path_length)
                )
                for start_idx in np.random.permutation(len(start_candidates)):
                    attempted_starts += 1
                    start_pos = np.asarray(start_candidates[int(start_idx)], dtype=float)
                    if self._xy_distance(start_pos, goal_pos) < min_start_goal_xy_distance:
                        last_reject_description = (
                            f"[TG] Rejected near target-first start for goal {goal_name}: too_short"
                        )
                        continue
                    reject_total_before = sum(self.reference_path_reject_counts.values())
                    goals = []
                    goals.extend(self._build_goals_for_prim(start_pos, goal_prim, task_type, navmesh_interface))
                    goals = self._filter_visible_goals(goals, navmesh_interface)
                    goals = self._filter_goals_by_target_prim_limit(goals, target_prim_counts)
                    if len(goals) == 0:
                        last_reject_description = f'[TG] Rejected target-first start for goal {goal_name}'
                        if sum(self.reference_path_reject_counts.values()) > reject_total_before:
                            reference_failures += 1
                            if reference_failure_blacklist > 0 and reference_failures >= reference_failure_blacklist:
                                blacklisted_target_prims.add(prim_path)
                                last_reject_description = (
                                    f"[TG] Blacklisted target {goal_name}: "
                                    f"{reference_failures} A* reference failures"
                                )
                                break
                        continue

                    closest_goal_idx = int(np.argmin([x['path_length'] for x in goals]))
                    closest_goal = goals[closest_goal_idx]
                    yaw_angle = calc_yaw(start_pos[:2], closest_goal['reference_path'][1][:2])
                    yaw_angle += np.random.uniform(-np.deg2rad(30.0), np.deg2rad(30.0))
                    settles, settle_info = self._task_generation_start_settles(
                        start_pos,
                        closest_goal["reference_path"],
                        yaw_angle,
                    )
                    if settles:
                        path_found = True
                        break
                    last_reject_description = (
                        f'[TG] Rejected unstable target-first start for goal {goal_name}: '
                        f'{settle_info.get("termination_reason", "settle")}'
                    )
                if not path_found:
                    retry_count += 1
                    if last_reject_description:
                        pbar.set_description(last_reject_description)
                    if attempted_starts >= blacklist_start_count:
                        blacklisted_target_prims.add(prim_path)
                        pbar.set_description(
                            f"[TG] Blacklisted target {goal_name}: {attempted_starts} local starts failed"
                        )
            if path_found:
                pbar.set_description(f'[TG] Found {len(goals)} paths for goal {goal_name}')
                # set initial yaw angle to the next waypoint and add some noise
                quat_xyzw = R.from_euler('z', yaw_angle).as_quat().tolist()
                quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
                episode_kwargs = {
                    "objnav": goal_name,
                    "instruction": "",
                    "episode_id": len(generated_episodes),
                    "goals": goals,
                    "start_position": start_pos.tolist(),
                    "start_rotation": quat_wxyz,
                    "closest_goal_idx": closest_goal_idx,
                    "task_type": task_type,
                    "start_settle": settle_info,
                    "task_generation_strict_endpoint": task_type == "placenav",
                    "task_generation_path_endpoint_tolerance_m": self._goal_endpoint_tolerance(
                        closest_goal["radius"],
                        task_type,
                    ),
                    "task_generation_follower_term_dist_m": float(
                        self._check("follower_term_dist_m", 0.3)
                    ),
                    "task_generation_require_astar_path": self._trajectory_planner_name(for_sampling=True) == "astar",
                }
                if task_type == "placenav":
                    episode_kwargs.update({
                        "placenav_goal": goal_name,
                        "placenav_category": "store",
                        "instruction": goal_name.replace("_", " "),
                    })
                episode = VLNEpisode(
                    data=self._episode_base_data(),
                    **episode_kwargs,
                )
                generated_episodes.append(episode)
                target_prim = self._episode_target_prim(episode)
                if target_prim:
                    target_prim_counts[target_prim] += 1
                pbar.update(1)
            else:
                pbar.set_description(f'[TG] Failed to find paths for goal {goal_name}')
        pbar.close()
        return generated_episodes

    def _placenav_distractor_config(self):
        config = self._sample("placenav_distractor_route", {}) or {}
        return {
            "min_distractors": int(config.get("min_distractors", 1)),
            "max_distractors": int(config.get("max_distractors", 3)),
            "max_route_attempts": int(config.get("max_route_attempts", 100)),
            "min_stop_spacing_m": float(config.get("min_stop_spacing_m", 5.0)),
            "max_target_distance_m": float(config.get("max_target_distance_m", 25.0)),
            "min_total_path_length": float(config.get("min_total_path_length", self._sample("min_path_length", 5.0))),
            "max_total_path_length": float(config.get("max_total_path_length", 80.0)),
            "require_different_goal_name": bool(config.get("require_different_goal_name", True)),
            "require_stop_visibility": bool(config.get("require_stop_visibility", True)),
            "require_distractor_visibility": bool(
                config.get("require_distractor_visibility", config.get("require_stop_visibility", True))
            ),
            "require_target_visibility": bool(
                config.get("require_target_visibility", config.get("require_stop_visibility", True))
            ),
            "distractor_endpoint_tolerance_m": float(config.get("distractor_endpoint_tolerance_m", 4.0)),
            "target_endpoint_tolerance_m": float(
                config.get("target_endpoint_tolerance_m", self._sample("placenav_goal_tolerance", 1.0))
            ),
            "max_leg_path_length": float(config.get("max_leg_path_length", config.get("max_total_path_length", 80.0))),
            "min_leg_path_length": float(config.get("min_leg_path_length", 0.25)),
            "adaptive_distractor_count": bool(config.get("adaptive_distractor_count", True)),
            "visit_point_radius_m": float(config.get("visit_point_radius_m", 8.0)),
            "visit_point_candidates": int(config.get("visit_point_candidates", 32)),
            "visit_pair_candidates": int(config.get("visit_pair_candidates", 12)),
            "visit_point_min_wall_distance_m": float(config.get("visit_point_min_wall_distance_m", 0.5)),
            "visit_endpoint_tolerance_m": float(config.get("visit_endpoint_tolerance_m", 1.0)),
            "start_path_tolerance_m": float(config.get("start_path_tolerance_m", 2.0)),
            "target_attempts_per_route": int(config.get("target_attempts_per_route", 12)),
            "distractor_store_pass_radius_m": float(config.get("distractor_store_pass_radius_m", 20.0)),
            "target_store_pass_radius_m": float(config.get("target_store_pass_radius_m", 20.0)),
        }

    def _placenav_store_candidates(self, goal_dict):
        candidates = []
        for goal_name, goal_item in goal_dict.items():
            if not self._goal_name_allowed(goal_name):
                continue
            for goal_prim in goal_item.get("prim", []):
                prim_path = str(self._goal_prim_path(goal_prim))
                candidates.append({
                    "goal_name": goal_name,
                    "goal_prim": goal_prim,
                    "prim_path": prim_path,
                    "position": self._cached_goal_prim_bbox_center(goal_prim),
                })
        return candidates

    def _placenav_store_pair_reachable(self, start_store, goal_store, navmesh_interface, config):
        return self._placenav_store_pair_path(
            start_store,
            goal_store,
            navmesh_interface,
            config,
            goal_role="target",
        ) is not None

    def _placenav_store_pair_path(
        self,
        start_store,
        goal_store,
        navmesh_interface,
        config,
        goal_role="distractor",
    ):
        scene_id = self.cfg.scene["scene_id"]
        start_store_radius = float(config["distractor_store_pass_radius_m"])
        goal_store_radius = float(
            config[
                "target_store_pass_radius_m"
                if goal_role == "target"
                else "distractor_store_pass_radius_m"
            ]
        )
        endpoint_tolerance = float(
            config[
                "target_endpoint_tolerance_m"
                if goal_role == "target"
                else "distractor_endpoint_tolerance_m"
            ]
        )
        cache_key = (
            scene_id,
            start_store["prim_path"],
            goal_store["prim_path"],
            goal_role,
            start_store_radius,
            goal_store_radius,
            endpoint_tolerance,
            float(config["max_leg_path_length"]),
        )
        if getattr(self, "_placenav_pair_reachability_cache_scene_id", None) != scene_id:
            self._placenav_pair_reachability_cache_scene_id = scene_id
            self._placenav_pair_reachability_cache = {}
        cache = self._placenav_pair_reachability_cache
        if cache_key in cache:
            return cache[cache_key]

        pair_candidates = max(1, int(config["visit_pair_candidates"]))
        start_points = [
            point
            for point in self._placenav_visit_point_candidates(start_store, navmesh_interface, config)
            if self._xy_distance(point, start_store["position"]) <= start_store_radius
        ][:pair_candidates]
        goal_points = [
            point
            for point in self._placenav_visit_point_candidates(goal_store, navmesh_interface, config)
            if self._xy_distance(point, goal_store["position"]) <= goal_store_radius
        ][:pair_candidates]
        for start_point in start_points:
            for goal_point in goal_points:
                path = np.asarray(navmesh_interface.find_paths(start_point, goal_point), dtype=float)
                if len(path) < 2:
                    continue
                start_gap = float(np.linalg.norm(path[0] - start_point))
                if start_gap > config["start_path_tolerance_m"]:
                    continue
                start_store_gap = self._xy_distance(path[0], start_store["position"])
                if start_store_gap > start_store_radius:
                    continue
                endpoint_gap = self._xy_distance(path[-1], goal_point)
                store_gap = self._xy_distance(path[-1], goal_store["position"])
                if endpoint_gap > endpoint_tolerance or store_gap > goal_store_radius:
                    continue
                path_length = self._reference_path_length(path)
                if path_length > config["max_leg_path_length"]:
                    continue
                result = {
                    "path": path,
                    "start_point": np.asarray(start_point, dtype=float),
                    "goal_point": np.asarray(path[-1], dtype=float),
                    "path_length": float(path_length),
                    "goal_path_gap_m": float(endpoint_gap),
                    "store_visit_gap_m": float(store_gap),
                    "start_store_visit_gap_m": float(start_store_gap),
                }
                cache[cache_key] = result
                return result
        cache[cache_key] = None
        return None

    def _select_distractor_stops(self, target, candidates, config, navmesh_interface):
        pool = [
            candidate for candidate in candidates
            if candidate["prim_path"] != target["prim_path"]
            and (
                not config["require_different_goal_name"]
                or candidate["goal_name"] != target["goal_name"]
            )
            and self._xy_distance(candidate["position"], target["position"]) <= config["max_target_distance_m"]
            and self._placenav_store_pair_reachable(candidate, target, navmesh_interface, config)
        ]
        if not pool:
            return []
        selected = []
        for idx in np.random.permutation(len(pool)):
            candidate = pool[int(idx)]
            if any(self._xy_distance(candidate["position"], stop["position"]) < config["min_stop_spacing_m"] for stop in selected):
                continue
            if self._xy_distance(candidate["position"], target["position"]) < config["min_stop_spacing_m"]:
                continue
            selected.append(candidate)
            if len(selected) >= config["max_distractors"]:
                break
        max_count = min(config["max_distractors"], len(selected))
        min_count = min(config["min_distractors"], max_count)
        if min_count <= 0:
            return []
        if config["adaptive_distractor_count"]:
            count = int(np.random.randint(min_count, max_count + 1))
        else:
            requested_max = min(config["max_distractors"], len(pool))
            count = int(np.random.randint(min_count, requested_max + 1))
            if len(selected) < count:
                return []
        return selected[:count]

    @staticmethod
    def _project_to_navmesh_surface(center, vertices, faces, max_points=8):
        center = np.asarray(center, dtype=float)
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        if center.shape[0] < 3 or vertices.ndim != 2 or vertices.shape[1] < 3:
            return []
        if faces.ndim == 1:
            if len(faces) % 3:
                return []
            faces = faces.reshape(-1, 3)
        if faces.ndim != 2 or faces.shape[1] < 3 or len(faces) == 0:
            return []

        faces = faces[:, :3]
        valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
        triangles = vertices[faces[valid]]
        if len(triangles) == 0:
            return []

        point_xy = center[:2]
        edge_starts = triangles[:, [0, 1, 2]]
        edge_ends = triangles[:, [1, 2, 0]]
        edge_delta = edge_ends - edge_starts
        edge_xy = edge_delta[:, :, :2]
        edge_length_sq = np.sum(edge_xy * edge_xy, axis=2)
        point_delta = point_xy[None, None, :] - edge_starts[:, :, :2]
        edge_t = np.divide(
            np.sum(point_delta * edge_xy, axis=2),
            edge_length_sq,
            out=np.zeros_like(edge_length_sq),
            where=edge_length_sq > 1e-12,
        )
        edge_t = np.clip(edge_t, 0.0, 1.0)
        edge_points = edge_starts + edge_t[:, :, None] * edge_delta
        edge_distance_sq = np.sum((edge_points[:, :, :2] - point_xy) ** 2, axis=2)
        nearest_edge = np.argmin(edge_distance_sq, axis=1)
        projected = edge_points[np.arange(len(triangles)), nearest_edge]

        a = triangles[:, 0, :2]
        ab = triangles[:, 1, :2] - a
        ac = triangles[:, 2, :2] - a
        ap = point_xy[None, :] - a
        denominator = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        nondegenerate = np.abs(denominator) > 1e-12
        weight_b = np.zeros(len(triangles), dtype=float)
        weight_c = np.zeros(len(triangles), dtype=float)
        weight_b[nondegenerate] = (
            ap[nondegenerate, 0] * ac[nondegenerate, 1]
            - ap[nondegenerate, 1] * ac[nondegenerate, 0]
        ) / denominator[nondegenerate]
        weight_c[nondegenerate] = (
            ab[nondegenerate, 0] * ap[nondegenerate, 1]
            - ab[nondegenerate, 1] * ap[nondegenerate, 0]
        ) / denominator[nondegenerate]
        inside = (
            nondegenerate
            & (weight_b >= -1e-8)
            & (weight_c >= -1e-8)
            & (weight_b + weight_c <= 1.0 + 1e-8)
        )
        projected[inside] = (
            triangles[inside, 0]
            + weight_b[inside, None] * (triangles[inside, 1] - triangles[inside, 0])
            + weight_c[inside, None] * (triangles[inside, 2] - triangles[inside, 0])
        )

        xy_distance = np.linalg.norm(projected[:, :2] - point_xy, axis=1)
        z_distance = np.abs(projected[:, 2] - center[2])
        order = np.lexsort((z_distance, xy_distance))
        result = []
        seen = set()
        for idx in order:
            point = projected[int(idx)]
            if not np.isfinite(point).all():
                continue
            key = tuple(np.round(point, 3).tolist())
            if key in seen:
                continue
            seen.add(key)
            result.append(point)
            if len(result) >= max(1, int(max_points)):
                break
        return result

    def _placenav_visit_point_candidates(self, store, navmesh_interface, config):
        scene_id = self.cfg.scene["scene_id"]
        cache_key = (
            scene_id,
            store["prim_path"],
            float(config["visit_point_radius_m"]),
            int(config["visit_point_candidates"]),
            float(config["visit_point_min_wall_distance_m"]),
        )
        if getattr(self, "_placenav_visit_point_cache_scene_id", None) != scene_id:
            self._placenav_visit_point_cache_scene_id = scene_id
            self._placenav_visit_point_cache = {}
        cache = self._placenav_visit_point_cache
        if cache_key in cache:
            return cache[cache_key]

        center = np.asarray(store["position"], dtype=float)
        visit_points = []
        vertices = np.asarray(getattr(navmesh_interface, "navmesh_v", []), dtype=float)
        if len(vertices):
            faces = np.asarray(getattr(navmesh_interface, "navmesh_t", []), dtype=int)
            surface_limit = min(8, max(1, int(config["visit_pair_candidates"])))
            visit_points.extend(
                self._project_to_navmesh_surface(
                    center,
                    vertices,
                    faces,
                    max_points=surface_limit,
                )
            )
            distances = np.linalg.norm(vertices[:, :2] - center[:2], axis=1)
            sorted_indices = np.argsort(distances)
            for idx in sorted_indices:
                if distances[int(idx)] > config["visit_point_radius_m"] and visit_points:
                    break
                visit_points.append(np.asarray(vertices[int(idx)], dtype=float))
                if len(visit_points) >= config["visit_point_candidates"]:
                    break
            if not visit_points:
                for idx in sorted_indices[:config["visit_point_candidates"]]:
                    visit_points.append(np.asarray(vertices[int(idx)], dtype=float))
        unique = []
        seen = set()
        for point in visit_points:
            point = np.asarray(point, dtype=float)
            key = tuple(np.round(point, 3).tolist())
            if key in seen:
                continue
            seen.add(key)
            unique.append(point)
        cache[cache_key] = unique
        return unique

    def _build_placenav_visit_goal(self, current_start, stop, role, navmesh_interface, config, visit_points_override=None):
        visit_points = (
            [np.asarray(point, dtype=float) for point in visit_points_override]
            if visit_points_override is not None
            else self._placenav_visit_point_candidates(stop, navmesh_interface, config)
        )
        if not visit_points:
            return None
        prim_path = stop["prim_path"]
        obj_radius = self.env.get_prim_radius(prim_path)
        bbox = [float(value) for value in self.env.get_prim_bounding_box(prim_path)]
        bbox_array = np.asarray(bbox, dtype=float)
        if bbox_array.shape[0] >= 6 and np.isfinite(bbox_array).all():
            target_center = 0.5 * (bbox_array[:3] + bbox_array[3:6])
        else:
            target_center = np.asarray(self.env.get_prim_position(prim_path), dtype=float)
        max_store_gap = (
            config["target_store_pass_radius_m"]
            if role == "target"
            else config["distractor_store_pass_radius_m"]
        )
        astar_goal_radius = max(0.0, max_store_gap - 1.0)
        for idx in np.random.permutation(len(visit_points)):
            visit_point = np.asarray(visit_points[int(idx)], dtype=float)
            path = self._find_reference_path(
                current_start,
                visit_point,
                navmesh_interface,
                goal_radius=astar_goal_radius,
            )
            path_array = np.asarray(path, dtype=float)
            validation = self.validate_reference_path(
                current_start,
                visit_point,
                path,
                {
                    "navmesh_interface": navmesh_interface,
                    "goal_radius": 0.0,
                    "task_type": "placenav",
                    "endpoint_tolerance_m": (
                        config["target_endpoint_tolerance_m"]
                        if role == "target"
                        else config["distractor_endpoint_tolerance_m"]
                    ),
                    "start_path_tolerance_m": config["start_path_tolerance_m"],
                    "min_path_length_m": config["min_leg_path_length"],
                    "max_path_length_m": config["max_leg_path_length"],
                    "goal_clearance_m": float(self._sample("goal_clearance_m", 0.0) or 0.0),
                },
            )
            snapped_to_endpoint = False
            store_gap = float("inf")
            if not validation["accepted"] and len(path_array) >= 2:
                path_length = self._reference_path_length(path_array)
                store_gap = self._xy_distance(path_array[-1], target_center)
                if (
                    path_length >= config["min_leg_path_length"]
                    and path_length <= config["max_leg_path_length"]
                    and store_gap <= max_store_gap
                    and float(np.linalg.norm(np.asarray(current_start, dtype=float) - path_array[0]))
                    <= config["start_path_tolerance_m"]
                ):
                    validation = {
                        "accepted": True,
                        "reference_path": path_array,
                        "reference_path_length_m": path_length,
                        "goal_path_gap_m": self._xy_distance(path_array[-1], visit_point),
                        "goal_path_gap_3d_m": float(np.linalg.norm(path_array[-1] - visit_point)),
                        "goal_path_gap_z_m": self._z_gap(path_array[-1], visit_point),
                    }
                    visit_point = path_array[-1]
                    snapped_to_endpoint = True
            if not validation["accepted"]:
                reason = validation.get("reject_reason") or "unknown"
                self.placenav_distractor_reject_counts[f"{role}_leg_{reason}"] += 1
                debug_count = int(getattr(self, "_placenav_visit_debug_examples", 0) or 0)
                if reason in {"goal_path_gap", "start_path_gap"} and debug_count < 5:
                    path_end = path_array[-1].tolist() if len(path_array) else None
                    print(
                        "[TG] Distractor visit debug: "
                        f"role={role} reason={reason} "
                        f"start={np.round(np.asarray(current_start, dtype=float), 2).tolist()} "
                        f"visit={np.round(visit_point, 2).tolist()} "
                        f"path_end={None if path_end is None else np.round(path_array[-1], 2).tolist()} "
                        f"goal_gap={validation.get('goal_path_gap_m')} "
                        f"start_gap={validation.get('start_path_gap_m')} "
                        f"store_center={np.round(target_center, 2).tolist()}"
                    )
                    self._placenav_visit_debug_examples = debug_count + 1
                continue
            visit_gap = self._xy_distance(visit_point, target_center)
            return {
                "instance": prim_path,
                "type": "place",
                "location": visit_point.tolist(),
                "target_center": target_center.tolist(),
                "bbox": bbox,
                "radius": obj_radius,
                "path_length": float(validation["reference_path_length_m"]),
                "reference_path": np.asarray(validation["reference_path"], dtype=float).tolist(),
                "goal_path_gap_m": float(validation.get("goal_path_gap_m", 0.0)),
                "goal_path_gap_3d_m": float(validation.get("goal_path_gap_3d_m", 0.0)),
                "goal_path_gap_z_m": float(validation.get("goal_path_gap_z_m", 0.0)),
                "placenav_stop_role": role,
                "placenav_store_visit_gap_m": float(store_gap if np.isfinite(store_gap) else visit_gap),
                "placenav_original_visit_point": np.asarray(visit_points[int(idx)], dtype=float).tolist(),
                "placenav_snapped_to_reachable_endpoint": snapped_to_endpoint,
                "reference_path_planner": "astar" if self._trajectory_planner_name(for_sampling=True) == "astar" else "navmesh",
            }
        return None

    def _placenav_route_start_candidates(
        self,
        first_stop,
        navmesh_interface,
        config,
        radius_m,
        num_points,
        min_distance_m,
        center_override=None,
        visit_points_override=None,
    ):
        center = np.asarray(center_override if center_override is not None else first_stop["position"], dtype=float)
        vertices = np.asarray(getattr(navmesh_interface, "navmesh_v", []), dtype=float)
        if len(vertices) == 0:
            return []
        distances = np.linalg.norm(vertices[:, :2] - center[:2], axis=1)
        candidate_indices = np.where((distances <= float(radius_m)) & (distances >= float(min_distance_m)))[0]
        if len(candidate_indices) == 0:
            candidate_indices = np.where(distances <= float(radius_m))[0]
        if len(candidate_indices) == 0:
            candidate_indices = np.argsort(distances)[:max(1, int(num_points))]
        shuffled = candidate_indices[np.random.permutation(len(candidate_indices))]
        if visit_points_override is not None:
            visit_points = [np.asarray(point, dtype=float) for point in visit_points_override]
        else:
            pair_candidates = max(1, int(config["visit_pair_candidates"]))
            visit_points = self._placenav_visit_point_candidates(first_stop, navmesh_interface, config)[:pair_candidates]
        starts = []
        seen = set()
        for idx in shuffled[:max(int(num_points) * 12, int(num_points))]:
            point = np.asarray(vertices[int(idx)], dtype=float)
            key = tuple(np.round(point, 3).tolist())
            if key in seen:
                continue
            reachable = False
            for visit_point in visit_points:
                path = np.asarray(navmesh_interface.find_paths(point, visit_point), dtype=float)
                if len(path) < 2:
                    continue
                path_length = self._reference_path_length(path)
                if path_length < config["min_leg_path_length"] or path_length > config["max_leg_path_length"]:
                    continue
                endpoint_gap = self._xy_distance(path[-1], visit_point)
                store_gap = self._xy_distance(path[-1], first_stop["position"])
                if visit_points_override is not None:
                    reachable = endpoint_gap <= config["start_path_tolerance_m"]
                else:
                    reachable = (
                        endpoint_gap <= config["distractor_endpoint_tolerance_m"]
                        or store_gap <= config["distractor_store_pass_radius_m"]
                    )
                if reachable:
                    reachable = True
                    break
            if not reachable:
                continue
            seen.add(key)
            starts.append(point)
            if len(starts) >= int(num_points):
                break
        return starts

    @staticmethod
    def _append_reference_segment(composite_path, segment):
        segment = np.asarray(segment, dtype=float)
        if len(segment) == 0:
            return composite_path
        if not composite_path:
            composite_path.extend(point.tolist() for point in segment)
            return composite_path
        start_idx = 1 if np.linalg.norm(np.asarray(composite_path[-1], dtype=float) - segment[0]) < 1e-4 else 0
        composite_path.extend(point.tolist() for point in segment[start_idx:])
        return composite_path

    def _select_placenav_distractor_chain(self, target, pool, desired_count, min_count, navmesh_interface, config):
        desired_count = min(int(desired_count), len(pool))
        min_count = max(1, min(int(min_count), desired_count))
        for count in range(desired_count, min_count - 1, -1):
            distractors_reversed = []
            pair_routes_reversed = []
            used_prims = {target["prim_path"]}
            used_goal_names = {target["goal_name"]}
            current = target
            for _ in range(count):
                selected = None
                selected_route = None
                for idx in np.random.permutation(len(pool)):
                    candidate = pool[int(idx)]
                    if candidate["prim_path"] in used_prims or candidate["goal_name"] in used_goal_names:
                        continue
                    if any(
                        self._xy_distance(candidate["position"], stop["position"]) < config["min_stop_spacing_m"]
                        for stop in [target, *distractors_reversed]
                    ):
                        continue
                    goal_role = "target" if current["prim_path"] == target["prim_path"] else "distractor"
                    pair_route = self._placenav_store_pair_path(
                        candidate,
                        current,
                        navmesh_interface,
                        config,
                        goal_role=goal_role,
                    )
                    if pair_route is None:
                        continue
                    selected = candidate
                    selected_route = pair_route
                    break
                if selected is None:
                    break
                distractors_reversed.append(selected)
                pair_routes_reversed.append(selected_route)
                used_prims.add(selected["prim_path"])
                used_goal_names.add(selected["goal_name"])
                current = selected
            if len(distractors_reversed) >= min_count:
                return list(reversed(distractors_reversed)), list(reversed(pair_routes_reversed))
        return None, None

    def _build_placenav_distractor_route(self, scene_id, goal_dict, target_prim_counts, navmesh_interface, config):
        reject_counts = Counter()
        candidates = self._placenav_store_candidates(goal_dict)
        if len(candidates) < config["min_distractors"] + 1:
            self.placenav_distractor_reject_counts["not_enough_candidates"] += 1
            return None
        available_target_indices = [
            idx for idx, candidate in enumerate(candidates)
            if self._target_prim_available(candidate["prim_path"], target_prim_counts)
        ]
        if not available_target_indices:
            reject_counts["no_available_targets"] += 1
            self.placenav_distractor_reject_counts.update(reject_counts)
            return None
        target_order = np.asarray(available_target_indices, dtype=int)[
            np.random.permutation(len(available_target_indices))
        ][:max(1, config["target_attempts_per_route"])]
        radius_m = float(self._sample("target_search_radius_m", 50.0))
        local_start_count = int(self._sample("target_first_start_points", 100))
        min_start_goal_xy_distance = float(self._sample("target_first_min_start_goal_xy_distance", self.min_path_length))
        for target_idx in target_order:
            target = candidates[int(target_idx)]
            pool = [
                candidate for candidate in candidates
                if candidate["prim_path"] != target["prim_path"]
                and (
                    not config["require_different_goal_name"]
                    or candidate["goal_name"] != target["goal_name"]
                )
                and self._xy_distance(candidate["position"], target["position"]) <= config["max_target_distance_m"]
                and self._placenav_store_pair_reachable(candidate, target, navmesh_interface, config)
            ]
            if not pool:
                reject_counts["no_distractors_for_target"] += 1
                continue

            max_count = min(config["max_distractors"], len(pool))
            min_count = min(config["min_distractors"], max_count)
            if min_count <= 0:
                reject_counts["no_distractors_for_target"] += 1
                continue
            desired_count = int(np.random.randint(min_count, max_count + 1))
            distractors, pair_routes = self._select_placenav_distractor_chain(
                target,
                pool,
                desired_count,
                min_count,
                navmesh_interface,
                config,
            )
            if not distractors:
                reject_counts["no_distractor_chain"] += 1
                continue

            first_stop = distractors[0]
            first_pair_route = pair_routes[0]
            start_candidates = self._placenav_route_start_candidates(
                first_stop,
                navmesh_interface,
                config,
                radius_m=radius_m,
                num_points=local_start_count,
                min_distance_m=min_start_goal_xy_distance,
                center_override=first_pair_route["start_point"],
                visit_points_override=[first_pair_route["start_point"]],
            )
            if not start_candidates:
                reject_counts["no_start_candidates"] += 1
                continue

            for start_idx in np.random.permutation(len(start_candidates)):
                start_pos = np.asarray(start_candidates[int(start_idx)], dtype=float)
                if self._xy_distance(start_pos, first_stop["position"]) < min_start_goal_xy_distance:
                    reject_counts["start_too_close"] += 1
                    continue
                first_segment = np.asarray(
                    navmesh_interface.find_paths(start_pos, first_pair_route["start_point"]),
                    dtype=float,
                )
                if len(first_segment) < 2:
                    reject_counts["no_start_pair_path"] += 1
                    continue
                first_start_gap = float(np.linalg.norm(first_segment[0] - start_pos))
                first_goal_gap = self._xy_distance(first_segment[-1], first_pair_route["start_point"])
                first_store_gap = self._xy_distance(first_segment[-1], first_stop["position"])
                if (
                    first_start_gap > config["start_path_tolerance_m"]
                    or first_goal_gap > config["start_path_tolerance_m"]
                    or first_store_gap > config["distractor_store_pass_radius_m"]
                ):
                    reject_counts["start_pair_path_gap"] += 1
                    continue
                if first_start_gap > 1e-4:
                    first_segment = np.vstack([np.asarray(start_pos, dtype=float), first_segment])
                first_length = self._reference_path_length(first_segment)
                if first_length < config["min_leg_path_length"] or first_length > config["max_leg_path_length"]:
                    reject_counts[
                        "distractor_leg_too_short"
                        if first_length < config["min_leg_path_length"]
                        else "distractor_leg_too_long"
                    ] += 1
                    continue

                composite_path = []
                self._append_reference_segment(composite_path, first_segment)
                route_stops = []
                route_stops.append({
                    "role": "distractor",
                    "order": 0,
                    "goal_name": first_stop["goal_name"],
                    "instance": first_stop["prim_path"],
                    "location": first_segment[-1].tolist(),
                    "target_center": first_stop["position"].tolist(),
                    "reach_tolerance_m": float(config["distractor_store_pass_radius_m"]),
                    "segment_length_m": float(first_length),
                    "store_visit_gap_m": float(first_store_gap),
                })
                for stop_index, pair_route in enumerate(pair_routes, start=1):
                    pair_path = np.asarray(pair_route["path"], dtype=float)
                    pair_segment_length = self._reference_path_length(
                        np.vstack([np.asarray(composite_path[-1], dtype=float), pair_path])
                    )
                    self._append_reference_segment(composite_path, pair_path)
                    next_role = "target" if stop_index == len(pair_routes) else "distractor"
                    next_stop = target if next_role == "target" else distractors[stop_index]
                    route_stops.append({
                        "role": next_role,
                        "order": stop_index,
                        "goal_name": next_stop["goal_name"],
                        "instance": next_stop["prim_path"],
                        "location": pair_path[-1].tolist(),
                        "target_center": next_stop["position"].tolist(),
                        "reach_tolerance_m": float(
                            config[
                                "target_store_pass_radius_m"
                                if next_role == "target"
                                else "distractor_store_pass_radius_m"
                            ]
                        ),
                        "segment_length_m": float(pair_segment_length),
                        "store_visit_gap_m": float(pair_route["store_visit_gap_m"]),
                    })

                total_path_length = self._reference_path_length(composite_path)
                min_total_path_length = float(config["min_total_path_length"])
                if total_path_length < min_total_path_length or total_path_length > config["max_total_path_length"]:
                    reject_counts[
                        "total_path_too_short" if total_path_length < min_total_path_length else "total_path_too_long"
                    ] += 1
                    continue
                reference_path = np.asarray(composite_path, dtype=float)
                yaw_angle = calc_yaw(reference_path[0][:2], reference_path[1][:2])
                yaw_angle += np.random.uniform(-np.deg2rad(30.0), np.deg2rad(30.0))
                settles, settle_info = self._task_generation_start_settles(start_pos, reference_path, yaw_angle)
                if not settles:
                    reject_counts["start_settle_failed"] += 1
                    continue

                final_pair_route = pair_routes[-1]
                final_path = np.asarray(final_pair_route["path"], dtype=float)
                prim_path = target["prim_path"]
                obj_radius = self.env.get_prim_radius(prim_path)
                bbox = [float(value) for value in self.env.get_prim_bounding_box(prim_path)]
                final_goal = {
                    "instance": prim_path,
                    "type": "place",
                    "location": final_path[-1].tolist(),
                    "target_center": np.asarray(target["position"], dtype=float).tolist(),
                    "bbox": bbox,
                    "radius": obj_radius,
                    "path_length": float(total_path_length),
                    "reference_path": reference_path.tolist(),
                    "goal_path_gap_m": float(final_pair_route["goal_path_gap_m"]),
                    "goal_path_gap_3d_m": float(np.linalg.norm(final_path[-1] - final_pair_route["goal_point"])),
                    "goal_path_gap_z_m": self._z_gap(final_path[-1], final_pair_route["goal_point"]),
                    "placenav_stop_role": "target",
                    "placenav_store_visit_gap_m": float(final_pair_route["store_visit_gap_m"]),
                    "placenav_original_visit_point": np.asarray(final_pair_route["goal_point"], dtype=float).tolist(),
                    "placenav_snapped_to_reachable_endpoint": True,
                    "reference_path_planner": (
                        "astar" if self._trajectory_planner_name(for_sampling=True) == "astar" else "navmesh"
                    ),
                }
                return {
                    "target": target,
                    "distractors": distractors,
                    "goal": final_goal,
                    "route_stops": route_stops,
                    "start_pos": start_pos,
                    "yaw_angle": yaw_angle,
                    "settle_info": settle_info,
                }
        self.placenav_distractor_reject_counts.update(reject_counts)
        return None

    def sample_placenav_distractor_episodes(self, num_episodes, goal_dict, navmesh_interface):
        generated_episodes = []
        target_prim_counts = Counter(self.target_prim_counts)
        scene_id = self.cfg.scene["scene_id"]
        config = self._placenav_distractor_config()
        self.placenav_distractor_reject_counts = Counter()
        pbar = tqdm(total=num_episodes, desc="Generating distractor placenav episodes")
        while len(generated_episodes) < num_episodes and not self.generate_finished:
            if self._max_episode_attempts_reached():
                self._skip_scene_after_max_attempts()
                break
            route = None
            for _ in range(max(1, config["max_route_attempts"])):
                route = self._build_placenav_distractor_route(
                    scene_id,
                    goal_dict,
                    target_prim_counts,
                    navmesh_interface,
                    config,
                )
                if route is not None:
                    break
            if route is None:
                self._last_sample_failure_reason = "placenav_distractor_route_failed"
                total_rejects = sum(self.placenav_distractor_reject_counts.values())
                if total_rejects:
                    print(
                        "[TG] Distractor route rejects: "
                        f"total={total_rejects} counts={dict(self.placenav_distractor_reject_counts)}"
                    )
                keep_sampling = self._record_episode_attempt(
                    "sample_episode_failed",
                    reason="placenav_distractor_route_failed",
                    skip_on_reach=True,
                )
                if not keep_sampling:
                    break
                continue
            target = route["target"]
            goal = route["goal"]
            quat_xyzw = R.from_euler('z', route["yaw_angle"]).as_quat().tolist()
            quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            episode = VLNEpisode(
                data=self._episode_base_data(),
                objnav=target["goal_name"],
                instruction=target["goal_name"].replace("_", " "),
                episode_id=len(generated_episodes),
                goals=[goal],
                start_position=route["start_pos"].tolist(),
                start_rotation=quat_wxyz,
                closest_goal_idx=0,
                task_type="placenav",
                placenav_goal=target["goal_name"],
                placenav_category="store",
                placenav_route_type="distractor_stores",
                placenav_target_store={
                    "goal_name": target["goal_name"],
                    "instance": target["prim_path"],
                    "location": target["position"].tolist(),
                    "visit_location": goal["location"],
                },
                placenav_distractors=[
                    {
                        "goal_name": distractor["goal_name"],
                        "instance": distractor["prim_path"],
                        "location": distractor["position"].tolist(),
                    }
                    for distractor in route["distractors"]
                ],
                placenav_route_stops=route["route_stops"],
                placenav_distractor_reach_tolerance_m=float(config["distractor_store_pass_radius_m"]),
                placenav_target_reach_tolerance_m=float(config["target_store_pass_radius_m"]),
                start_settle=route["settle_info"],
                task_generation_preserve_reference_path=True,
                task_generation_strict_endpoint=True,
                task_generation_path_endpoint_tolerance_m=self._goal_endpoint_tolerance(goal["radius"], "placenav"),
                task_generation_follower_term_dist_m=float(self._check("follower_term_dist_m", 0.3)),
                task_generation_require_astar_path=self._trajectory_planner_name(for_sampling=True) == "astar",
            )
            generated_episodes.append(episode)
            target_prim = self._episode_target_prim(episode)
            if target_prim:
                target_prim_counts[target_prim] += 1
            pbar.set_description(
                f"[TG] Distractor placenav {target['goal_name']} via {len(route['distractors'])} stores"
            )
            pbar.update(1)
        pbar.close()
        return generated_episodes

    def sample_mixed_innout_episodes(self):
        self.check_navmesh(self.cfg.scene["scene_id"])
        episode_mix = self._flow("episode_mix", {})
        remaining = {
            task_type: max(
                0,
                int(episode_mix.get(task_type, 0))
                - sum(ep.get("task_type") == task_type for ep in self.generated_episodes),
            )
            for task_type in ("placenav", "objnav")
        }
        task_type = "placenav" if remaining["placenav"] > 0 else "objnav"
        if remaining[task_type] <= 0:
            return []
        goal_dict = self.goal_dict_by_task.get(task_type, {})
        if not goal_dict:
            raise RuntimeError(f"No {task_type} goals found for innout scene")
        print(
            f"[TG] InNOut mixed episode remaining: placenav={remaining['placenav']}, "
            f"objnav={remaining['objnav']}; sampling {task_type}"
        )
        return self.sample_episodes_innout(
            1,
            task_type=task_type,
            goal_dict=goal_dict,
            check_navmesh=False,
        )

    def _innout_connector_prim(self):
        stage = self.manager_env.scene.stage
        connector_prim = stage.GetPrimAtPath("/World/ground/InnOutDoorConnector")
        if not connector_prim.IsValid():
            connector_prim = stage.GetPrimAtPath("/World/InnOutDoorConnector")
        if not connector_prim.IsValid():
            connector_prim = next(
                (prim for prim in stage.Traverse() if prim.GetName() == "InnOutDoorConnector"),
                connector_prim,
            )
        if not connector_prim.IsValid():
            raise RuntimeError("InNOut door connector prim is missing")
        return connector_prim

    def _stitch_innout_door_path(
        self,
        start_pos,
        reference_path,
        navmesh_interface,
        plan,
    ):
        path = np.asarray(reference_path, dtype=float)
        if len(path) < 2 or not self._sample("innout_centered_door_path", True):
            return path

        door_xy = np.asarray(plan["door_center_xy"], dtype=float)
        outward = np.asarray(plan["_outward_normal_xy"], dtype=float)
        inside_offset = float(self._sample("innout_door_waypoint_offset_m", 1.0))
        threshold_outer = float(
            (plan.get("threshold_summary") or {}).get("outer_distance_m", inside_offset)
        )
        outer_clearance = float(self._sample("innout_door_outer_waypoint_clearance_m", 0.25))
        outside_offset = max(inside_offset, threshold_outer + outer_clearance)
        tolerance = float(self._sample("innout_door_waypoint_tolerance_m", 0.25))
        if inside_offset <= 0.0 or outer_clearance < 0.0 or tolerance <= 0.0:
            raise ValueError("InNOut door waypoint offsets and tolerance must be valid")

        waypoint_z = float(np.asarray(start_pos, dtype=float)[2])
        inside = np.asarray([*(door_xy - outward * inside_offset), waypoint_z], dtype=float)
        doorway = np.asarray([*door_xy, waypoint_z], dtype=float)
        outside = np.asarray([*(door_xy + outward * outside_offset), waypoint_z], dtype=float)
        waypoints = [np.asarray(start_pos, dtype=float), inside, doorway, outside, path[-1]]
        segments = []
        for segment_index, (segment_start, segment_end) in enumerate(
            zip(waypoints[:-1], waypoints[1:], strict=True)
        ):
            segment = np.asarray(
                navmesh_interface.find_paths(segment_start, segment_end),
                dtype=float,
            )
            if len(segment) < 2:
                return None
            if (
                np.linalg.norm(segment[0, :2] - segment_start[:2]) > tolerance
                or np.linalg.norm(segment[-1, :2] - segment_end[:2]) > tolerance
            ):
                return None
            if segment_index in {1, 2}:
                segment = np.asarray([segment_start, segment_end], dtype=float)
            segments.append(segment)

        stitched = np.concatenate(
            [segment if index == 0 else segment[1:] for index, segment in enumerate(segments)],
            axis=0,
        )
        if len(stitched) < 2 or not np.isfinite(stitched).all():
            return None
        keep = np.concatenate(
            ([True], np.linalg.norm(np.diff(stitched, axis=0), axis=1) > 1e-6)
        )
        return stitched[keep]

    def _center_innout_goal_paths(
        self,
        goals,
        start_pos,
        navmesh_interface,
        plan,
    ):
        centered_goals = []
        for goal in goals:
            centered_path = self._stitch_innout_door_path(
                start_pos,
                goal["reference_path"],
                navmesh_interface,
                plan,
            )
            if centered_path is None:
                continue
            path_length = self._reference_path_length(centered_path)
            if path_length < self.min_path_length or path_length > self.max_path_length:
                continue
            goal["reference_path"] = centered_path.tolist()
            goal["path_length"] = path_length
            goal["innout_door_path_method"] = "center_waypoints"
            centered_goals.append(goal)
        return centered_goals

    def _build_innout_target_goals(
        self,
        start_pos,
        goal_prim,
        task_type,
        navmesh_interface,
        plan,
        connector_lateral_margin,
        connector_side_margin,
        connector_min_forward_dot,
        connector_max_lateral_offset,
    ):
        goal_positions = (
            self._innout_storefront_goal_positions(start_pos, goal_prim)
            if task_type == "placenav"
            else [None]
        )
        goals = []
        for goal_position in goal_positions:
            candidate_goals = self._build_goals_for_prim(
                start_pos,
                goal_prim,
                task_type,
                navmesh_interface,
                goal_position=goal_position,
            )
            if task_type == "placenav":
                for goal in candidate_goals:
                    goal["innout_storefront_goal_method"] = "facade_outward"
            goals.extend(candidate_goals)
        if not goals:
            return [], "no_reference_path", False

        anchor = self._goal_prim_anchor(goal_prim).tolist()
        for goal in goals:
            goal["innout_target_anchor"] = anchor
        goals = self._center_innout_goal_paths(
            goals,
            start_pos,
            navmesh_interface,
            plan,
        )
        if not goals:
            return [], "centered_door_path_failed", False
        goals = [
            goal
            for goal in goals
            if self._path_crosses_innout_door(
                goal["reference_path"],
                plan,
                lateral_margin_m=connector_lateral_margin,
                side_margin_m=connector_side_margin,
                min_forward_dot=connector_min_forward_dot,
                max_lateral_offset_m=connector_max_lateral_offset,
            )
        ]
        if not goals:
            return [], "path_misses_door_connector", False

        visibility_candidates = goals
        goals = self._filter_visible_goals(visibility_candidates, navmesh_interface)
        if goals:
            return goals, None, False
        visibility_reasons = {
            goal.get("visibility_reject_reason")
            for goal in visibility_candidates
            if goal.get("visibility_reject_reason")
        }
        missing_instance = visibility_reasons == {"target_instance_not_in_segmentation"}
        reason = "target_instance_not_in_segmentation" if missing_instance else "goal_visibility_failed"
        return [], reason, missing_instance

    def _sample_innout_target(
        self,
        goal_prim,
        task_type,
        navmesh_interface,
        plan,
        start_center,
        indoor_sample_radius,
        start_constraints,
        start_min_wall_distance,
        connector_lateral_margin,
        connector_side_margin,
        connector_min_forward_dot,
        connector_max_lateral_offset,
    ):
        failure_reason = "no_valid_indoor_start"
        target_start_attempts = max(
            1,
            int(self._sample("innout_target_start_attempts", 20)),
        )
        for _ in range(target_start_attempts):
            start_candidates = self.sample_valid_starts_near_point(
                navmesh_interface,
                start_center,
                radius_m=indoor_sample_radius,
                num_points=1,
                constraints=start_constraints,
                start_min_wall_distance=start_min_wall_distance,
            )
            if not start_candidates:
                continue
            start_pos = start_candidates[0]
            goals, failure_reason, terminal_failure = self._build_innout_target_goals(
                start_pos,
                goal_prim,
                task_type,
                navmesh_interface,
                plan,
                connector_lateral_margin,
                connector_side_margin,
                connector_min_forward_dot,
                connector_max_lateral_offset,
            )
            if not goals:
                if terminal_failure:
                    break
                continue
            closest_goal_idx = int(np.argmin([goal["path_length"] for goal in goals]))
            closest_goal = goals[closest_goal_idx]
            yaw_angle = calc_yaw(start_pos[:2], closest_goal["reference_path"][1][:2])
            yaw_angle += np.random.uniform(-np.deg2rad(30.0), np.deg2rad(30.0))
            settles, settle_info = self._task_generation_start_settles(
                start_pos,
                closest_goal["reference_path"],
                yaw_angle,
            )
            if settles:
                return {
                    "start_pos": start_pos,
                    "goals": goals,
                    "closest_goal_idx": closest_goal_idx,
                    "yaw_angle": yaw_angle,
                    "settle_info": settle_info,
                }, None
            failure_reason = f"start_settle:{settle_info.get('termination_reason', 'failed')}"
        return None, failure_reason

    def sample_episodes_innout(self, num_episodes, task_type="objnav", goal_dict=None, check_navmesh=True):
        if check_navmesh:
            self.check_navmesh(self.cfg.scene["scene_id"])
        navmesh_interface = self.navmesh_interface
        indoor_prim = self.manager_env.scene.stage.GetPrimAtPath(
            "/World/ground/terrain/start_result_navigation"
        )
        if not indoor_prim.IsValid():
            raise RuntimeError("InNOut indoor navigation prim is missing")
        indoor_bbox = np.asarray(self.env.get_prim_bounding_box(indoor_prim.GetPrimPath()), dtype=float)
        plan = self._innout_plan()
        self._innout_connector_prim()
        connector_lateral_margin = float(self._sample("innout_connector_lateral_margin_m", 0.1))
        connector_side_margin = float(self._sample("innout_connector_side_margin_m", 0.1))
        connector_min_forward_dot = float(
            self._sample("innout_connector_min_forward_dot", 0.0)
        )
        connector_max_lateral_offset = self._sample(
            "innout_connector_max_lateral_offset_m"
        )
        if connector_max_lateral_offset is not None:
            connector_max_lateral_offset = float(connector_max_lateral_offset)
        start_floor_height_range = float(self._sample("innout_start_floor_height_range_m", 0.6))
        indoor_center_xy = self._innout_room_center_xy(plan)
        local_room_size = (
            np.asarray(plan["interior_scene_bbox_max"], dtype=float)[:2]
            - np.asarray(plan["interior_scene_bbox_min"], dtype=float)[:2]
        ) * float(plan["room_effective_scale"])
        indoor_sample_radius = float(np.linalg.norm(local_room_size) * 0.5 + 0.5)
        source_goal_dict = goal_dict if goal_dict is not None else self.goal_dict
        filtered_goal_dict = self._filter_innout_goal_dict(source_goal_dict, plan)
        if not filtered_goal_dict:
            raise RuntimeError(f"No exterior {task_type} goals found near the indoor space")

        generated_episodes = []
        target_prim_counts = Counter(self.target_prim_counts)
        start_min_wall_distance = float(self._sample("start_min_wall_distance", 0.0))
        start_center = [
            indoor_center_xy[0],
            indoor_center_xy[1],
            (indoor_bbox[2] + indoor_bbox[5]) * 0.5,
        ]
        start_constraints = {
            "indoor_bbox": indoor_bbox,
            "innout_plan": plan,
            "innout_room_margin_m": float(self._sample("innout_start_room_margin_m", 0.15)),
            "innout_door_interior_margin_m": float(
                self._sample("innout_start_door_interior_margin_m", 0.25)
            ),
            "exclude_building_interiors": False,
            "z_range": (
                float(indoor_bbox[2]) - 0.1,
                float(indoor_bbox[2]) + start_floor_height_range,
            ),
            "start_min_wall_distance": start_min_wall_distance,
            "near_point_sample_attempts": 12,
        }
        sample_attempts = 0
        max_sample_attempts = max(1, int(self._sample("innout_episode_sample_attempts", 20)))
        pbar = tqdm(total=num_episodes, desc=f"Generating innout {task_type} episodes")
        while len(generated_episodes) < num_episodes and not self.generate_finished:
            if sample_attempts >= max_sample_attempts:
                self._last_sample_failure_reason = f"innout_{task_type}_sampling_exhausted"
                break
            sample_attempts += 1
            selected_target = self._choose_innout_target(
                filtered_goal_dict,
                target_prim_counts,
            )
            if selected_target is None:
                self._last_sample_failure_reason = f"no_exterior_{task_type}_targets_available"
                break
            goal_name, goal_prim, prim_path = selected_target
            sampled, sampling_failure_reason = self._sample_innout_target(
                goal_prim,
                task_type,
                navmesh_interface,
                plan,
                start_center,
                indoor_sample_radius,
                start_constraints,
                start_min_wall_distance,
                connector_lateral_margin,
                connector_side_margin,
                connector_min_forward_dot,
                connector_max_lateral_offset,
            )
            if sampled is None:
                self._record_innout_target_failure(prim_path, sampling_failure_reason)
                pbar.set_description(
                    f"[TG] Failed innout {task_type} goal {goal_name}: "
                    f"{sampling_failure_reason}"
                )
                continue

            start_pos = sampled["start_pos"]
            goals = sampled["goals"]
            closest_goal_idx = sampled["closest_goal_idx"]
            closest_goal = goals[closest_goal_idx]
            yaw_angle = sampled["yaw_angle"]
            settle_info = sampled["settle_info"]

            quat_xyzw = R.from_euler("z", yaw_angle).as_quat().tolist()
            episode_kwargs = {
                "objnav": goal_name,
                "instruction": "",
                "episode_id": len(generated_episodes),
                "goals": goals,
                "start_position": start_pos.tolist(),
                "start_rotation": [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                "closest_goal_idx": closest_goal_idx,
                "task_type": task_type,
                "start_settle": settle_info,
                "innout_start_region": "indoor",
                "innout_goal_region": "outdoor",
                "innout_door_center_xy": list(plan["door_center_xy"]),
                "innout_door_outward_normal_xy": plan["_outward_normal_xy"].tolist(),
                "innout_start_door_signed_distance_m": self._innout_door_signed_distance(
                    start_pos,
                    plan,
                ),
                "innout_connector_validation": "oriented_door_plane",
                "task_generation_strict_endpoint": task_type == "placenav",
                "task_generation_path_endpoint_tolerance_m": self._goal_endpoint_tolerance(
                    closest_goal["radius"],
                    task_type,
                ),
                "task_generation_follower_term_dist_m": float(
                    self._check("follower_term_dist_m", 0.3)
                ),
                "task_generation_require_astar_path": self._trajectory_planner_name(for_sampling=True) == "astar",
            }
            if task_type == "placenav":
                episode_kwargs.update({
                    "placenav_goal": goal_name,
                    "placenav_category": "store",
                    "instruction": goal_name.replace("_", " "),
                })
            episode = VLNEpisode(data=self._episode_base_data(), **episode_kwargs)
            generated_episodes.append(episode)
            sample_attempts = 0
            target_prim = self._episode_target_prim(episode)
            if target_prim:
                target_prim_counts[target_prim] += 1
            pbar.update(1)
        pbar.close()
        return generated_episodes
    
    def get_world_bb(self, scene_id):
        scene_config = self.episode_data(scene_id)
        navmesh_exclude = scene_config.get("navmesh_exclude", [])
        valid_exclude_prims = []
        for prim_path in navmesh_exclude:
            exclude_prim = self.manager_env.scene.stage.GetPrimAtPath(prim_path)
            if not exclude_prim.IsValid():
                print(f"[TG] Skip missing navmesh_exclude prim while computing bbox: {prim_path}")
                continue
            valid_exclude_prims.append(exclude_prim)
            prim_utils.set_prim_visibility(exclude_prim, False)
        world_bb = self.env.get_prim_bounding_box("/World/ground/terrain")
        for exclude_prim in valid_exclude_prims:
            prim_utils.set_prim_visibility(exclude_prim, True)
        return world_bb

    def toggle_ceiling(self, scene_id):
        if not hasattr(self, 'ceiling_data') or self.ceiling_data["scene_id"] != scene_id:
            print("[TG] Detecting ceiling")
            prim_list = []
            world_bb = self.get_world_bb(scene_id)
            for prim in self.manager_env.scene.stage.Traverse():
                if str(prim.GetPath()).startswith("/World/ground/terrain"):
                    prim_bb = self.env.get_prim_bounding_box(prim.GetPrimPath())
                    if prim_bb[2] > world_bb[5]-1.0 and prim_bb[2] < 1e3:
                        prim_list.append(prim)
            self.ceiling_data = {
                "prim_list": prim_list,
                "prim_hidden": False,
                "scene_id": scene_id,
            }
        self.ceiling_data["prim_hidden"] = not self.ceiling_data["prim_hidden"]
        for prim in self.ceiling_data["prim_list"]:
            prim_utils.set_prim_visibility(prim, not self.ceiling_data["prim_hidden"])
        print(f"[TG] Ceiling is now {'visible' if self.ceiling_data['prim_hidden'] else 'hidden'}")
    
    def create_bev_camera(self, scene_id):
        # create bev camera
        world_bb = np.array(self.get_world_bb(scene_id))
        scene_type, scene_name = self.parse_scene_id(scene_id)
        world_center = (world_bb[:3] + world_bb[3:]) / 2
        world_size = world_bb[3:] - world_bb[:3]
        if world_size[0] > 100.0 or world_size[1] > 100.0:
            print("[TG] World size is too large, checking large prim now")
            for prim in self.manager_env.scene.stage.Traverse():
                prim_bb = self.env.get_prim_bounding_box(prim.GetPrimPath())
                if prim_bb[3] - prim_bb[0] > 100.0 or prim_bb[4] - prim_bb[1] > 100.0:
                    print(f"[TG] Found large prim: {prim.GetPrimPath()}")
                    if "model_" in str(prim.GetPrimPath()) and scene_type != "vc":
                        prim_utils.set_prim_visibility(prim, False)
            world_bb = np.array(self.get_world_bb(scene_id))
            world_center = (world_bb[:3] + world_bb[3:]) / 2
            world_size = world_bb[3:] - world_bb[:3]
        horizontal_span = max(float(world_size[0]), float(world_size[1]))
        camera_height_margin = max(10.0, min(50.0, horizontal_span * 0.02))
        camera_pos = world_center.copy()
        camera_pos[2] = world_bb[5] + camera_height_margin
        padding = 1.0
        image_width = min(int((world_size[0]+padding*2) * 100), 3840)
        image_height = int(image_width * world_size[1] / world_size[0])
        self.bev_data = {
            "scene_id": scene_id,
            "px_per_meter": image_width / (world_size[0]+padding*2),
            "world_center": world_center.tolist(),
            "world_size": world_size.tolist(),
            "world_bb": world_bb.tolist(),
            "camera_position": camera_pos.tolist(),
            "camera_height_margin": camera_height_margin,
            "image_width": image_width,
            "image_height": image_height,
        }
        if hasattr(self, 'bev_camera'):
            try:
                prim_utils.delete_prim(self.bev_camera.cfg.prim_path)
            except Exception as e:
                print(f"[TG] Failed to delete previous BEV camera: {e}")
        if not hasattr(self, 'bev_camera_count'):
            self.bev_camera_count = 0
        self.bev_camera_count += 1
        self.bev_camera = self.env.create_camera(
            prim_path=f"/World/bev_camera_{self.bev_camera_count}",
            perspective=False,
            pos=camera_pos,
            quat_opengl=[1, 0, 0, 0],
            horizontal_aperture=(world_size[0]+padding*2)*10,
            clipping_range_min=0.1,
            clipping_range_max=max(200.0, float(camera_pos[2] - world_bb[2] + camera_height_margin)),
            width=image_width,
            height=image_height
        )
        print(
            "[TG] Created BEV map camera "
            f"at z={camera_pos[2]:.2f}, scene_z=[{world_bb[2]:.2f}, {world_bb[5]:.2f}]"
        )

    def create_bev_map(self, scene_id, file_name="bev_map", clip_range="ceiling", ceiling_height=None):
        """ When clip_range is "ceiling", the bev image sees everything under the ceiling.
            When clip_range is "robot", the bev image only see things under robot height + 0.5m,
            which can be used for occupancy.
        """
        # init task queue and lock if not initialized
        if not hasattr(self, 'check_bev_map_queue'):
            self.bev_camera_queue = Queue()
            self.bev_camera_lock = Lock()
            def check_bev_map_queue():
                if self.bev_camera_queue.empty() or self.bev_camera_lock.locked():
                    return
                scene_id, file_name, clip_range, ceiling_height = self.bev_camera_queue.get()
                self._create_bev_map(scene_id, file_name, clip_range, ceiling_height)
            self.vln_sim.add_callback('step_finished', check_bev_map_queue)
            self.check_bev_map_queue = check_bev_map_queue
        if (not hasattr(self, 'bev_camera')) or self.bev_data.get("scene_id") != scene_id:
            self.create_bev_camera(scene_id)
        # add task to queue
        self.bev_camera_queue.put((scene_id, file_name, clip_range, ceiling_height))

    def _create_bev_map(self, scene_id, file_name, clip_range, ceiling_height):
        self.bev_camera_lock.acquire()
        # update camera
        if ceiling_height is None:
            scene_config = self.episode_data(scene_id)
            ceiling_height = scene_config.get("ceiling_height", 1.0)
        self.update_bev_camera_clip(scene_id, clip_range, ceiling_height)
        self.bev_camera.reset()
        # hide robot
        robot_prim = self.manager_env.scene.stage.GetPrimAtPath("/World/envs/env_0/Robot")
        prim_utils.set_prim_visibility(robot_prim, False)
        def save_bev_map():
            try:
                try:
                    sim = self.manager_env.sim
                    sim.forward()
                    sim.render()
                    if hasattr(self.bev_camera, "_is_outdated"):
                        self.bev_camera._is_outdated[:] = True
                    self.bev_camera.update(0.0, force_recompute=True)
                except Exception as e:
                    print(f"[TG] Failed to refresh BEV camera before save: {e}")
                if self.bev_camera.frame>-1 and "rgba" in self.bev_camera.data.output and "distance_to_image_plane" in self.bev_camera.data.output:
                    rgb = self.bev_camera.data.output['rgba'].cpu().numpy()[0]
                    depth = self.bev_camera.data.output['distance_to_image_plane'].cpu().numpy()[0]
                    data_folder = self.episode_data_scene_folder(scene_id)
                    data_folder.mkdir(parents=True, exist_ok=True)
                    np.savez(data_folder / f"{file_name}.npz", rgb=rgb, depth=depth)
                    cv2.imwrite(str(data_folder / f"{file_name}_rgb.png"), rgb[...,[2,1,0,3]])
                    cv2.imwrite(str(data_folder / f"{file_name}_depth.png"), depth)
                    with (data_folder / f"{file_name}_info.txt").open('w') as f:
                        json.dump(self.bev_data, f)
                    print(f"[TG] Saved BEV map to {data_folder}/{file_name}")
                    prim_utils.set_prim_visibility(robot_prim, True)
                    self.vln_sim.remove_callback('step_finished', save_bev_map)
                    self.bev_camera_lock.release()
                    # prim_utils.delete_prim(self.bev_camera.cfg.prim_path)
                else:
                    print(
                        f"[TG] BEV camera is not initialized yet, frame: {self.bev_camera.frame}, "
                        f"env_step: {self.env.env_step}, outputs: {list(self.bev_camera.data.output.keys())}"
                    )
            except Exception as e:
                print(f"[TG] Error saving BEV map: {e}")
                raise e
        self.vln_sim.add_callback('step_finished', save_bev_map)

    def update_bev_camera_clip(self, scene_id, clip_range, ceiling_height):
        print(f"[TG] Clip bev camera to {clip_range} with ceiling height {ceiling_height}")
        world_bb = np.array(self.get_world_bb(scene_id))
        camera_pos = np.asarray(self.bev_data["camera_position"], dtype=float)
        if clip_range=="ceiling":
            clipping_range_min = camera_pos[2] - (world_bb[5]-ceiling_height)
        elif clip_range=="robot":
            clipping_range_min = camera_pos[2] - (self.env.get_cam_pose()[0][2]+0.3)
        else:
            raise ValueError(f"Unsupported BEV clip range: {clip_range}")
        clipping_range_min = max(0.1, float(clipping_range_min))
        clipping_range_max = max(
            clipping_range_min + 1.0,
            float(camera_pos[2] - world_bb[2] + 10.0),
        )
        camera_prim = self.manager_env.scene.stage.GetPrimAtPath(f"/World/bev_camera_{self.bev_camera_count}")
        clipping_range = camera_prim.GetAttribute('clippingRange').Get()
        clipping_range[0] = clipping_range_min
        clipping_range[1] = clipping_range_max
        camera_prim.GetAttribute('clippingRange').Set(clipping_range)
        # update task config
        _, scene_name = self.parse_scene_id(scene_id)
        self.task_config.setdefault("scene", {}).setdefault("scenes", {}).setdefault(scene_name, {})["ceiling_height"] = ceiling_height

    def save_status(self, scene_id, **kwargs):
        status_file = self._status_file(scene_id)
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_data = {}
        if status_file.exists():
            with status_file.open('r') as f:
                try:
                    status_data = json.load(f)
                except json.JSONDecodeError:
                    pass

        status_data.update(kwargs)
        
        with status_file.open('w') as f:
            json.dump(status_data, f)

    def timing_status(self, scene_id, status_name, **kwargs):
        @contextlib.contextmanager
        def context():
            self.save_status(scene_id, status=status_name)
            start_time = time.time()
            try:
                yield
                duration = time.time() - start_time
                kwargs[f"{status_name}"] = "success"
                kwargs[f"{status_name}_time"] = duration
                self.save_status(scene_id, **kwargs)
            except Exception as e:
                duration = time.time() - start_time
                kwargs[f"{status_name}"] = "error"
                kwargs[f"{status_name}_time"] = duration
                kwargs[f"{status_name}_error"] = str(e)
                self.save_status(scene_id, **kwargs)
                raise e
        return context()

    def load_status(self, scene_id):
        status_file = self._status_file(scene_id)
        if not status_file.exists():
            return None
        with status_file.open('r') as f:
            status_data = json.load(f)
            return status_data.get("status", None)
