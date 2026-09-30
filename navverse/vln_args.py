import json
import sys
from pathlib import Path

from dotenv import dotenv_values
from omegaconf import DictConfig, ListConfig, OmegaConf

from navverse.config import load_config, resolve_task_config_path
from navverse.episode import episode_json_paths


DEFAULT_KIT_ARGS = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=fatal --/rtx-transient/resourcemanager/enableTextureStreaming=false --/plugins/carb.tasking.plugin/threadCount=16 --/plugins/omni.tbb.globalcontrol/maxThreadCount=16"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _strip_config_overrides(arg_list):
    parser_args = []
    config_overrides = []
    for item in arg_list:
        if not item.startswith("--") and "=" in item:
            config_overrides.append(item)
        else:
            parser_args.append(item)
    return parser_args, config_overrides


def _explicit_arg_names(arg_list):
    names = set()
    for item in arg_list:
        if not item.startswith("--") or item.startswith("--/"):
            continue
        option = item[2:].split("=", 1)[0]
        names.add(option.replace("-", "_"))
    return names


def _namespace_value(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return value
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _merge_config_into_args(args, config, explicit_args):
    for key, value in config.items():
        if key in explicit_args:
            continue
        setattr(args, key, _namespace_value(value))
    return args


def _resolve_config_path(config_path=None):
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    return config_path.resolve()


def _episode_config_path(episode_folder, episode_label):
    if not episode_label or episode_label == "none":
        return None
    try:
        scene_id, episode_id = episode_label.rsplit("_", 1)
        episode_id = int(episode_id)
    except (TypeError, ValueError):
        return None

    try:
        paths = episode_json_paths(episode_folder)
    except OSError:
        return None
    paths.sort(key=lambda path: Path(path).stem != scene_id)
    for path in paths:
        try:
            episodes = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(episodes, list):
            continue
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            if episode.get("scene_id") != scene_id or str(episode.get("episode_id")) != str(episode_id):
                continue
            return episode.get("config")
    return None


def load_runtime_config(config_path=None, dotlist_overrides=None):
    config_path = _resolve_config_path(config_path)
    cfg = OmegaConf.create(load_config(config_path))
    OmegaConf.set_struct(cfg, False)
    if dotlist_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist_overrides))
    cfg = OmegaConf.merge(cfg, {"config": str(config_path)})
    return cfg


def add_vln_args(parser):
    arg_group = parser.add_argument_group("VLN benchmark", description="Arguments for VLN benchmark.")
    arg_group.add_argument(
        "--episode_folder",
        type=str,
        default="navverse_data/episodes",
        help="Path to the episode data installed from the NavVerse dataset.",
    )
    arg_group.add_argument("--episode_label", type=str, default="none", help="Test specific episode label. Format: {scene_id}_{episode_id}), e.g., nv_apartment_0.")
    arg_group.add_argument("--test_scene_id", type=str, default="none", help="Scene id used for task generation.")
    arg_group.add_argument("--scene_folder", type=str, default=None, help="Path to the scene USD file.")
    arg_group.add_argument("--navmesh_path", type=str, default=None, help="Path to the navmesh file.")
    arg_group.add_argument("--trajectory_planner", type=str, default=None, choices=["navmesh", "astar"], help="Override task-generation reference path planner. Defaults to each scene config or navmesh.")
    arg_group.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Hydra runtime config entrypoint.",
    )
    arg_group.add_argument("--disable_camera", type=str2bool, default=False, help="Disable camera sensors entirely.")
    arg_group.add_argument("--on_demand_render", type=str2bool, default=True, help="Render socket camera images only when sensor data is requested.")
    arg_group.add_argument("--enable_locomotion_reward", action="store_true", help="Compute low-level locomotion reward terms during evaluation/serving.")
    arg_group.add_argument("--enable_contact_sensor", type=str2bool, default=False, help="Keep contact force sensors enabled without locomotion rewards.")
    arg_group.add_argument("--rebuild_dynamic_navmesh", type=str2bool, default=False, help="Rebuild a navmesh from the loaded stage for generator scenes instead of loading a prebuilt navmesh file.")
    arg_group.add_argument("--profile_step_timing", type=str2bool, default=False, help="Print coarse simulator step/render timing breakdowns.")
    arg_group.add_argument("--profile_step_interval", type=int, default=100, help="Number of sim steps between profile timing summaries.")
    arg_group.add_argument("--task_type", type=str, default="objnav", help="Type of task to simulate.", choices=["objnav", "vln", "locnav", "placenav"])
    arg_group.add_argument("--instruction_type", type=str, default="target", choices=["target", "vln"], help="Instruction to send for each episode.")
    arg_group.add_argument("--num_envs", type=int, default=1, help="How many robots to simulate.")
    arg_group.add_argument("--robot_name", type=str, default="spot", help="Robot to simulate.", choices=["spot", "turtlebot", "tidybot", "cylinder", "drone"])
    arg_group.add_argument(
        "--tidybot_embodiment",
        type=str,
        default="full",
        choices=["full", "habitat_pe"],
        help="TidyBot geometry/control profile; ignored for other robots.",
    )
    arg_group.add_argument(
        "--timed_trajectory_control_dt",
        type=float,
        default=None,
        help="Set the simulator control period used by TIMED_TRAJECTORY commands.",
    )
    from navverse_tools.terrain_eval.arguments import add_terrain_eval_args

    add_terrain_eval_args(parser)
    arg_group = parser.add_argument_group("Debug Options")
    arg_group.add_argument("--disable_termination", type=str2bool, default=False, help="Disable episode termination.")
    arg_group = parser.add_argument_group("Task Generation")
    arg_group.add_argument("--tg_config_path", type=str, default=None, help="Path to the unified scene-type config file.")
    arg_group.add_argument("--tg_mode", type=str, default="manual", help="Mode for task generation.", choices=["manual", "auto"])
    arg_group.add_argument("--replay_episode_data", type=str2bool, default=False, help="Replay existing episodes to replace recorded episode data without changing episode JSON.")
    arg_group.add_argument("--replay_episode_labels_file", type=str, default=None, help="Optional one-label-per-line allowlist for episode replay.")
    arg_group.add_argument("--bev_only", action="store_true", help="Generate BEV maps without generating or replaying episodes.")
    arg_group.add_argument("--force_regenerate_bev", action="store_true", help="Regenerate BEV maps even when complete outputs already exist.")
    arg_group = parser.add_argument_group("Server", description="Arguments for socket server.")
    arg_group.add_argument("--disable_socket_server", type=str2bool, default=False, help="Disable socket server.")
    arg_group.add_argument("--host", type=str, default="localhost", help="Host for socket server.")
    arg_group.add_argument("--port", type=int, default=12300, help="Port for socket server.")
    arg_group.add_argument("--foxglove_port", type=int, default=0, help="Port for foxglove server.")

def parse_args(parser):
    raw_arg_list = sys.argv[1:]
    arg_list, config_overrides = _strip_config_overrides(raw_arg_list)
    explicit_args = _explicit_arg_names(arg_list)
    args = parser.parse_args(arg_list)
    dot_args = dotenv_values(".env")
    for key, value in dot_args.items():
        key_name = key.lower()
        if key_name in args.__dict__ and key_name not in explicit_args:
            arg_list += [f"--{key_name}", value]
            explicit_args.add(key_name)
            print(f"Using {key} from .env: {dot_args[key]}")
    args = parser.parse_args(arg_list)
    user_kit_args = getattr(args, "kit_args", None)
    if "kit_args" in args.__dict__:
        merged_kit_args = DEFAULT_KIT_ARGS if not user_kit_args else f"{DEFAULT_KIT_ARGS} {user_kit_args}"
        if "--kit_args" not in arg_list:
            arg_list += ["--kit_args", merged_kit_args]
        print(f"Using kit args: {merged_kit_args}")
        args = parser.parse_args(arg_list)
        args.kit_args = merged_kit_args
    config_path = args.config
    if "config" not in explicit_args:
        episode_config = _episode_config_path(args.episode_folder, args.episode_label)
        if episode_config:
            config_path = episode_config
            print(f"Using config from episode {args.episode_label}: {episode_config}")
        elif args.test_scene_id not in (None, "", "none"):
            config_path = str(resolve_task_config_path(None, scene_id=args.test_scene_id))
            print(f"Using config inferred from scene {args.test_scene_id}: {config_path}")
        elif args.tg_config_path:
            config_path = args.tg_config_path
            print(f"Using task-generator config as runtime config: {config_path}")
    runtime_config = load_runtime_config(
        config_path=config_path,
        dotlist_overrides=config_overrides,
    )
    args = _merge_config_into_args(args, runtime_config, explicit_args)
    args._explicit_args = explicit_args
    args._config_dotlist_overrides = config_overrides
    args.runtime_config = runtime_config
    args.enable_cameras = True
    return args
