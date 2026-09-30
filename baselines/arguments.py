import argparse

from navverse.launcher.registry import agent_names
from navverse.termination_cfg import DEFAULT_EPISODE_TIMEOUT_S


def comma_separated_ints(s):
    return [int(x) for x in s.split(",") if x]

def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Visual-Language-Navigation')
    # General Arguments
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # Module Arguments
    parser.add_argument('--agent', type=str, default="Dummy", help='agent class name',
        choices=agent_names())
    parser.add_argument('--action_type', type=str, default="single", help='action type',
        choices=["single", "multi"])
    parser.add_argument('--continuous', action="store_true",
                        help="if set, the agent will not reset after stop action")
    # Backend
    parser.add_argument('--backend', type=str, default="isaac", help='what backend to use for sending actions')
    parser.add_argument('--host', type=str, default="localhost", help='host name of the perception server')
    parser.add_argument('--port', type=int, default=12300)
    
    # Logging, loading models, visualization
    parser.add_argument('--log_file', type=str, default="agent.log",
                        help='path to log file')
    parser.add_argument('-d', '--dump_location', type=str, default="./dump",
                        help='path to dump models and log (default: ./tmp/)')
    parser.add_argument('--exp_name', type=str, default="isaac",
                        help='experiment name')
    parser.add_argument('--foxglove', type=bool, default=True,
                        help="Start Foxglove live visualization")
    parser.add_argument('--foxglove_host', type=str, default="0.0.0.0",
                        help="Foxglove host")
    parser.add_argument('--foxglove_port', type=int, default=0,
                        help="The default port will be backend port + 1000")
    parser.add_argument('--save_rgbd', action="store_true",
                        help='save rgbd images')
    parser.add_argument('--save_map', action="store_true",
                        help='save map images')
    parser.add_argument('--save_obstacle_map', action="store_true",
                        help='save obstacle map image')
    parser.add_argument('--save_scene_graph', action="store_true", 
                        help='save scene graph')
    parser.add_argument(
        '--save_video',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='save visualization as video',
    )
    parser.add_argument(
        '--canonical_video_index',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='upload one manifest-sorted W&B video array after a complete run',
    )
    parser.add_argument(
        '--canonical_video_prefix',
        type=str,
        default='episode_video',
        help='W&B key prefix for manifest-sorted video arrays',
    )
    parser.add_argument(
        "--obs_action_bev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a BEV path panel to obs_action videos when map metadata is available",
    )
    parser.add_argument(
        "--modern_obs_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save an additional fixed-layout observation, instruction, and BEV video",
    )
    parser.add_argument('--save-parquet', dest='save_parquet', action="store_true",
                        help='save per-episode step, pose, image, action, and metric records as parquet')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('-n', '--num_processes', type=int, default=1, 
                        help='number of processes to use for each gpu')
    parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use, 0 for all available GPUs')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_assign', type=comma_separated_ints, default=[1], help='number of processes to assign to each GPU')
    parser.add_argument('--episode_labels', type=str, default="all", help='use specific episode label or key defined in constants')
    # gui
    parser.add_argument('--path_npz', type=str, default="./saved_pcd/",
                        help='path to saved pcd (default: ./saved_pcd/)')

    
    # Environment, dataset and episode specifications
    parser.add_argument("--task_config", type=str,
                        default='objectnav_hm3d_rgbd_semantic.yaml',
                        help="path to config yaml containing task information")
    parser.add_argument("--instruction_type", type=str, default="target",
                        choices=["target", "vln"],
                        help="instruction type to select in each episode")
    parser.add_argument("--exp_config", type=str,
                        default="ObjectOnly.yaml",
                        help="path to config yaml containing scene graph information")
    parser.add_argument('--episode_count', type=int, default=-1)
    parser.add_argument('--split', type=str, default="val")

    # Model Hyperparameters
    parser.add_argument('--turn_angle', type=int, default=30)
    
    # Mapping
    parser.add_argument('--hfov', type=float, default=47.1686, # 79.0
                        help="horizontal field of view in degrees")
    parser.add_argument('--primary_camera', type=str, default="",
                        help="camera name to use for policy/perception (default: first available)")
    parser.add_argument('--mapping_cameras', type=str, default="",
                        help="comma-separated camera names to use for mapping (default: all available)")
    parser.add_argument('--min_depth', type=float, default=1.0,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=100.0,
                        help="Maximum depth for depth sensor in meters")
    parser.add_argument('--map_resolution', type=float, default=0.05,
                        help="Resolution of the map in meter")
    parser.add_argument('--map_size', type=float, default=25,
                        help="Size of the map in meters")
    parser.add_argument('--collision_threshold', type=float, default=0.10)
    parser.add_argument('--temporal_collision', default=True, type=bool,
                        help="if set, the temporal collision will be used")
    parser.add_argument('--map_voxel_size', type=float, default=0.05,
                        help="Voxel size for the map in meters")

    # SAM setting
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.5)

    parser.add_argument("--sam_variant", type=str, default="sam",
                        choices=['fastsam', 'mobilesam', "lighthqsam", "sam"])
    parser.add_argument("--detector", type=str, default="dino", 
                        choices=["yolo", "dino", "none"], 
                        help="If none, no tagging and detection will be used and the SAM will be run in dense sampling mode. ")
    parser.add_argument("--add_bg_classes", action="store_true", 
                        help="If set, add background classes (wall, floor, ceiling) to the class set. ")
    parser.add_argument("--accumu_classes", action="store_true",
                        help="if set, the class set will be accumulated over frames")


    # LLM setting
    parser.add_argument('--vln_mode', type=str, default="vlm_game",
                        choices=['clip', 'vlm', 'vlm_rank', "vlm_game"])
    # parser.add_argument('--gpt_type', type=int, default=2,
    #                     help="""0: text-davinci-003
    #                             1: gpt-3.5-turbo
    #                             2: gpt-4o
    #                             3: gpt-4o-mini
    #                             (default: 2)""")
    parser.add_argument('--api', type=str, default="xx-xxxx")

    parser.add_argument('--scene_graph_prediction_llm', type=str, default="gpt-4o-mini")
    parser.add_argument('--group_caption_vlm', type=str, default="gpt-4o-mini") # gpt-4o-mini, llama3.2-vision
    parser.add_argument('--ollama_host', type=str, default="http://localhost", help='do not add port, it will be added automatically')
    parser.add_argument('--ollama_port_start', type=int, default="11888", help='every process will use a different port ollama_port_start+i')
    parser.add_argument('--vllm_base_url', type=str, default="http://localhost:8002/v1", help='openai compatible api base url for vllm')
    parser.add_argument('--vllm_model', type=str, default="google/gemma-3-4b-it", help='model name for vllm')

    parser.add_argument('--load', type=str, default="0",
                    help="""model path to load,
                            0 to not reload (default: 0)""")
    parser.add_argument('--longnav_server_url', type=str, default="",
                        help="HTTP endpoint for LongNav action inference. Leave empty to auto-spawn "
                             "one server per agent worker on base_port + worker_id. If set, may contain "
                             "{port}/{worker_id} placeholders for external server pools.")
    parser.add_argument('--longnav_base_port', type=int, default=8000,
                        help="Base port used by each LongNav agent worker; worker i listens on base_port + i")
    parser.add_argument('--longnav_checkpoint', type=str,
                        default="Aasdfip/hm3d_rpp_ke_standard-checkpoint_231",
                        help="Checkpoint dir or HuggingFace repo id for the auto-spawned LongNav server")
    parser.add_argument('--longnav_server_python', type=str, default="",
                        help="Path to python interpreter used to run serve_longnav.py. "
                             "Falls back to $LONGNAV_SERVER_PYTHON, then "
                             "baselines/longnav_r1/.venv/bin/python.")
    parser.add_argument('--longnav_attn_impl', type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="attention backend for the auto-spawned LongNav server")
    parser.add_argument('--longnav_action_map', type=str,
                        default="STOP,FORWARD,TURN_LEFT,TURN_RIGHT",
                        help="comma-separated NavVerse action names for model action ids 0..3")
    parser.add_argument('--longnav_request_timeout', type=float, default=None,
                        help="request timeout in seconds for the LongNav action server; defaults to episode timeout")
    parser.add_argument('--longnav_max_steps', type=int, default=200,
                        help="force STOP after this many LongNav steps; <=0 disables")
    parser.add_argument('--longnav_stop_prob_threshold', type=float, default=None,
                        help="only execute sampled STOP when model stop probability is at least this value")
    parser.add_argument('--internvla_n1_server_url', type=str, default="",
                        help="external InternVLA-N1 /act endpoint; empty auto-spawns one server per worker")
    parser.add_argument('--internvla_n1_base_port', type=int, default=8100)
    parser.add_argument('--internvla_n1_checkpoint', type=str,
                        default="InternRobotics/InternVLA-N1-DualVLN")
    parser.add_argument('--internvla_n1_server_python', type=str, default="")
    parser.add_argument('--internvla_n1_num_history', type=int, default=8)
    parser.add_argument('--internvla_n1_plan_step_gap', type=int, default=8)
    parser.add_argument('--internvla_n1_action_map', type=str,
                        default="STOP,FORWARD,TURN_LEFT_SMALL,TURN_RIGHT_SMALL")
    parser.add_argument('--internvla_n1_request_timeout', type=float, default=None)
    parser.add_argument('--internvla_n1_max_steps', type=int, default=500)
    parser.add_argument('--internvla_n1_depth_scale_m', type=float, default=10.0)
    parser.add_argument('--streamvln_server_url', type=str, default="",
                        help="external StreamVLN /act endpoint; empty auto-spawns one server per worker")
    parser.add_argument('--streamvln_base_port', type=int, default=8200)
    parser.add_argument('--streamvln_checkpoint', type=str,
                        default="mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3")
    parser.add_argument('--streamvln_tokenizer', type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument('--streamvln_server_python', type=str, default="")
    parser.add_argument('--streamvln_attn_impl', choices=["flash_attention_2", "sdpa", "eager"],
                        default="flash_attention_2")
    parser.add_argument('--streamvln_num_history', type=int, default=8)
    parser.add_argument('--streamvln_num_frames', type=int, default=32)
    parser.add_argument('--streamvln_num_future_steps', type=int, default=4)
    parser.add_argument('--streamvln_action_map', type=str,
                        default="STOP,FORWARD,TURN_LEFT_SMALL,TURN_RIGHT_SMALL")
    parser.add_argument('--streamvln_request_timeout', type=float, default=None)
    parser.add_argument('--streamvln_max_steps', type=int, default=500)
    parser.add_argument('--gpt_server_url', type=str, default="",
                        help="external GPT Codex /act endpoint; empty auto-spawns one server per worker")
    parser.add_argument('--gpt_base_port', type=int, default=8300)
    parser.add_argument('--gpt_model', type=str, default="gpt-6-astra")
    parser.add_argument('--gpt_reasoning_effort', choices=["low", "medium", "high", "xhigh", "max"],
                        default="low")
    parser.add_argument('--gpt_server_python', type=str, default="",
                        help="Python with openai-codex installed; falls back to GPT_SERVER_PYTHON or conda env gpt")
    parser.add_argument('--gpt_compact_at_fraction', type=float, default=0.70,
                        help="summarize and compact a GPT episode thread at this context-window fraction")
    parser.add_argument('--gpt_request_timeout', type=float, default=None)
    parser.add_argument('--gpt_max_steps', type=int, default=175)
    parser.add_argument('--gpt_stop_range_m', type=float, default=1.6)
    parser.add_argument('--gpt_action_map', type=str,
                        default="STOP,FORWARD,TURN_LEFT,TURN_RIGHT")
    parser.add_argument('--oracle_action_type', type=str, default="waypoint",
                        choices=["waypoint", "transform", "task_generation_internal"],
                        help="Oracle control output: simulator waypoints or absolute GT transforms")
    parser.add_argument(
        '--oracle_capture_internal_frames',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture camera frames during task-generation-internal diagnostics",
    )
    parser.add_argument(
        '--oracle_stale_waypoint_skip',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the Oracle waypoint controller to skip a target using wall-clock no-progress checks",
    )
    parser.add_argument('--oracle_discrete_angle_threshold_deg', type=float, default=30.0,
                        help="yaw error threshold for OracleDiscreteAction TURN_LEFT/TURN_RIGHT")
    parser.add_argument('--oracle_discrete_waypoint_switch_distance_m', type=float, default=1.0,
                        help="distance for OracleDiscreteAction to advance to the next intermediate waypoint")
    parser.add_argument('--oracle_discrete_stop_distance_m', type=float, default=0.5,
                        help="minimum final-waypoint stop distance for OracleDiscreteAction")
    parser.add_argument('--dummy_action_sequence', type=str, default="FORWARD,TURN_LEFT,TURN_RIGHT",
                        help="comma-separated action sequence for the Dummy action-test agent")
    parser.add_argument('--dummy_trajectory_suite', type=str, default="",
                        help="comma-separated Dummy trajectory tests to run; supported: circle, square, zigzag2, zigzag3, zigzag4")
    parser.add_argument('--dummy_action_repeats', type=int, default=1,
                        help="number of times Dummy repeats dummy_action_sequence before STOP")
    parser.add_argument('--dummy_action_sleep_s', type=float, default=0.0,
                        help="optional sleep before each Dummy action decision")
    parser.add_argument('--dummy_action_max_sim_time_s', type=float, default=3.0,
                        help="maximum allowed simulator seconds for each Dummy action-test action")
    parser.add_argument('--dummy_action_pos_threshold_m', type=float, default=0.12,
                        help="position tolerance for Dummy action-test success")
    parser.add_argument('--dummy_action_yaw_threshold_deg', type=float, default=7.0,
                        help="yaw tolerance for Dummy action-test success")
    parser.add_argument('--dummy_action_turn_deg', type=float, default=30.0,
                        help="target yaw change for Dummy TURN_* action tests")
    parser.add_argument('--action_wait_timeout_s', type=float, default=None,
                        help=f"wall-clock fallback timeout while waiting for a waypoint action; defaults to {DEFAULT_EPISODE_TIMEOUT_S:.0f}s")
    parser.add_argument('--action_wait_timeout_sim_s', type=float, default=2.0,
                        help="sim-time timeout while waiting for a waypoint action; default 2s aborts the wait if the robot can't reach the target within 2 sim-seconds")
    parser.add_argument('--action_wait_pos_threshold_m', type=float, default=None,
                        help="position tolerance while waiting for a waypoint action; defaults to backend preset")
    parser.add_argument('--action_wait_yaw_threshold_deg', type=float, default=None,
                        help="yaw tolerance while waiting for a waypoint action; defaults to backend preset")
    parser.add_argument('--discrete_action_forward_distance_m', type=float, default=None,
                        help="override forward distance for discrete FORWARD actions; defaults to backend preset")
    parser.add_argument('--llm_reperception', action="store_true",
                        help="if set, the llm will be used to re-perceive the objects")
    parser.add_argument('--reperception_llm', type=str, default="llama3.2-vision")
    
    # planning 
    parser.add_argument("--motion_planner",choices=["astar", "fmm"],default="fmm",help="Select global path planner")
    parser.add_argument('--imagine_nav_planner', default=True,
                        help="if set, the planner will use the Imagine Nav planner which includes exploration and exploitation scores")
    parser.add_argument('--rl_planner', action="store_true",
                        help="if set, the planner will use the RL planner")
    parser.add_argument('--no_llm', action="store_true", default=False,
                        help="if set, disable region caption and scene graph prediction")
    parser.add_argument('--no_deal_stair_spawn', action="store_true",
                        help="if set, the agent will not deal with case when spawn on stairs")
    parser.add_argument('--early_climb', action="store_true",
                        help="if set, the agent will climb stairs earlier according to the criteria")
    parser.add_argument('--edge_goal', action="store_true",
                        help="if set, the agent will add the edge goal to navigate")
    parser.add_argument('--obj_merge_threshold', type=float, default=0.4,
                        help="The threshold for merging objects with detection results.")

    # debugging
    parser.add_argument('-v', '--verbose', action="store_true", help='if true, print verbose messages')
    parser.add_argument('--save_step_data', action="store_true", help='if true, save all debug data')
    parser.add_argument('--disable_action', action="store_true", help='if true, disable action output')
    parser.add_argument('--debug', action="store_true", help='if true, raise exceptions; if false, print exceptions and continue')
    parser.add_argument('--debug_frontier_score', action="store_false", help='label frontier in semantic map and print scores in visualization')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help="skip the first n frames")
    parser.add_argument('--keyboard_actor', action="store_true",
                        help="if set, the actor will be controlled by the keyboard")
    # parser.add_argument('--stair_climbing_agent', action="store_true")
    # parser.add_argument('--no_resume', action="store_true")
    # parser.add_argument('--stair_aim', type=int, default=1,
    #                     help="upstair=1, downstair=0")
    parser.add_argument('--no_stair_climbing', action="store_true",
                        help="if set, the agent will not climb stairs")
    parser.add_argument('--save_perception_results', action="store_true",
                        help="if set, the perception results will be saved")
    parser.add_argument("--gt_perception", action="store_true", help="if set, the gt perception will be generated")
    parser.add_argument("--gt_scenegraph", action="store_true", help="if set, the gt scene graph will be generated")
    parser.add_argument("--gt_goal_position", action="store_true", help="if set, the gt goal position will be given in the obs")
    
    # Tunable Parameters
    parser.add_argument('--ROBOT_HEIGHT', type=float, default=0.7,
                        help='Height of the robot (default: 0.7)')
    parser.add_argument('--FORWARD_DISTANCE', type=float, default=0.5,
                        help='Forward distance for each forward action (default: 0.5)')
    parser.add_argument('--TURN_ANGLE', type=float, default=30.0,
                        help='Turn angle for each turn action (default: 30.0)')
    parser.add_argument('--GRADIENT_THRESHOLD', type=float, default=0.3,
                        help='Threshold for height gradient to be considered as stairs (default: 0.3)')
    parser.add_argument('--OBSTACLE_HEIGHT', type=float, default=0.3,
                        help='Height threshold for obstacles (default: 0.3)')
    parser.add_argument('--LOCAL_GRID_RANGE', type=int, default=40,
                        help='Radius for local region to compute the height gradient (default: 40)')
    parser.add_argument('--FLOOR_HEIGHT', type=float, default=1.0,
                        help='Height threshold for different floors (default: 1.0)')
    parser.add_argument('--FLOOR_POINT_THRESHOLD', type=int, default=800,
                        help='Threshold for new floor detection (default: 800)')
    parser.add_argument('--ACTUAL_CAMERA_HEIGHT', type=float, default=0.7,
                        help='Actual camera height (default: 1.0)')
    parser.add_argument('--AT_FRONTEIR_THRESHOLD', type=int, default=6,
                        help='Threshold for the robot to be considered at the goal (default: 6)')
    parser.add_argument('--NEAR_GOAL_DISTANCE', type=float, default=8,
                        help='Threshold for the robot to be considered at the goal (default: 8)')
    # parse arguments
    args, remaining = parser.parse_known_args(argv)

    return args, remaining
