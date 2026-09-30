import os
import sys
import torch

# Make the PoliFormer submodule (cloned under baselines/submodules/poliformer/)
# importable as top-level packages: its internal modules reference
# `architecture.X`, `training.X`, `poliformer_utils.X` without a parent prefix.
_BASELINES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASELINES_DIR, "submodules", "poliformer"))

from baselines.base.base_agent import BaseAgent

from training.online.online_eval import (
    InferenceAgentVIDA,
    model_config_type,
    model_config_params,
    img_encoder_type,
    parse_args,
)
from poliformer_utils.constants.stretch_initialization_utils import stretch_long_names

def load_agent(device):
    args = parse_args()
    params = model_config_params[args.model_config]()
    params.num_train_processes = 0
    if args.ignore_text_goal:
        params.use_text_goal = False
    if any("box" in s for s in args.input_sensors):
        params.use_bbox = True
    else:
        params.use_bbox = False
    agent_input = dict(
        exp_config_type=model_config_type[args.model_config],
        params=params,
        img_encoder_rgb_mean=img_encoder_type[args.img_encoder_type]["mean"],
        img_encoder_rgb_std=img_encoder_type[args.img_encoder_type]["std"],
        greedy_sampling=args.greedy,
        test_augmentation=args.test_augmentation,
        ckpt_path=args.ckpt_path,
    )
    print("start building agent")
    agent = InferenceAgentVIDA.build_agent(**agent_input, device=device)
    print("agent built", flush=True)
    if hasattr(agent, "model"):
        agent.model.eval()
    elif hasattr(agent, "actor_critic"):
        agent.actor_critic.eval()
    else:
        raise NotImplementedError
    return args, agent

class PoliFormerAgent(BaseAgent):
    def __init__(self, args, log_queue=None):
        super().__init__(args)
        self.args = args
        self.l_step = 0
        self.log_queue = log_queue
        self.device = torch.device("cuda")
        # poliformer args and agent
        self.poliformer_args, self.agent = load_agent(self.device)
        self.action_list = self.agent.get_action_list()
        self.input_sensors = self.poliformer_args.input_sensors
        print(f"PoliFormerAgent loaded with input sensors: {self.input_sensors}")
    def reset(self):
        self.l_step = 0
        self.vis_frames = {}
        self.agent.reset()
        print("PoliFormerAgent reset")
    @torch.no_grad()
    def act(self, obs, step_data_queue=None):
        self.l_step += 1
        agent = self.agent
        poliformer_obs = {
            "raw_navigation_camera": obs["rgb"].copy(),
        }
        goal = obs['objectgoal']
        action, probs = agent.get_action(poliformer_obs, goal)
        action_name = stretch_long_names[action]
        print(f"PoliFormerAgent action: {action_name}")
        action_map = {
            "done": "STOP",
            "move_ahead": "FORWARD",
            "rotate_right": "TURN_RIGHT",
            "rotate_left": "TURN_LEFT",
            "rotate_left_small": "TURN_LEFT_SMALL",
            "rotate_right_small": "TURN_RIGHT_SMALL",
        } # move_back, subdone
        if action_name in action_map:
            return {"action": action_map[action_name]}
        else:
            return None
