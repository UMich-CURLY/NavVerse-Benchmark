# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import spot

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{spot.__name__}.spot_rough_env_cfg:SpotRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{spot.__name__}.rsl_rl_ppo_cfg:SpotRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{spot.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Spot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{spot.__name__}.spot_rough_env_cfg:SpotRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{spot.__name__}.rsl_rl_ppo_cfg:SpotRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{spot.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
