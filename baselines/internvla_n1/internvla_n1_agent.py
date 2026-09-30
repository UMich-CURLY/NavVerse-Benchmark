from pathlib import Path

from baselines.base.http_action_agent import HTTPActionServerAgent


class InternVLAN1Agent(HTTPActionServerAgent):
    def __init__(self, args):
        repo_root = Path(__file__).resolve().parents[2]
        super().__init__(
            args,
            name="InternVLA-N1",
            argument_prefix="internvla_n1",
            serve_script=repo_root / "baselines" / "internvla_n1" / "serve_internvla_n1.py",
            default_checkpoint="InternRobotics/InternVLA-N1-DualVLN",
            default_port=8100,
            default_env_name="internvla_n1",
            python_env_var="INTERNVLA_N1_SERVER_PYTHON",
            uses_depth=False,
            server_cli_args=(
                "--num_history",
                str(getattr(args, "internvla_n1_num_history", 8) or 8),
                "--plan_step_gap",
                str(getattr(args, "internvla_n1_plan_step_gap", 8) or 8),
            ),
        )
