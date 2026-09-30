from pathlib import Path

from baselines.base.http_action_agent import HTTPActionServerAgent


class StreamVLNAgent(HTTPActionServerAgent):
    def __init__(self, args):
        repo_root = Path(__file__).resolve().parents[2]
        super().__init__(
            args,
            name="StreamVLN",
            argument_prefix="streamvln",
            serve_script=repo_root / "baselines" / "streamvln" / "serve_streamvln.py",
            default_checkpoint="mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3",
            default_port=8200,
            default_env_name="streamvln",
            python_env_var="STREAMVLN_SERVER_PYTHON",
            server_cli_args=(
                "--tokenizer",
                getattr(args, "streamvln_tokenizer", "Qwen/Qwen2-7B-Instruct")
                or "Qwen/Qwen2-7B-Instruct",
                "--attn_impl",
                getattr(args, "streamvln_attn_impl", "flash_attention_2") or "flash_attention_2",
                "--num_history",
                str(getattr(args, "streamvln_num_history", 8) or 8),
                "--num_frames",
                str(getattr(args, "streamvln_num_frames", 32) or 32),
                "--num_future_steps",
                str(getattr(args, "streamvln_num_future_steps", 4) or 4),
            ),
        )
