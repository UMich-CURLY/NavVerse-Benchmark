from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    factory: str
    conda_env: str
    agent_vram_gb: float
    sim_vram_gb: float
    supported_tasks: tuple[str, ...] = ("objnav", "vln", "placenav")
    required_paths: tuple[str, ...] = field(default_factory=tuple)
    required_modules: tuple[str, ...] = field(default_factory=tuple)
    defaults: dict[str, Any] = field(default_factory=dict)
    validation_warning: str | None = None

    def probe(self, repo_root: Path = REPO_ROOT) -> tuple[bool, list[str]]:
        problems: list[str] = []
        for raw_path in self.required_paths:
            expanded = os.path.expandvars(raw_path)
            path = Path(expanded)
            if not path.is_absolute():
                path = repo_root / path
            if not path.exists():
                problems.append(f"missing {path}")
        for module in self.required_modules:
            if importlib.util.find_spec(module) is None:
                problems.append(f"missing Python module {module}")
        return not problems, problems


AGENT_REGISTRY: dict[str, AgentSpec] = {
    "Dummy": AgentSpec("Dummy", "Deterministic action smoke-test agent.", "baselines.dummy.dummy_agent:DummyAgent", "navverse", 1, 20),
    "Oracle": AgentSpec("Oracle", "Reference-path waypoint oracle.", "baselines.oracle.oracle_agent:OracleAgent", "navverse", 1, 20),
    "OracleDiscreteAction": AgentSpec("OracleDiscreteAction", "Reference-path oracle using discrete actions.", "baselines.oracle.oracle_discrete_action_agent:OracleDiscreteActionAgent", "navverse", 1, 20),
    "UniNaVid": AgentSpec(
        "UniNaVid", "Uni-NaVid 7B video navigation baseline.", "baselines.uninavid.uninavid_agent:UniNaVidAgent", "uninavid", 20, 20,
        required_paths=("baselines/submodules/Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2", "baselines/submodules/Uni-NaVid/model_zoo/eva_vit_g.pth"),
        defaults={"primary_camera": "pov", "mapping_cameras": "pov"},
    ),
    "LongNav": AgentSpec(
        "LongNav", "Original LongNav-R1 discrete-action evaluation policy.", "baselines.longnav_r1.longnav_agent:LongNavAgent", "navverse", 25, 60,
        required_paths=("baselines/longnav_r1/serve_longnav.py",), defaults={"primary_camera": "pov", "mapping_cameras": "pov"},
    ),
    "InternVLA-N1": AgentSpec(
        "InternVLA-N1", "Latest InternVLA-N1 DualVLN RGB dual-system policy.",
        "baselines.internvla_n1.internvla_n1_agent:InternVLAN1Agent", "navverse", 28, 60,
        supported_tasks=("vln",),
        required_paths=("baselines/submodules/InternNav", "baselines/internvla_n1/serve_internvla_n1.py"),
        defaults={"primary_camera": "pov", "mapping_cameras": "pov"},
    ),
    "StreamVLN": AgentSpec(
        "StreamVLN", "Streaming VLN policy with SlowFast context memory.",
        "baselines.streamvln.streamvln_agent:StreamVLNAgent", "navverse", 25, 60,
        supported_tasks=("vln",),
        required_paths=("baselines/submodules/StreamVLN", "baselines/streamvln/serve_streamvln.py"),
        defaults={"primary_camera": "pov", "mapping_cameras": "pov"},
    ),
    "GPT": AgentSpec(
        "GPT", "GPT navigation via the Codex runtime.",
        "baselines.gpt.gpt_agent:GPTAgent", "navverse", 1, 20,
        required_paths=("baselines/gpt/serve_gpt.py",),
        defaults={"primary_camera": "pov", "mapping_cameras": "pov"},
    ),
    "PoliFormer": AgentSpec(
        "PoliFormer", "PoliFormer object-navigation baseline.", "baselines.poliformer.poliformer_agent:PoliFormerAgent", "poliformer", 20, 16,
        supported_tasks=("objnav",), required_paths=("baselines/submodules/poliformer",),
        validation_warning="PoliFormer has not passed the public clean-install and action-mapping checks.",
    ),
}


def get_agent_spec(name: str) -> AgentSpec:
    try:
        return AGENT_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent {name!r}; choose one of: {', '.join(AGENT_REGISTRY)}") from exc


def agent_names() -> list[str]:
    return list(AGENT_REGISTRY)


def load_agent_class(name: str):
    spec = get_agent_spec(name)
    module_name, class_name = spec.factory.split(":", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def instantiate_agent(
    name: str,
    args: Any,
    evaluation_over: Any = None,
    step_data_queue: Any = None,
    foxglove_log_queue: Any = None,
):
    agent_class = load_agent_class(name)
    if name == "PoliFormer":
        return agent_class(args, foxglove_log_queue)
    return agent_class(args)
