from types import SimpleNamespace
from unittest import mock

import numpy as np

from baselines.internvla_n1.internvla_n1_agent import InternVLAN1Agent
from baselines.preset_config import load_preset
from baselines.streamvln.streamvln_agent import StreamVLNAgent
from navverse.launcher.registry import AGENT_REGISTRY


def _args(**overrides):
    values = {
        "process_label": 0,
        "task_type": "objnav",
        "hfov": 79.0,
        "internvla_n1_server_url": "http://internvla.test/act",
        "internvla_n1_action_map": "STOP,FORWARD,TURN_LEFT_SMALL,TURN_RIGHT_SMALL",
        "internvla_n1_request_timeout": 10.0,
        "internvla_n1_max_steps": 500,
        "internvla_n1_depth_scale_m": 10.0,
        "internvla_n1_num_history": 8,
        "internvla_n1_plan_step_gap": 8,
        "streamvln_server_url": "http://streamvln.test/act",
        "streamvln_action_map": "STOP,FORWARD,TURN_LEFT_SMALL,TURN_RIGHT_SMALL",
        "streamvln_request_timeout": 10.0,
        "streamvln_max_steps": 500,
        "streamvln_attn_impl": "flash_attention_2",
        "streamvln_tokenizer": "Qwen/Qwen2-7B-Instruct",
        "streamvln_num_history": 8,
        "streamvln_num_frames": 32,
        "streamvln_num_future_steps": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _observation():
    return {
        "rgb": np.zeros((8, 12, 3), dtype=np.uint8),
        "depth": np.ones((8, 12), dtype=np.float32),
        "objectgoal": "chair",
        "hfov": 79.0,
        "position": np.array([0.0, 0.0, 0.0]),
        "rotation": np.array([0.0, 0.0, 0.0, 1.0]),
    }


def _response(action_id):
    response = mock.Mock()
    response.json.return_value = {"action_id": action_id}
    return response


def test_internvla_dualvln_posts_rgb_only_and_resets_once():
    agent = InternVLAN1Agent(_args())
    with mock.patch(
        "baselines.base.http_action_agent.requests.post",
        side_effect=[_response(2), _response(1)],
    ) as post:
        assert agent.act(_observation()) == {"action": "TURN_LEFT_SMALL"}
        assert agent.act(_observation()) == {"action": "FORWARD"}

    first = post.call_args_list[0].kwargs
    second = post.call_args_list[1].kwargs
    assert set(first["files"]) == {"rgb"}
    assert first["data"]["instruction"] == "Find the chair and stop when you are close to it."
    assert first["data"]["reset"] == "true"
    assert second["data"]["reset"] == "false"


def test_streamvln_posts_rgb_without_depth():
    agent = StreamVLNAgent(_args(task_type="vln"))
    observation = _observation()
    observation["objectgoal"] = "Walk through the doorway and stop by the sofa."
    with mock.patch(
        "baselines.base.http_action_agent.requests.post",
        return_value=_response(3),
    ) as post:
        assert agent.act(observation) == {"action": "TURN_RIGHT_SMALL"}

    request = post.call_args.kwargs
    assert set(request["files"]) == {"rgb"}
    assert request["data"]["instruction"] == observation["objectgoal"]


def test_internvla_trajectory_is_returned_as_absolute_waypoints():
    agent = InternVLAN1Agent(_args())
    observation = _observation()
    observation["position"] = np.array([10.0, 20.0, 0.0])
    observation["rotation"] = np.array([0.0, 0.0, 2**-0.5, 2**-0.5])
    response = mock.Mock()
    response.json.return_value = {"trajectory": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]}

    with mock.patch("baselines.base.http_action_agent.requests.post", return_value=response):
        result = agent.act(observation)

    np.testing.assert_allclose(
        result["waypoint"],
        [[10.0, 21.0, np.pi / 2.0], [9.0, 21.0, -np.pi]],
        atol=1e-7,
    )


def test_new_agents_are_registered_with_official_upstreams():
    assert AGENT_REGISTRY["InternVLA-N1"].required_paths[0] == "baselines/submodules/InternNav"
    assert AGENT_REGISTRY["StreamVLN"].required_paths[0] == "baselines/submodules/StreamVLN"
    assert AGENT_REGISTRY["InternVLA-N1"].supported_tasks == ("vln",)
    assert AGENT_REGISTRY["StreamVLN"].supported_tasks == ("vln",)


def test_new_presets_keep_navverse_forward_distance():
    for name, key in (
        ("benchmark_internvla_n1", "internvla_n1_action_map"),
        ("benchmark_streamvln", "streamvln_action_map"),
    ):
        arguments = load_preset(name)["arguments"]
        assert "discrete_action_forward_distance_m" not in arguments
        assert arguments[key] == "STOP,FORWARD,TURN_LEFT_SMALL,TURN_RIGHT_SMALL"
