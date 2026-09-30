import json
from types import SimpleNamespace

import pytest

from baselines.oracle.oracle_agent import OracleAgent
from baselines.oracle.oracle_discrete_action_agent import OracleDiscreteActionAgent


@pytest.mark.parametrize("agent_class", [OracleAgent, OracleDiscreteActionAgent])
def test_oracle_loads_external_episode_folder(tmp_path, monkeypatch, agent_class):
    episode_dir = tmp_path / "external_dataset"
    episode_dir.mkdir()
    path = [[0, 0, 0], [2, 1, 0]]
    (episode_dir / "example_scene.json").write_text(json.dumps([{
        "scene_id": "example_scene",
        "episode_id": 1,
        "goals": [{"reference_path": path, "radius": 0.3}],
    }]))
    monkeypatch.chdir(tmp_path)
    agent = agent_class(SimpleNamespace(episode_folder=str(episode_dir)))
    assert agent._load_reference_path({
        "scene_id": "example_scene", "episode_label": "example_scene_1",
    }) == path
    assert agent.goal_radius == 0.3
