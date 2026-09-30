import numpy as np
import pytest

from baselines.longnav_r1.serve_longnav import discrete_action_response
from baselines.longnav_r1.longnav_agent import LongNavAgent


def test_discrete_response_keeps_four_action_contract():
    assert discrete_action_response(2, np.array([0.1, 0.2, 0.3, 0.4])) == {
        "action_id": 2,
        "action_probs": [0.1, 0.2, 0.3, 0.4],
    }
    with pytest.raises(RuntimeError):
        discrete_action_response(4, np.ones(5))


def test_step_cap_stop_preserves_last_action_record():
    agent = LongNavAgent.__new__(LongNavAgent)
    agent.max_steps = 20
    agent.l_step = 20
    history = {20: {"action": "FORWARD"}}

    action = agent.act({})
    history[agent.l_step] = action

    assert history == {20: {"action": "FORWARD"}, 21: {"action": "STOP"}}
