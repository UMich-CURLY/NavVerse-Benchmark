import unittest

from baselines.longnav_r1.serve_longnav import discrete_action_response as longnav_response
from navverse_tools.real_world_exp_server.serve_poliformer import discrete_action_response as poliformer_response


class DiscreteAgentContractTest(unittest.TestCase):
    def test_longnav_returns_action_id_without_trajectory(self):
        response = longnav_response(2, [0.1, 0.2, 0.6, 0.1])
        self.assertEqual(response["action_id"], 2)
        self.assertNotIn("trajectory", response)

    def test_longnav_rejects_out_of_range_action(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported action_id"):
            longnav_response(4, [0, 0, 0, 0, 1])

    def test_poliformer_strips_non_action_fields(self):
        response = poliformer_response(
            {"action": "TURN_LEFT_SMALL", "trajectory": [[0, 0], [1, 0]]}
        )
        self.assertEqual(response, {"action": "TURN_LEFT_SMALL"})

    def test_poliformer_rejects_unknown_action(self):
        with self.assertRaisesRegex(RuntimeError, "no supported action"):
            poliformer_response({"action": "FOLLOW_TRAJECTORY"})


if __name__ == "__main__":
    unittest.main()
