import unittest

from baselines.internvla_n1.serve_internvla_n1 import (
    discrete_action_ids,
    trajectory_to_action_ids,
)


class InternVLAActionAdapterTest(unittest.TestCase):
    def test_native_actions_take_precedence(self):
        self.assertEqual(discrete_action_ids([2, 1], [[0, 0], [1, 0]]), [2, 1])

    def test_straight_trajectory_becomes_forward_actions(self):
        self.assertEqual(trajectory_to_action_ids([[0, 0], [1, 0]]), [1, 1, 1, 1])

    def test_lateral_trajectories_turn_before_moving(self):
        left = trajectory_to_action_ids([[0, 0], [0, 1]])
        right = trajectory_to_action_ids([[0, 0], [0, -1]])
        self.assertEqual(left[:6], [2] * 6)
        self.assertEqual(right[:6], [3] * 6)
        self.assertEqual(left[6:], [1] * 4)
        self.assertEqual(right[6:], [1] * 4)

    def test_short_trajectory_becomes_stop(self):
        self.assertEqual(trajectory_to_action_ids([[0, 0], [0.1, 0]]), [0])

    def test_unsupported_native_action_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported actions"):
            discrete_action_ids([5], None)


if __name__ == "__main__":
    unittest.main()
