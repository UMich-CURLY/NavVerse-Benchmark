import numpy as np

from navverse.path_following import KinematicPIDPoseFollower


def test_tidybot_pid_matches_habitat_pe_acceleration_limit_on_first_tick():
    follower = KinematicPIDPoseFollower()
    next_pose = follower.update(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.03, 0.0, 0.0]),
        0.04,
    )
    np.testing.assert_allclose(next_pose, [0.0128, 0.0, 0.0], atol=1e-9)


def test_tidybot_pid_tracks_one_reference_per_tick_without_teleporting():
    follower = KinematicPIDPoseFollower()
    pose = np.zeros(3, dtype=float)
    references = np.column_stack(
        [np.arange(1, 11) * 0.03, np.zeros(10), np.zeros(10)]
    )
    actual = []
    for reference in references:
        pose = follower.update(pose, reference, 0.04)
        actual.append(pose.copy())
    actual = np.asarray(actual)
    assert np.all(np.diff(actual[:, 0]) > 0.0)
    assert np.max(np.diff(np.concatenate([[0.0], actual[:, 0]]))) <= 2.0 * 0.04
    assert actual[-1, 0] < references[-1, 0]


def test_tidybot_pid_uses_observed_pose_after_a_blocked_tick():
    follower = KinematicPIDPoseFollower()
    start = np.zeros(3, dtype=float)
    first = follower.update(start, np.array([0.03, 0.0, 0.0]), 0.04)
    assert first[0] > 0.0
    second = follower.update(start, np.array([0.06, 0.0, 0.0]), 0.04)
    assert second[0] > 0.0
    assert np.isfinite(second).all()
