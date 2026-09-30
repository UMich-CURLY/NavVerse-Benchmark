import pytest

from navverse.timed_trajectory import control_decimation, trajectory_tick_index


def test_longnav_control_period_is_eight_isaac_physics_steps():
    assert control_decimation(0.005, 0.04) == 8


def test_control_period_must_land_on_the_physics_grid():
    with pytest.raises(ValueError, match="integer multiple"):
        control_decimation(0.008, 0.05)


def test_each_control_step_selects_exactly_one_reference():
    assert [trajectory_tick_index(step, 7, 3) for step in range(7, 11)] == [
        0,
        1,
        2,
        None,
    ]
