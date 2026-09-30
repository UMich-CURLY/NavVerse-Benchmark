import math


def control_decimation(physics_dt: float, control_dt: float) -> int:
    """Return the exact number of physics steps in one trajectory control tick."""
    physics_dt = float(physics_dt)
    control_dt = float(control_dt)
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError(f"physics_dt must be positive, got {physics_dt}")
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError(f"control_dt must be positive, got {control_dt}")
    decimation = round(control_dt / physics_dt)
    if decimation < 1 or not math.isclose(
        decimation * physics_dt, control_dt, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"trajectory control_dt={control_dt} must be an integer multiple of "
            f"physics_dt={physics_dt}"
        )
    return decimation


def trajectory_tick_index(current_step: int, start_step: int, point_count: int):
    """Map one simulator control step to one trajectory reference row."""
    point_count = int(point_count)
    if point_count <= 0:
        raise ValueError(f"point_count must be positive, got {point_count}")
    index = max(0, int(current_step) - int(start_step))
    return None if index >= point_count else index
