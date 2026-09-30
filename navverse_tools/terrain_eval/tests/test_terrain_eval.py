"""Tests for the self-contained terrain-evaluation tooling."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from navverse_tools.terrain_eval.terrain import (
    ControlledTerrainManager,
    PATCH_CENTER_FRACTIONS,
    SLOPE_ANGLE_DEG,
    TERRAIN_PROFILES,
    _height_grid,
    load_controlled_terrain_manifest,
    path_length_xy,
    shared_patch_placements,
)
from navverse.measures import (
    CoveredArea,
    HorizontalPathLength,
    MeasureManager,
    NavigableSurface,
)

from navverse_tools.terrain_eval import run_cell as run_controlled_terrain_direct
from navverse_tools.terrain_eval import (
    run_material_matrix as run_controlled_material_matrix,
)
from navverse_tools.terrain_eval import (
    run_reference_sharded as run_controlled_reference_sharded,
)


class PoseEnv:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def get_body_pose(self):
        return self.position, np.asarray([0.0, 0.0, 0.0, 1.0])


def test_shared_patch_placements_are_fixed_at_three_path_fractions():
    path = [[float(x), 0.0, 1.0] for x in range(25)]
    placements = shared_patch_placements(path)
    assert placements is not None
    assert [row["center_fraction"] for row in placements] == list(PATCH_CENTER_FRACTIONS)
    assert [row["center_s_m"] for row in placements] == [6.0, 12.0, 18.0]


def test_short_or_sharply_turning_paths_are_rejected():
    assert shared_patch_placements([[0.0, 0.0, 0.0], [19.0, 0.0, 0.0]]) is None
    path = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [20.0, 5.0, 0.0]]
    assert shared_patch_placements(path) is None


def test_slope_is_mild_and_returns_to_ground_height():
    x, _, heights = _height_grid("slope", seed=1)
    assert np.isclose(heights[0].max(), 0.0)
    assert np.isclose(heights[-1].max(), 0.0)
    expected_peak = np.tan(np.deg2rad(SLOPE_ANGLE_DEG)) * 2.0
    assert np.isclose(heights[len(x) // 2].max(), expected_peak)


def test_uneven_mesh_is_reused_deterministically():
    _, _, first = _height_grid("uneven", seed=123)
    _, _, second = _height_grid("uneven", seed=123)
    _, _, different = _height_grid("uneven", seed=124)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.allclose(first[[0, -1], :], 0.0)
    assert np.allclose(first[:, [0, -1]], 0.0)


def test_hard_profile_increases_relief_without_changing_patch_edges():
    _, _, standard_slope = _height_grid("slope", seed=1)
    _, _, hard_slope = _height_grid("slope", seed=1, profile="hard")
    assert hard_slope.max() > standard_slope.max()
    assert np.allclose(hard_slope[[0, -1], :], 0.0)

    _, _, hard_uneven = _height_grid("uneven", seed=123, profile="hard")
    assert hard_uneven.max() > 0.06
    assert hard_uneven.max() <= TERRAIN_PROFILES["hard"]["uneven_height_range_m"][1]
    assert np.allclose(hard_uneven[[0, -1], :], 0.0)
    assert np.allclose(hard_uneven[:, [0, -1]], 0.0)


def test_extra_hard_profile_expands_and_increases_relief():
    extra = TERRAIN_PROFILES["extra_hard"]
    x, y, slope = _height_grid("slope", seed=1, profile="extra_hard")
    _, _, uneven = _height_grid("uneven", seed=123, profile="extra_hard")

    assert np.isclose(x[-1] - x[0], 5.0)
    assert np.isclose(y[-1] - y[0], 2.5)
    assert slope.max() > _height_grid("slope", seed=1, profile="hard")[2].max()
    assert uneven.max() > TERRAIN_PROFILES["hard"]["uneven_height_range_m"][1]
    assert uneven.max() <= extra["uneven_height_range_m"][1]
    assert np.allclose(slope[[0, -1], :], 0.0)
    assert np.allclose(uneven[[0, -1], :], 0.0)
    assert np.allclose(uneven[:, [0, -1]], 0.0)


def test_balanced_profile_is_between_hard_and_extra_hard():
    hard = TERRAIN_PROFILES["hard"]
    balanced = TERRAIN_PROFILES["balanced"]
    extra = TERRAIN_PROFILES["extra_hard"]

    for key in ("patch_length_m", "patch_width_m", "slope_angle_deg", "stair_height_m"):
        assert hard[key] < balanced[key] < extra[key]
    assert hard["uneven_height_range_m"][1] < balanced["uneven_height_range_m"][1]
    assert balanced["uneven_height_range_m"][1] < extra["uneven_height_range_m"][1]


def test_moderate_profile_expands_area_with_midrange_relief():
    standard = TERRAIN_PROFILES["standard"]
    moderate = TERRAIN_PROFILES["moderate"]
    hard = TERRAIN_PROFILES["hard"]

    assert moderate["patch_length_m"] > hard["patch_length_m"]
    assert moderate["patch_width_m"] > hard["patch_width_m"]
    assert standard["slope_angle_deg"] < moderate["slope_angle_deg"] <= hard["slope_angle_deg"]
    assert standard["stair_height_m"] < moderate["stair_height_m"] < hard["stair_height_m"]
    assert standard["uneven_height_range_m"][1] < moderate["uneven_height_range_m"][1]
    assert moderate["uneven_height_range_m"][1] < hard["uneven_height_range_m"][1]


def test_moderate_v2_preserves_area_and_reduces_relief():
    moderate = TERRAIN_PROFILES["moderate"]
    moderate_v2 = TERRAIN_PROFILES["moderate_v2"]

    assert moderate_v2["patch_length_m"] == moderate["patch_length_m"]
    assert moderate_v2["patch_width_m"] == moderate["patch_width_m"]
    assert moderate_v2["slope_angle_deg"] == moderate["slope_angle_deg"]
    assert moderate_v2["stair_height_m"] < moderate["stair_height_m"]
    assert (
        moderate_v2["uneven_height_range_m"][1]
        < moderate["uneven_height_range_m"][1]
    )


def test_moderate_final_keeps_v2_except_for_traversable_step():
    moderate_v2 = TERRAIN_PROFILES["moderate_v2"]
    final = TERRAIN_PROFILES["moderate_final"]

    assert final["patch_length_m"] == moderate_v2["patch_length_m"]
    assert final["patch_width_m"] == moderate_v2["patch_width_m"]
    assert final["slope_angle_deg"] == moderate_v2["slope_angle_deg"]
    assert final["uneven_height_range_m"] == moderate_v2["uneven_height_range_m"]
    assert final["stair_height_m"] < moderate_v2["stair_height_m"]


def test_moderate_compact_retains_expanded_area():
    standard = TERRAIN_PROFILES["standard"]
    final = TERRAIN_PROFILES["moderate_final"]
    compact = TERRAIN_PROFILES["moderate_compact"]

    assert standard["patch_length_m"] < compact["patch_length_m"]
    assert compact["patch_length_m"] < final["patch_length_m"]
    assert compact["patch_width_m"] == final["patch_width_m"]
    assert compact["slope_angle_deg"] == final["slope_angle_deg"]
    assert compact["stair_height_m"] == final["stair_height_m"]
    assert compact["uneven_height_range_m"] == final["uneven_height_range_m"]


def test_moderate_wide_expands_area_without_moving_longitudinal_edges():
    standard = TERRAIN_PROFILES["standard"]
    wide = TERRAIN_PROFILES["moderate_wide"]

    assert wide["patch_length_m"] == standard["patch_length_m"]
    assert wide["patch_width_m"] > standard["patch_width_m"]
    assert (
        wide["patch_length_m"] * wide["patch_width_m"]
        > standard["patch_length_m"] * standard["patch_width_m"]
    )


def test_moderate_traversable_uses_turtlebot_scale_step():
    wide = TERRAIN_PROFILES["moderate_wide"]
    traversable = TERRAIN_PROFILES["moderate_traversable"]

    assert traversable["patch_length_m"] == wide["patch_length_m"]
    assert traversable["patch_width_m"] == wide["patch_width_m"]
    assert traversable["slope_angle_deg"] == wide["slope_angle_deg"]
    assert traversable["uneven_height_range_m"] == wide["uneven_height_range_m"]
    assert traversable["stair_height_m"] == 0.02


def test_micro_rough_profile_uses_millimeter_relief_and_finer_grid():
    x, y, uneven = _height_grid("uneven", seed=123, profile="micro_rough")

    assert np.isclose(x[1] - x[0], 0.1)
    assert np.isclose(y[1] - y[0], 0.1)
    assert 0.002 <= uneven.max() <= 0.006
    assert np.allclose(uneven[[0, -1], :], 0.0)
    assert np.allclose(uneven[:, [0, -1]], 0.0)


def test_micro_rough_profiles_preserve_ordered_millimeter_relief():
    fine = _height_grid("uneven", seed=123, profile="micro_rough_fine")[2]
    medium = _height_grid("uneven", seed=123, profile="micro_rough_medium")[2]
    coarse = _height_grid("uneven", seed=123, profile="micro_rough")[2]

    assert fine.max() < medium.max() < coarse.max()
    assert TERRAIN_PROFILES["micro_rough_fine"]["z_offset_m"] < 0.001


def test_manifest_requires_paired_10_by_10_structure(tmp_path):
    episodes = []
    for scene_idx in range(10):
        for episode_idx in range(10):
            episodes.append(
                {
                    "scene_id": f"vcp_scene_{scene_idx}",
                    "episode_label": f"vcp_scene_{scene_idx}_{episode_idx}",
                    "patches": [{}, {}, {}],
                }
            )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"episodes": episodes}))
    manifest = load_controlled_terrain_manifest(path)
    assert len(manifest["_episode_map"]) == 100
    assert path_length_xy([[0, 0], [3, 4]]) == 5.0


def test_original_condition_does_not_touch_stage(tmp_path):
    episodes = []
    for scene_idx in range(10):
        for episode_idx in range(10):
            episodes.append(
                {
                    "scene_id": f"vcp_scene_{scene_idx}",
                    "episode_label": f"vcp_scene_{scene_idx}_{episode_idx}",
                    "patches": [{}, {}, {}],
                }
            )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"episodes": episodes}))
    manager = ControlledTerrainManager(path, "original")

    metrics = manager.configure_episode(None, "vcp_scene_0_0")

    assert metrics["terrain_patches_reached"] == 0.0
    assert metrics["terrain_patches_completed"] == 0.0


def test_horizontal_metrics_ignore_nonfinite_terminal_pose():
    env = PoseEnv([10.0, 20.0, 1.0])
    manager = MeasureManager()
    path_length = HorizontalPathLength(env, {}, manager)
    covered_area = CoveredArea(env, {}, manager)
    path_length.reset_metric()
    covered_area.reset_metric()

    env.position = np.asarray([11.0, 20.0, 1.0])
    path_length.update_metric()
    covered_area.update_metric()
    env.position = np.asarray([np.nan, np.nan, np.nan])
    path_length.update_metric()
    covered_area.update_metric()

    assert path_length.get_metric() == 1.0
    assert covered_area.get_metric() == 2.0


def test_navigable_surface_ignores_nonfinite_reset_pose():
    class SurfaceMap:
        def is_navigable_surface(self, _position):
            raise AssertionError("surface map must not receive a non-finite pose")

    env = PoseEnv([np.nan, np.nan, np.nan])
    measure = NavigableSurface(env, {}, MeasureManager())
    measure._surface_map = SurfaceMap()
    measure.update_metric()

    assert np.isnan(measure.get_metric())
    assert measure.surface_type == "unknown"


def test_scene_retry_skips_results_already_written(monkeypatch):
    calls = []
    sink = SimpleNamespace(rows=[])

    def run_scene(_repo_root, _args, labels, result_sink, _scene_index):
        calls.append(list(labels))
        if len(calls) == 1:
            result_sink.rows.append({"episode_label": labels[0]})
            raise RuntimeError("transient simulator exit")

    monkeypatch.setattr(run_controlled_terrain_direct, "run_scene", run_scene)
    monkeypatch.setattr(run_controlled_terrain_direct.time, "sleep", lambda _seconds: None)
    run_controlled_terrain_direct.run_scene_with_retries(
        None,
        SimpleNamespace(scene_retries=1),
        ["vcp_scene_0", "vcp_scene_1"],
        sink,
        0,
    )

    assert calls == [["vcp_scene_0", "vcp_scene_1"], ["vcp_scene_1"]]


def test_direct_runner_allows_distinct_material_cell_names():
    args = SimpleNamespace(cell_name="material_mu0p5_spot", terrain="original", robot="spot")
    assert run_controlled_terrain_direct.resolve_cell_name(args) == "material_mu0p5_spot"

    args.cell_name = ""
    assert run_controlled_terrain_direct.resolve_cell_name(args) == "original_spot"


def test_material_result_validation_requires_exact_filtered_set(tmp_path):
    labels = [f"vcp_scene_{index}" for index in range(80)]
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            [
                {
                    "episode_label": label,
                    "success": 1.0,
                    "average_horizontal_speed": 1.0,
                }
                for label in labels
            ]
        )
    )

    valid, message = run_controlled_material_matrix.validate_result(
        result_path, labels
    )

    assert valid
    assert message == "80/80 exact unique results"


def test_simulator_fatal_log_detection(tmp_path):
    log_path = tmp_path / "sim.log"
    log_path.write_text("normal simulator output\n")
    assert not run_controlled_terrain_direct.simulator_log_has_fatal_error(log_path)

    log_path.write_text(
        "normal simulator output\nWarp CUDA error 700: an illegal memory access was encountered\n"
    )
    assert run_controlled_terrain_direct.simulator_log_has_fatal_error(log_path)

    retry_offset = log_path.stat().st_size
    log_path.write_text(log_path.read_text() + "healthy retry output\n")
    assert not run_controlled_terrain_direct.simulator_log_has_fatal_error(
        log_path, retry_offset
    )


def test_contiguous_shards_preserve_manifest_order():
    labels = [f"episode_{index}" for index in range(100)]
    shards = run_controlled_reference_sharded.contiguous_shards(labels, 8)

    assert [len(shard) for shard in shards] == [13, 13, 13, 13, 12, 12, 12, 12]
    assert [label for shard in shards for label in shard] == labels


def test_sharded_merge_restores_manifest_order(tmp_path):
    labels = [f"episode_{index}" for index in range(8)]
    shards = run_controlled_reference_sharded.contiguous_shards(labels, 2)
    for shard_index, shard_labels in enumerate(shards):
        result_dir = (
            tmp_path
            / "shards"
            / f"shard_{shard_index:02d}"
            / "original_spot"
        )
        result_dir.mkdir(parents=True)
        result_dir.joinpath("result.json").write_text(
            json.dumps(
                [
                    {
                        "episode_label": label,
                        "success": 1.0,
                        "average_horizontal_speed": 1.0,
                        "terrain_patches_completed": 0.0,
                    }
                    for label in reversed(shard_labels)
                ]
            )
        )

    merged = run_controlled_reference_sharded.merge_shard_results(tmp_path, shards)

    assert [row["episode_label"] for row in merged] == labels


def test_summary_honors_selected_robot_and_terrain(tmp_path):
    labels = [f"vcp_scene_{index}" for index in range(100)]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"episodes": [{"episode_label": label} for label in labels]})
    )
    run_root = tmp_path / "run"
    cell_dir = run_root / "flat_turtlebot"
    cell_dir.mkdir(parents=True)
    cell_dir.joinpath("result.json").write_text(
        json.dumps(
            [
                {
                    "episode_label": label,
                    "success": 1.0,
                    "average_horizontal_speed": 0.5,
                    "terrain_patches_completed": 3.0,
                    "termination_reason": "stop_called",
                }
                for label in labels
            ]
        )
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "navverse_tools.terrain_eval.summarize",
            str(run_root),
            "--manifest",
            str(manifest_path),
            "--robots",
            "turtlebot",
            "--terrains",
            "flat",
        ],
        check=True,
    )

    summary = json.loads(run_root.joinpath("summary.json").read_text())
    assert summary["validation_errors"] == []
    assert summary["robots"] == ["turtlebot"]
    assert summary["terrains"] == ["flat"]
    assert set(summary["cells"]) == {"flat_turtlebot"}
