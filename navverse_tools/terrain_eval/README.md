# Terrain evaluation

This folder contains the reproducible tooling for the material, terrain, and
embodiment isolation experiment. The main NavVerse runtime only exposes thin
hooks; terrain geometry, material overrides, frozen-manifest preparation,
multi-GPU scheduling, and aggregation live here.

## Layout

- `terrain.py`: deterministic slope, single-step stairs, and rocky terrain.
- `drone_env_cfg.py`: fixed-height, collision-free controlled-evaluation drone.
- `runtime.py`: the small adapter called by the simulator environment wrapper.
- `prepare_manifest.py`: freezes shared scenes, episodes, and patch placements.
- `run_cell.py`: executes one terrain and embodiment cell.
- `run_matrix.py`: schedules the standard shared-manifest matrix.
- `run_profile_matrix.py`: schedules a selected profile and episode subset.
- `run_material_matrix.py`: runs the material friction matrix.
- `run_surface_sweep.py`: small material calibration sweeps.
- `turtlebot_motion_smoke.py`: forward, left-turn, and right-turn motion check.
- `summarize.py`: validates and aggregates per-episode measurements.
- `tests/`: deterministic geometry, manifest, metric, and scheduler tests.

Run tools from the repository root as modules, for example:

```bash
conda activate navverse
python -m navverse_tools.terrain_eval.run_matrix --gpu-ids 1,2,3
```

All executions write transient artifacts under `dump/`, which is ignored by
Git. Result JSON, logs, videos, generated tables, and local run summaries are
not part of the source release and must not be committed. Only code, frozen
input manifests, tests, and documentation belong in the repository.
