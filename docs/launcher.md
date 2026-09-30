# NavVerse Launcher

`navverse-run` provides four public tasks: `setup`, `download`, `build`, and
`benchmark`. It supports a keyboard-driven selector when invoked without a task
and a stable command-line interface for automation.

Before installing the package, use the module entrypoint:

```bash
python -m navverse.launcher.cli setup --env-name navverse
```

After setup, inspect the available interface with:

```bash
navverse-run --list-tasks
navverse-run --list-presets
navverse-run benchmark --help
```

Every explicit task accepts `--dry-run`. This resolves the task and backend and
prints the generated command without running it.

## Setup, download, and build

```bash
navverse-run setup --env-name navverse
navverse-run download
navverse-run download --raw
navverse-run build --image umich-curly/navverse:latest
```

The release downloader installs the runtime layout under `navverse_data/` by
default and updates `SCENE_FOLDER` and `EPISODE_FOLDER` in `.env`. `setup`,
`download`, and `build` are native host tasks; only `benchmark` supports the
Docker and SageMaker backends.

## Benchmark presets and precedence

Benchmark presets are packaged YAML files under `baselines/presets/`:

```bash
navverse-run benchmark --preset benchmark_uninavid_450
navverse-run benchmark --preset benchmark_longnav \
  --split-set benchmark450 --gpu-ids 0,1,2,3
```

Use `--config path/to/preset.yaml` for an external YAML file. Preset inheritance
is resolved first, environment-backed preset values are applied next, launcher
options override those values, and remaining evaluator arguments are appended
last. The evaluator therefore receives explicit command-line values after
preset defaults.

The same preset format can be used directly:

```bash
python baselines/eval_navverse.py --preset benchmark_streamvln \
  --episode_labels benchmark450 --gpu_ids 0,1
python baselines/eval_navverse.py \
  --config baselines/presets/benchmark_uninavid.yaml
```

`--split-set` selects a manifest under the configured episode folder. When a
split's episode JSON files live in an isolated subdirectory, the evaluator
resolves that directory before starting workers.

Benchmark episodes can carry `target_type: obj|place` and an `instructions`
list. Each item has `instruction_type: target|vln` and
`instruction_content`. A `target` instruction names the object or place;
`vln` contains a route description only when one exists. Select the text to
send with `--instruction_type target` (the default) or `vln`. The evaluator
passes each episode's `target_type` to the agent independently, so mixed
ObjectNav and PlaceNav splits can run in one launch. Selecting `vln` for an
episode without a route description fails explicitly.

## Docker

Docker uses the same resolved benchmark command and mounts the parent repository
plus any external episode or scene roots read-only:

Install and configure NVIDIA Container Toolkit's `nvidia` runtime on the host.
The launcher enables that runtime and graphics capabilities because Isaac Sim
needs Vulkan as well as CUDA; `--gpus` alone may leave Vulkan drivers unavailable.

```bash
navverse-run benchmark --preset benchmark_smoke \
  --backend docker --image umich-curly/navverse:latest --dry-run
```

## SageMaker

SageMaker requires an ECR image, scene input prefix, execution role, and output
bucket. Credentials remain in the normal AWS configuration and are never stored
in launcher YAML:

```bash
navverse-run benchmark --preset benchmark_uninavid_450 \
  --backend sagemaker \
  --image <account>.dkr.ecr.<region>.amazonaws.com/navverse-benchmark:<tag> \
  --scenes-s3-uri s3://<bucket>/navverse/scenes/ \
  --sagemaker-role-arn arn:aws:iam::<account>:role/<role> \
  --sagemaker-bucket <bucket> \
  --dry-run
```

The container entrypoint reads the resolved config channel from
`/opt/ml/input/data/config`, prepares the scene and optional UniNaVid channels,
and writes outputs to `/opt/ml/model`.

## Migration from legacy launch presets

`baselines/launch_preset.sh` and `configs/launch/` are removed. Replace:

```text
bash baselines/launch_preset.sh <preset>
```

with:

```text
navverse-run benchmark --preset <preset>
```

The launcher provides evaluation workflows. Set up the original LongNav-R1
policy with `bash baselines/longnav_r1/setup.sh`; see its adapter README for
standalone inference and external-server configuration.
