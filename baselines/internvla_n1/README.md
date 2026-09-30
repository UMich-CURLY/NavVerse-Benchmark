# InternVLA-N1

This adapter runs the official InternNav implementation in an isolated inference
server. It sends DualVLN trajectories to NavVerse as waypoints and retains the
four-action output as a fallback.

The default checkpoint is
[`InternRobotics/InternVLA-N1-DualVLN`](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN).
It is the latest RGB dual-system release and the strongest published InternVLA-N1
variant in the official model table as of 2026-08-05. The checkpoint is downloaded
to the Hugging Face cache on first use.

The DualVLN `nextdit_async` trajectory generator uses RGB features and does not
consume the depth tensor accepted by the upstream real-world wrapper. This adapter
therefore makes depth optional and uses a zero placeholder for that wrapper argument.
NavDP-based checkpoints still require a real depth observation.

## Setup

```bash
bash baselines/internvla_n1/setup.sh
```

The setup defaults to CUDA 12.8 PyTorch and FlashAttention builds so the server
can run on this repository's RTX PRO 6000 Blackwell hosts. Version environment
variables in `setup.sh` can be overridden on other hardware.

## Run

```bash
navverse-run benchmark --preset benchmark_internvla_n1
```

Useful overrides:

```bash
INTERNVLA_N1_CHECKPOINT=/path/to/checkpoint \
INTERNVLA_N1_SERVER_PYTHON=/path/to/python \
navverse-run benchmark --preset benchmark_internvla_n1
```

The benchmark process and model server use separate Python environments. One
server is started per NavVerse agent worker on `8100 + worker_id`; change the base
port with `INTERNVLA_N1_BASE_PORT`.

The preset leaves NavVerse's scene-specific `FORWARD` behavior unchanged and
maps model turn ids to the 15-degree `TURN_*_SMALL` actions. DualVLN continuous
trajectory outputs are transformed into NavVerse waypoints instead of being
quantized into discrete actions.
