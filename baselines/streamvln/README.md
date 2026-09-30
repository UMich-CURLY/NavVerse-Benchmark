# StreamVLN

This adapter uses StreamVLN's online SlowFast memory and returns its predicted
action sequence to NavVerse one action at a time.

The default checkpoint is the official 2025-09-28 benchmark update,
[`mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3).
The checkpoint is downloaded to the Hugging Face cache on first use.

## Setup

```bash
bash baselines/streamvln/setup.sh
```

The setup keeps StreamVLN's Python dependencies but replaces its legacy
PyTorch 2.1 pin with CUDA 12.8 PyTorch/FlashAttention defaults for this
repository's RTX PRO 6000 Blackwell hosts.

## Run

```bash
navverse-run benchmark --preset benchmark_streamvln
```

Useful overrides:

```bash
STREAMVLN_CHECKPOINT=/path/to/checkpoint \
STREAMVLN_SERVER_PYTHON=/path/to/python \
navverse-run benchmark --preset benchmark_streamvln
```

One isolated model server is started per agent worker on `8200 + worker_id`;
change the base port with `STREAMVLN_BASE_PORT`. For physical-robot deployment,
the upstream project provides a separate real-world checkpoint; the NavVerse
benchmark preset intentionally uses the stronger benchmark checkpoint.

The preset leaves NavVerse's scene-specific `FORWARD` behavior unchanged and
maps StreamVLN left/right ids to the 15-degree `TURN_*_SMALL` actions.
