# Dummy

The Dummy agent runs deterministic movement sequences to check action
directions and trajectory recording. It uses the NavVerse environment and
requires no model weights.

```bash
python -m navverse.launcher.cli setup --env-name navverse
navverse-run benchmark --preset benchmark_dummy
```
