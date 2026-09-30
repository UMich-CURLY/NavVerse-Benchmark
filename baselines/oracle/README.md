# Reference-path oracles

Oracle follows the reference waypoints stored in each episode. The discrete
variant uses the same reference path but emits the public discrete actions.
Neither variant requires model weights.

```bash
python -m navverse.launcher.cli setup --env-name navverse
navverse-run benchmark --preset benchmark_oracle_gt
navverse-run benchmark --preset benchmark_oracle_gt_discrete
```
