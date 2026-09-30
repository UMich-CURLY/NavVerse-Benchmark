# Benchmark baselines

`navverse-run benchmark` resolves a preset and episode split, then starts
simulation and agent workers. The simulator sends observations to the selected
adapter in `baselines/`; the adapter returns actions through the shared agent
interface. The evaluator records trajectories, metrics, and optional videos.
Model-serving adapters run their policy in a separate environment and exchange
observations and actions with the evaluation worker. Agent registrations are
defined in `navverse/launcher/registry.py`, and presets are in
`baselines/presets/`.

| Baseline | Overview |
| --- | --- |
| [Dummy](dummy/README.md) | Deterministic motion sequences for checking simulator actions and trajectory recording. |
| [Oracle](oracle/README.md) | Follows the episode reference path with waypoint control. |
| [OracleDiscreteAction](oracle/README.md) | Follows the reference path through the discrete action interface. |
| [UniNaVid](uninavid/README.md) | Video-based navigation policy using visual observations and task instructions. |
| [GPT](gpt/README.md) | GPT navigation through Codex with structured actions and step sizes. |
| [LongNav-R1](longnav_r1/README.md) | Original R1 evaluation policy with discrete actions. |
| [InternVLA-N1](internvla_n1/README.md) | Vision-language navigation policy for VLN tasks. |
| [StreamVLN](streamvln/README.md) | Streaming VLN policy with temporal context. |
| [PoliFormer](poliformer/README.md) | Object-navigation policy using visual and text inputs. |

Each linked README covers that adapter's installation, model files, and
benchmark command.
