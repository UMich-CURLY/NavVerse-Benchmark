# GPT navigation through Codex

This backend uses the official OpenAI Codex Python SDK and the local Codex
login. It runs the model through the Codex runtime, so record the method as
**the selected GPT model via Codex** rather than as a raw Responses API call. Each episode
uses one ephemeral, read-only thread with approval requests denied and a strict
four-action output schema. Every action turn appends its RGB observation while
retaining the episode history. Near the configured context threshold, the policy
first creates a navigation summary, compacts the thread, then re-injects that
summary with the next observation. The prompt, parser, image encoding, and audit
fields are shared across the supported simulators. Every action prompt includes the
current policy step and the evaluator's maximum step count. It defines success
from the episode goal, uses SPL to favor shorter successful routes, searches the
most likely target areas first, and widens the search after those areas fail.

## Install and sign in

Use a small CPU-only environment for the SDK server. The NavVerse simulator and
model environments do not need the SDK installed:

```bash
python -m venv "$HOME/.venvs/navverse-gpt"
"$HOME/.venvs/navverse-gpt/bin/pip" install -r baselines/gpt/requirements.txt
export GPT_SERVER_PYTHON="$HOME/.venvs/navverse-gpt/bin/python"
```

Keep the experiment account separate from the Codex account used by your
normal shell. The helper uses its own `CODEX_HOME` (default
`~/.codex-navverse-astra`) and file-based auth cache; it never reads or edits
`~/.codex/auth.json`:

```bash
export GPT_CODEX_HOME="$HOME/.codex-navverse-astra"
baselines/gpt/with_codex_home.sh codex login --device-auth
baselines/gpt/with_codex_home.sh codex login status
```

Complete the device login in a private browser window or a browser profile
signed into the intended second account. The separate home isolates Codex
configuration, credentials, sessions, and logs. Do not copy either account's
credentials between homes or print them.

Use a model available to the account and workspace signed into this isolated home.

## NavVerse

Use the included preset and select the NavVerse episode labels to evaluate:

```bash
GPT_SERVER_PYTHON="$GPT_SERVER_PYTHON" \
  baselines/gpt/with_codex_home.sh \
  conda run --no-capture-output -n navverse navverse-run benchmark \
  --preset benchmark_gpt --episode_labels grCommercial_scene2_2 \
  --gpt_max_steps 500 --gpu_ids 5 --num_gpu 1 --no-wandb
```

The preset caps each episode at 500 policy steps and each action at 5 seconds
of simulation time, with 2 cm position and 1 degree yaw tolerances. For a one-call simulator
smoke, add `--gpt_max_steps 1`; that confirms RGB transport and action
execution but does not measure navigation success.

## Implementation

- `policy.py`: shared model invocation, observation history, structured action
  parsing, summarization/compaction and token accounting.
- `gpt_agent.py` and `serve_gpt.py`: NavVerse agent and HTTP server.
- `with_codex_home.sh`: isolated account configuration.

Select `--agent GPT` or the `benchmark_gpt` preset through the standard
NavVerse launcher. Select another available GPT model with `--gpt_model`.

## Optional HM3D support

The same policy also supports HM3D through the Habitat PE adapter on develop.
See the [HM3D instructions](https://github.com/UMich-CURLY/NavVerse-Benchmark-Dev/blob/develop/baselines/gpt/habitat.md).
Habitat PE integration is not included in the NavVerse release.
