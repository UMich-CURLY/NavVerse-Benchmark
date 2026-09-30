# LongNav-R1 evaluation

This adapter evaluates the original discrete LongNav-R1 policy from
[UMich-CURLY/LongNav-R1](https://github.com/UMich-CURLY/LongNav-R1), pinned to
`927b4a4c972e4cce5ea7c4c8cb0fd12a5ce4cbf0`.

```bash
bash baselines/longnav_r1/setup.sh
navverse-run benchmark --preset benchmark_longnav --split-set benchmark450
```

Use Python 3.11 for setup (`LONGNAV_SETUP_PYTHON=python3.11` when necessary).
The script creates an isolated `.venv` next to the adapter and downloads the
public source into `upstream`. Both are installation artifacts excluded from
source distributions. Repeating setup preserves local upstream modifications
by stopping rather than resetting them. Habitat and training submodules are
not needed by the inference bridge.

The default weights are `Aasdfip/hm3d_rpp_ke_standard-checkpoint_231` and the
base model referenced by that adapter. Model files download to the normal
Hugging Face cache; optimizer and scheduler states are excluded. The upstream
model's original base checkpoint name is retained.

For a standalone server:

```bash
baselines/longnav_r1/.venv/bin/python baselines/longnav_r1/serve_longnav.py \
  --host 127.0.0.1 --port 8000
```

`POST /act` accepts a multipart image named `file` and optional
`instr_or_goal`. Supplying a nonempty instruction resets episode state. The response contains `action_id` and `action_probs` in
the upstream order STOP, FORWARD, TURN_LEFT, TURN_RIGHT. `/health` and
`POST /reset` support managed clients. Actions use upstream probability
sampling; no STOP probability threshold is imposed by default.

The launcher starts one server per evaluation worker. Override its Python
with `--longnav_server_python` or `LONGNAV_SERVER_PYTHON`, or pass
`--longnav_server_url` for an external server. Each worker must have its own
stateful server. This release provides evaluation only.

The upstream package metadata declares MIT; consult the upstream repository
for its license terms.
