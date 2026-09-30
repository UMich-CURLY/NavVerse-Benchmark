# NavVerse Real-World Experiment Server

`navverse_tools.real_world_exp_server` turns a phone into a human-executed navigation robot. The phone captures an RGB observation, the server asks a navigation agent for the next action, and the page guides a person to physically execute that action. The user confirms each completed action and explicitly reports the episode termination reason.

This is a practical MVP for real-world data collection and agent evaluation. It does not claim robot-grade odometry: browser IMU heading is useful for turn and direction feedback, but double-integrating phone acceleration is too drift-prone for reliable distance. The runner displays a drift-limited IMU distance estimate while the agent's forward distance remains a target confirmed by the human executor.

## Quick start and stop

Run the controller from the repository root:

```bash
navverse_tools/real_world_exp_server/serverctl
```

It opens a single-key terminal menu:

```text
[s] Start       [x] Stop        [r] Restart
[t] Status      [l] Logs        [i] Install/update
[q] Quit
```

No Enter key is needed after a menu choice. `Start` launches only the lightweight Sites bridge. A navigation model is loaded on GPU 0 after the phone starts an episode and selects that model. `Stop` shuts down the bridge and all model processes managed by its systemd service.

The same controller supports non-interactive commands:

```bash
navverse_tools/real_world_exp_server/serverctl start
navverse_tools/real_world_exp_server/serverctl status
navverse_tools/real_world_exp_server/serverctl logs
navverse_tools/real_world_exp_server/serverctl stop
```

On the first `start`, the controller installs the systemd user service automatically. It starts the service for the current session but does not enable automatic startup at login. `install` only installs or updates the unit; it does not start it. The older standalone installer remains available and preserves its original install-and-enable behavior:

```bash
bash navverse_tools/real_world_exp_server/install_bridge_service.sh
```

Bridge connection values live in the ignored `.sites-bridge.env` file. At minimum, a Sites deployment uses:

```dotenv
REAL_WORLD_EXP_SERVER_SITE_URL=https://your-site.example
REAL_WORLD_EXP_SERVER_BRIDGE_TOKEN=replace-with-the-private-bridge-token
REAL_WORLD_EXP_SERVER_SITES_BYPASS_TOKEN=replace-if-the-site-requires-one
```

Do not commit this file. Completed episodes and videos are saved under `runs/<episode_id>/`; bridge and model logs are saved under `logs/` and are also available through `serverctl logs`.

## What is included

- Mobile-first camera and IMU webpage with iOS sensor permission handling.
- Optional SwiftUI ARKit host that keeps the Sites UI while capturing synchronized RGB-D, intrinsics, and camera pose.
- Landscape-only execution view with screen-angle-corrected heading and drift-limited IMU acceleration double integration.
- First-screen selection of the local agent profiles currently registered by the bridge.
- A fixed-height runner that keeps camera, guidance, and termination controls in one viewport.
- Discrete `STOP`, `MOVE_FORWARD`, `MOVE_BACKWARD`, `TURN_LEFT`, and `TURN_RIGHT` guidance.
- Local-trajectory guidance for agents returning `{"trajectory": [[x, y], ...]}`.
- Human termination buttons for success, collision, stuck, timeout, agent stop, and abort.
- Incremental, atomic episode persistence with every JPEG frame, sensor snapshot, raw agent response, action, and final `result.json`.
- Automatic NavVerse-style `episode.mp4` replay with RGB, action, probabilities, phone heading, latency, instruction, and termination result.
- Idempotent step requests, reload recovery, optional bearer-token access control, and one active episode by default for stateful agents.
- No runtime Python dependencies outside the standard library.

## ARKit RGB-D host

`navverse_tools/real_world_exp_server/ios/` contains an iOS 17+ SwiftUI host generated with XcodeGen. It runs one `ARSession` and opens the same Sites page in `WKWebView`. When the page requests a frame, the native bridge encodes one synchronized `ARFrame` as:

```text
RGB JPEG                    sensor-native orientation
depth PNG                   UInt16 millimeters, scale_m = 0.001
confidence PNG              UInt8 ARConfidenceLevel values
intrinsics JSON             K_rgb, scaled K_depth, both resolutions
pose JSON                   ARKit world_from_camera 4x4 and timestamp
```

The webpage performs the authenticated upload, so native code never copies the private Sites cookies. Sites keeps RGB/depth/confidence in R2 until the local bridge claims the job; the bridge then saves them as `frames/NNNN.jpg`, `frames/NNNN.depth.png`, `frames/NNNN.confidence.png`, `frames/NNNN.intrinsics.json`, and `frames/NNNN.pose.json` before deleting the relay blobs.

ARKit scene depth is enabled only when `ARWorldTrackingConfiguration.supportsFrameSemantics` reports support. Non-LiDAR devices still provide RGB, intrinsics, and camera pose. See `navverse_tools/real_world_exp_server/ios/README.md` for project generation and device build instructions.

## Quick local demo

From the repository root:

```bash
python -m navverse_tools.real_world_exp_server.server --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. Without `--agent-url`, the server runs the deterministic demo sequence configured by `--demo-actions`.

## Use from a phone

Camera and motion APIs are restricted to secure browser contexts. `localhost` works on the same machine, but a phone opening a server by LAN IP normally needs trusted HTTPS. Place a trusted certificate and private key outside version control (the ignored `navverse_tools/real_world_exp_server/certs/` directory is available), then run:

```bash
export REAL_WORLD_EXP_SERVER_TOKEN='choose-a-long-random-token'
python -m navverse_tools.real_world_exp_server.server \
  --host 0.0.0.0 \
  --port 8765 \
  --certfile navverse_tools/real_world_exp_server/certs/server.pem \
  --keyfile navverse_tools/real_world_exp_server/certs/server-key.pem
```

Open `https://<server-hostname>:8765` on the phone and enter the same token. The certificate must be trusted by the phone; merely clicking through an untrusted-certificate warning may not make the page a secure context. A local CA tool such as `mkcert`, a trusted reverse proxy, or an existing HTTPS ingress can provide the certificate. Do not expose an unauthenticated instance to an untrusted network.

### Sites frontend with a local model server

The deployed Sites application acts as a relay, so the workstation only makes outbound HTTPS requests and does not expose a listening port. This deployment writes the private connection values to the ignored `navverse_tools/real_world_exp_server/.sites-bridge.env` file, which the bridge loads automatically. Install the lightweight bridge as a user service once:

```bash
bash navverse_tools/real_world_exp_server/install_bridge_service.sh
```

The first page then lists the configured profiles. Starting an episode creates a warmup job; the bridge launches the selected model, waits for `/health`, and handles the first observation after the model is ready. Every managed model child is forced to physical GPU 0 with `CUDA_VISIBLE_DEVICES=0`; an already occupied model port is rejected because its GPU assignment cannot be verified.

After an episode finishes, its model remains warm for 15 idle minutes by default. A new episode using the same model cancels and renews that timer; selecting a different model stops the previous model before launching the new one. Override the grace period with `--model-idle-timeout` in seconds.

Finishing an episode also resets the selected agent's recurrent/history state immediately while keeping its model weights resident during the idle grace period. The phone stores both net horizontal displacement and accumulated horizontal path distance under each sensor snapshot's `imu_odometry`, including the final estimate in `result.json`. It uses gravity-free acceleration when available and otherwise rotates the gravity-including vector into the world frame before removing standard gravity. These are deliberately labeled estimates: browser IMU double integration drifts and is not a substitute for metric odometry.

The default profiles are:

- Uni-NaVid in the `uninavid` environment on port 8300.
- LongNav-R1 in `baselines/longnav_r1/.venv` on port 8000; run `bash baselines/longnav_r1/setup.sh` first.
- InternVLA-N1 in the `internvla_n1` environment on port 8100.
- PoliFormer in the `poliformer` environment on port 8400. It is currently unverified and selection displays a warning.
- Demo mode without a model process.
- SG-VLN is shown but disabled for the current observation contract. Its mapping and planning stack requires synchronized metric depth plus camera/body pose; RGB and phone IMU alone are not a faithful input substitute.

The phone adapter exposes InternVLA-N1 as discrete actions only. Native action
sequences are returned directly; System 1 trajectories are discretized into
15-degree turns and 0.25-meter forward steps before leaving the local server.

To run the bridge interactively instead of as a service:

```bash
conda run -n navverse python -m navverse_tools.real_world_exp_server.sites_bridge
```

To expose only one custom action server, override the profile explicitly:

```bash
python -m navverse_tools.real_world_exp_server.sites_bridge \
  --agent-id my-agent \
  --agent-label "My local agent" \
  --agent-url http://127.0.0.1:9000/act \
  --agent-protocol navverse
```

The phone uploads one JPEG to the private Sites relay. The local bridge claims the job, downloads and saves the frame under `navverse_tools/real_world_exp_server/runs/`, calls the local model, and returns the action. Sites deletes the relayed frame immediately after accepting the action. D1 retains episode and action metadata so the phone can recover after a reload.

The page still supports direct HTTPS-server mode for development: enter a server URL in the optional field and launch `navverse_tools.real_world_exp_server.server` with an exact `--cors-origin` matching the Sites deployment.

## Connect a NavVerse action server

The default `navverse` protocol sends the phone JPEG as multipart field `rgb`, plus `instruction`, `reset`, `hfov`, and `depth_scale_m`. It accepts either `action_id` or `trajectory` in the JSON response:

```bash
python -m navverse_tools.real_world_exp_server.server \
  --agent-url http://127.0.0.1:8200/act \
  --agent-protocol navverse \
  --action-map STOP,FORWARD,TURN_LEFT_SMALL,TURN_RIGHT_SMALL
```

This matches the RGB-only action servers used by StreamVLN and the newer `HTTPActionServerAgent` interface. The official `InternRobotics/InternVLA-N1-DualVLN` weights use RGB observations. NavDP-based InternVLA-N1 variants remain RGB-D.

The legacy LongNav endpoint uses multipart field `file` and `instr_or_goal`:

```bash
python -m navverse_tools.real_world_exp_server.server \
  --agent-url http://127.0.0.1:8000/act \
  --agent-protocol longnav
```

For a custom server, use `--agent-protocol json`. The request is:

```json
{
  "image_base64": "...",
  "instruction": "Find the red chair",
  "reset": true,
  "hfov": 79.0
}
```

Supported responses are:

```json
{"action_id": 1}
```

```json
{"action": "MOVE_FORWARD"}
```

```json
{"trajectory": [[0.0, 0.0], [0.8, -0.2]]}
```

The trajectory convention matches the repository adapter: `x` is forward and `y` is lateral, so `atan2(y, x)` becomes the relative phone guidance angle.

## Saved results

The default output root is `navverse_tools/real_world_exp_server/runs/`:

```text
navverse_tools/real_world_exp_server/runs/<episode_id>/
├── episode.json
├── episode.mp4
├── result.json
└── frames/
    ├── 0000.jpg
    ├── 0000.depth.png
    ├── 0000.confidence.png
    ├── 0000.intrinsics.json
    ├── 0000.pose.json
    └── 0001.jpg
```

`episode.json` is rewritten atomically after every accepted step, so an interrupted run retains completed observations. `result.json` is written only after a termination button is confirmed.

## Validation

```bash
python -m unittest discover -s navverse_tools/real_world_exp_server/tests -v
python -m compileall -q navverse_tools.real_world_exp_server
```

For an actual phone smoke test, verify camera preview, a changing heading, left/right alignment feedback, one retry with the same request ID, each desired termination button, and the corresponding frame/JSON output on the server.
