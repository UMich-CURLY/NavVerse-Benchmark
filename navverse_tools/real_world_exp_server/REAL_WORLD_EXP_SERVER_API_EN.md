# NavVerse Real-World Experiment Server API (English)

This is the interface contract used by:
- Public browser UI: `https://your-site.example`
- Optional direct requests to your relay site
- Local bridge process `navverse_tools/real_world_exp_server/sites_bridge.py`

All public paths are relative to the site base URL (for example `/api/episodes`) unless stated otherwise.

## 1) Health

`GET /api/health`

Response:

```json
{
  "ok": true,
  "auth_required": false,
  "secure_context": true,
  "relay": true
}
```

## 2) List online local agents

`GET /api/agents`

Only agents currently heartbeating in the last 60 seconds are returned.

The current selectable model IDs are `uninavid`, `longnav`,
`internvla-n1`, and `poliformer`. `longnav` uses the original LongNav-R1 checkpoint.
Selecting `poliformer` shows an unverified-baseline warning.
Managed models start automatically on physical GPU 0 when an episode is
created.

Response:

```json
{
  "agents": [
    {
      "id": "internvla-n1",
      "label": "InternVLA-N1 DualVLN",
      "description": "RGB discrete-action policy.",
      "capabilities": {
        "state": "ready",
        "observation": "rgb",
        "selectable": true
      },
      "updated_at": "2026-08-07T07:00:00Z"
    }
  ]
}
```

## 3) List episodes

`GET /api/episodes`

Returns the latest 50 episodes:

```json
{
  "episodes": [
    {
      "episode_id": "phone_20260807",
      "instruction": "go to the red chair and stop",
      "agent": {"id": "internvla-n1", "label": "InternVLA-N1 DualVLN"},
      "status": "active",
      "started_at": "2026-08-07T07:00:00Z",
      "updated_at": "2026-08-07T07:00:05Z",
      "num_steps": 3,
      "termination": null
    }
  ]
}
```

## 4) Create episode

`POST /api/episodes`

Headers:
- Optional: `Authorization: Bearer <token>` (if your deployment requires auth).

Request body:

```json
{
  "episode_id": "optional_phone_episode_id",
  "agent_id": "internvla-n1",
  "instruction": "go to the red chair and stop",
  "hfov": 72,
  "client_metadata": {
    "user_agent": "...",
    "platform": "...",
    "screen": {"width": 0, "height": 0, "pixel_ratio": 1.0},
    "capture_mode": "web_rgb_imu | arkit_rgbd_pose",
    "native_capabilities": {}
  }
}
```

Validation:
- `instruction` is required and non-empty.
- `agent_id` must match regex `^[a-z0-9][a-z0-9._-]{0,63}$`.
- `hfov` must be 1–179.
- The selected agent must be online; otherwise you get `409 selected local agent is offline`.
- Only one active episode is allowed at a time.

Success (`201`):

```json
{
  "episode": {
    "episode_id": "phone_20260807_0001",
    "instruction": "go to the red chair and stop",
    "agent": {"id": "internvla-n1", "label": "InternVLA-N1 DualVLN"},
    "status": "active",
    "hfov": 72,
    "steps": []
  }
}
```

Important behavior:
- `POST /api/episodes` creates a **warmup job** in the `jobs` table.
- The local bridge polls `/api/bridge/jobs/next`, receives that warmup job, and calls `ensure_ready` on the selected local agent.
- So yes: creating an episode **does start/ensure the selected agent automatically**.

## 5) Send one step (photo + optional ARKit data)

`POST /api/episodes/{episode_id}/step`

Request for normal web RGB:

```json
{
  "request_id": "req-0001",
  "captured_at": "2026-08-07T07:00:11Z",
  "image": "data:image/jpeg;base64,/9j/...",
  "sensor": {
    "orientation": {"heading_deg": 12.5, "alpha": 0, "beta": 0, "gamma": 0},
    "motion": {},
    "imu_odometry": {}
  }
}
```

Request for ARKit RGB-D:

```json
{
  "request_id": "req-0002",
  "captured_at": "2026-08-07T07:00:22Z",
  "image": "data:image/jpeg;base64,/9j/...",
  "attachments": {
    "depth": "data:image/png;base64,iVBORw0K...",
    "confidence": "data:image/png;base64,iVBORw0K..."
  },
  "sensor": {
    "arkit": {
      "assets": {"depth": true, "confidence": true},
      "...": "full ARKit metadata"
    },
    "orientation": {},
    "motion": {},
    "imu_odometry": {}
  }
}
```

Validation:
- `request_id` is required and max 160 chars.
- `image` must be a JPEG (base64 data URL or raw base64).
- `depth`/`confidence` must be PNG if provided, and ARKit metadata is required when either attachment is provided.

Success (`202`):

```json
{ "pending": true, "pending_job_id": "req-0001" }
```

If the same `request_id` was already submitted, the response returns its current status directly:
- `200` with `{"step": {...}}` if already complete.
- `202` with `{"pending": true,...}` if still pending.

## 6) Poll step result

`GET /api/steps/{request_id}`

Response:
- `200`:

```json
{
  "step": {
    "step_index": 0,
    "request_id": "req-0001",
    "captured_at": "2026-08-07T07:00:11Z",
    "received_at": "2026-08-07T07:00:11Z",
    "sensor": {},
    "action": "MOVE_FORWARD",
    "label": "Move forward",
    "target_delta_deg": 0,
    "distance_m": 0.25,
    "latency_ms": 120,
    "agent_response": {}
  }
}
```

- `202` while waiting:

```json
{ "pending": true, "status": "pending" } // or "processing"
```

## 7) Finish episode

`POST /api/episodes/{episode_id}/finish`

Request:

```json
{
  "reason": "success | collision | stuck | timeout | agent_stop | user_abort",
  "notes": "optional note",
  "client_time": "2026-08-07T07:01:00Z",
  "sensor": {
    "imu_odometry": {"available": true, "path_distance_m": 2.3}
  }
}
```

Response:
- `200` with the updated `episode` object.
- Bridge creates a finish job and handles agent reset/shutdown timing.

## Internal bridge-only endpoints (for `navverse_tools/real_world_exp_server/sites_bridge.py`)

These require a valid bridge token header:
`Authorization: Bearer <BRIDGE_TOKEN>`

- `GET /api/bridge/jobs/next?agent_id=<id>[&agent_id=<id>...]`
- `GET /api/bridge/jobs/{jobId}/image`
- `GET /api/bridge/jobs/{jobId}/asset/{depth|confidence}`
- `POST /api/bridge/jobs/{jobId}/result`

`jobs/next` returns at most one pending/expired processing job and marks it as processing.

`result` payload:
- For step jobs: `{ "action": "TURN_LEFT", "label": "...", ... }`
- On failure: `{ "error": "message" }`

## Quick cURL sequence

1. `GET /api/agents`
2. `POST /api/episodes`
3. `POST /api/episodes/{episode_id}/step`
4. `GET /api/steps/{request_id}` (repeat until step appears)
5. `POST /api/episodes/{episode_id}/finish`
