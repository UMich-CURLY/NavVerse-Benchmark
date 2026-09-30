from __future__ import annotations

import argparse
import atexit
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from navverse_tools.real_world_exp_server.agent import DemoAgent, GuidanceConfig, HTTPAgent
from navverse_tools.real_world_exp_server.managed_agent import BlockedAgent, LaunchSpec, ManagedHTTPAgent
from navverse_tools.real_world_exp_server.store import EpisodeStore, StoreError
from navverse_tools.real_world_exp_server.video import render_episode_video


@dataclass(frozen=True)
class AgentProfile:
    id: str
    label: str
    description: str
    capabilities: dict[str, Any]
    agent: Any

    def registration(self) -> dict[str, Any]:
        runtime = self.agent.status() if hasattr(self.agent, "status") else {
            "state": "ready",
            "detail": "Ready.",
            "auto_start": False,
            "selectable": True,
            "gpu": None,
        }
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "capabilities": {**self.capabilities, **runtime},
        }


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("REAL_WORLD_EXP_SERVER_"):
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


class SitesClient:
    def __init__(self, site_url: str, bridge_token: str, bypass_token: str, timeout_s: float):
        self.site_url = site_url.rstrip("/")
        self.bridge_token = bridge_token
        self.bypass_token = bypass_token
        self.timeout_s = timeout_s

    def register_agents(self, profiles: dict[str, AgentProfile]) -> None:
        self.request(
            "POST",
            "/api/bridge/agents",
            {
                "agents": [
                    profile.registration()
                    for profile in profiles.values()
                ]
            },
        )

    def register_runs(self, runs: list[dict[str, Any]]) -> None:
        self.request("POST", "/api/bridge/runs", {"runs": runs})

    def next_job(self, agent_ids: list[str]) -> dict[str, Any] | None:
        query = urllib.parse.urlencode({"agent_id": agent_ids}, doseq=True)
        status, payload = self.request("GET", f"/api/bridge/jobs/next?{query}")
        return None if status == 204 else payload["job"]

    def next_video_request(self) -> dict[str, Any] | None:
        status, payload = self.request("GET", "/api/bridge/videos/next")
        return None if status == 204 else payload["request"]

    def image(self, job_id: int) -> bytes:
        _status, payload = self.request("GET", f"/api/bridge/jobs/{job_id}/image", raw=True)
        return payload

    def asset(self, job_id: int, name: str) -> bytes:
        if name not in {"depth", "confidence"}:
            raise ValueError(f"Unsupported observation asset: {name}")
        _status, payload = self.request(
            "GET",
            f"/api/bridge/jobs/{job_id}/asset/{name}",
            raw=True,
        )
        return payload

    def complete(self, job_id: int, result: dict[str, Any]) -> None:
        self.request("POST", f"/api/bridge/jobs/{job_id}/result", result)

    def fail(self, job_id: int, error: str) -> None:
        self.request("POST", f"/api/bridge/jobs/{job_id}/result", {"error": error[:1000]})

    def fail_video_request(self, request_id: int, error: str) -> None:
        self.request(
            "POST",
            f"/api/bridge/videos/{request_id}/result",
            {"error": error[:1000]},
        )

    def upload_video(self, episode_id: str, video_path: Path) -> None:
        request = urllib.request.Request(
            self.site_url + f"/api/bridge/runs/{urllib.parse.quote(episode_id)}/video",
            data=video_path.read_bytes(),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.bridge_token}",
                "Content-Type": "video/mp4",
            },
            method="PUT",
        )
        if self.bypass_token:
            request.add_header("OAI-Sites-Authorization", f"Bearer {self.bypass_token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(f"Sites returned HTTP {exc.code}: {detail}") from exc

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        raw: bool = False,
    ) -> tuple[int, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/octet-stream" if raw else "application/json",
            "Authorization": f"Bearer {self.bridge_token}",
        }
        if self.bypass_token:
            headers["OAI-Sites-Authorization"] = f"Bearer {self.bypass_token}"
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.site_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
                if response.status == 204:
                    return response.status, None
                return response.status, body if raw else json.loads(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(f"Sites returned HTTP {exc.code}: {detail}") from exc


class Bridge:
    def __init__(
        self,
        client: SitesClient,
        store: EpisodeStore,
        agents: dict[str, AgentProfile],
        poll_interval_s: float,
        model_idle_timeout_s: float = 900.0,
    ):
        self.client = client
        self.store = store
        self.agents = agents
        self.poll_interval_s = poll_interval_s
        self.model_idle_timeout_s = model_idle_timeout_s
        self.last_registration = 0.0

    def run(self) -> None:
        labels = ", ".join(profile.label for profile in self.agents.values())
        print(f"Phone bridge connected to {self.client.site_url}; agents: {labels}", flush=True)
        while True:
            try:
                if time.monotonic() - self.last_registration >= 5.0:
                    self.publish_state()
                video_request = self.client.next_video_request()
                if video_request is not None:
                    self.process_video_request(video_request)
                    continue
                job = self.client.next_job(list(self.agents))
                if job is None:
                    time.sleep(self.poll_interval_s)
                    continue
                try:
                    self.process(job)
                except Exception as exc:
                    self.client.fail(int(job["id"]), str(exc))
                    print(f"Failed job={job['id']}: {exc}", flush=True)
            except KeyboardInterrupt:
                self.shutdown()
                return
            except Exception as exc:
                print(f"Bridge error: {exc}", flush=True)
                time.sleep(max(1.0, self.poll_interval_s))

    def publish_agents(self) -> None:
        self.client.register_agents(self.agents)

    def publish_state(self) -> None:
        self.publish_agents()
        self.publish_runs()
        self.last_registration = time.monotonic()

    def publish_runs(self) -> None:
        self.client.register_runs(self.local_runs())

    def shutdown(self) -> None:
        for profile in self.agents.values():
            if hasattr(profile.agent, "shutdown"):
                profile.agent.shutdown()

    def local_runs(self) -> list[dict[str, Any]]:
        by_episode: dict[str, dict[str, Any]] = {}
        for manifest in self.store.root.glob("*/episode.json"):
            try:
                episode = json.loads(manifest.read_text(encoding="utf-8"))
                episode_id = str(episode["episode_id"])
                agent = (episode.get("client_metadata") or {}).get("agent") or {}
                video_path = manifest.parent / "episode.mp4"
                summary = {
                    "episode_id": episode_id,
                    "instruction": str(episode.get("instruction") or ""),
                    "agent": {
                        "id": str(agent.get("id") or "unknown"),
                        "label": str(agent.get("label") or agent.get("id") or "Unknown agent"),
                    },
                    "status": str(episode.get("status") or "active"),
                    "started_at": episode.get("started_at"),
                    "updated_at": episode.get("updated_at"),
                    "num_steps": len(episode.get("steps") or []),
                    "termination": episode.get("termination"),
                    "video": {
                        "available": video_path.is_file(),
                        "size_bytes": video_path.stat().st_size if video_path.is_file() else None,
                    },
                }
                previous = by_episode.get(episode_id)
                if previous is None or str(summary["updated_at"]) > str(previous["updated_at"]):
                    by_episode[episode_id] = summary
            except (KeyError, OSError, json.JSONDecodeError, TypeError):
                continue
        return sorted(
            by_episode.values(),
            key=lambda run: str(run["updated_at"]),
            reverse=True,
        )[:100]

    def video_path(self, episode_id: str) -> Path | None:
        candidates: list[tuple[str, Path]] = []
        for manifest in self.store.root.glob("*/episode.json"):
            try:
                episode = json.loads(manifest.read_text(encoding="utf-8"))
                video_path = manifest.parent / "episode.mp4"
                if str(episode.get("episode_id")) == episode_id and video_path.is_file():
                    candidates.append((str(episode.get("updated_at") or ""), video_path))
            except (OSError, json.JSONDecodeError):
                continue
        return max(candidates, default=("", None), key=lambda item: item[0])[1]

    def process_video_request(self, request: dict[str, Any]) -> None:
        request_id = int(request["id"])
        episode_id = str(request["episode_id"])
        try:
            video_path = self.video_path(episode_id)
            if video_path is None:
                raise RuntimeError("local run video is unavailable")
            self.client.upload_video(episode_id, video_path)
            print(f"Streamed requested video episode={episode_id}", flush=True)
        except Exception as exc:
            self.client.fail_video_request(request_id, str(exc))
            print(f"Failed video request={request_id} episode={episode_id}: {exc}", flush=True)

    def process(self, job: dict[str, Any]) -> None:
        episode_id = str(job["episode_id"])
        profile = self.agents.get(str(job.get("agent_id", "")))
        if profile is None:
            raise RuntimeError(f"Unsupported agent profile: {job.get('agent_id')}")
        if job["kind"] == "warmup":
            if self._stop_other_agents(profile.id):
                self.publish_agents()
            if hasattr(profile.agent, "ensure_ready"):
                profile.agent.ensure_ready(self.publish_agents)
            self.client.complete(int(job["id"]), {"warmup_ack": True})
            self.publish_agents()
            print(f"Ready agent={profile.id} episode={episode_id}", flush=True)
            return

        episode = self._ensure_episode(job)
        if job["kind"] == "finish":
            termination = job.get("termination") or {}
            self.store.finish(
                episode_id=episode_id,
                reason=str(termination.get("reason", "user_abort")),
                notes=str(termination.get("notes", "")),
                sensor=termination.get("sensor") or {},
                client_time=termination.get("client_time"),
            )
            try:
                reset_applied = bool(profile.agent.reset()) if hasattr(profile.agent, "reset") else False
                self.store.update_result(episode_id, {"agent_reset": reset_applied})
                print(f"Reset agent={profile.id} episode={episode_id} applied={reset_applied}", flush=True)
            except Exception as exc:
                self.store.update_result(episode_id, {"agent_reset_error": str(exc)[:1000]})
                print(f"Failed reset agent={profile.id} episode={episode_id}: {exc}", flush=True)
            try:
                video = render_episode_video(self.store.root / episode_id)
                self.store.update_result(episode_id, {"video": video})
                print(f"Saved video episode={episode_id} path={video['path']}", flush=True)
            except Exception as exc:
                self.store.update_result(episode_id, {"video_error": str(exc)[:1000]})
                print(f"Failed video episode={episode_id}: {exc}", flush=True)
            self.client.complete(int(job["id"]), {"finish_ack": True})
            try:
                self.publish_runs()
            except Exception as exc:
                print(f"Deferred run index update episode={episode_id}: {exc}", flush=True)
            if hasattr(profile.agent, "schedule_idle_shutdown"):
                profile.agent.schedule_idle_shutdown(
                    self.model_idle_timeout_s,
                    self.publish_agents,
                )
                self.publish_agents()
            print(f"Finished episode={episode_id} reason={termination.get('reason')}", flush=True)
            return

        existing = next(
            (step for step in episode["steps"] if step["request_id"] == job["request_id"]),
            None,
        )
        if existing is not None:
            result = {
                key: value
                for key, value in existing.items()
                if key
                not in {
                    "step_index",
                    "request_id",
                    "captured_at",
                    "received_at",
                    "image",
                    "sensor",
                }
            }
        else:
            image = self.client.image(int(job["id"]))
            sensor = job.get("sensor") or {}
            observation = sensor.get("arkit") if isinstance(sensor, dict) else None
            asset_flags = observation.get("assets", {}) if isinstance(observation, dict) else {}
            observation_assets = {
                name: self.client.asset(int(job["id"]), name)
                for name in ("depth", "confidence")
                if asset_flags.get(name) is True
            }
            result = profile.agent.act(
                image_bytes=image,
                instruction=str(job["instruction"]),
                episode_id=episode_id,
                reset=not episode["steps"],
                hfov=float(job.get("hfov", 79.0)),
                observation=observation,
                depth_bytes=observation_assets.get("depth"),
                confidence_bytes=observation_assets.get("confidence"),
            )
            self.store.add_step(
                episode_id=episode_id,
                request_id=str(job["request_id"]),
                image_bytes=image,
                captured_at=job.get("captured_at"),
                sensor=sensor,
                action=result,
                observation_assets=observation_assets,
            )
        self.client.complete(int(job["id"]), result)
        print(f"Completed episode={episode_id} action={result.get('action')}", flush=True)

    def _stop_other_agents(self, selected_id: str) -> bool:
        stopped = False
        for profile_id, candidate in self.agents.items():
            if profile_id != selected_id and hasattr(candidate.agent, "stop"):
                stopped = candidate.agent.stop(
                    f"Stopped because the active agent switched to {selected_id}."
                ) or stopped
        return stopped

    def _ensure_episode(self, job: dict[str, Any]) -> dict[str, Any]:
        episode_id = str(job["episode_id"])
        try:
            return self.store.get(episode_id)
        except StoreError:
            return self.store.start(
                episode_id=episode_id,
                instruction=str(job["instruction"]),
                hfov=float(job.get("hfov", 79.0)),
                client_metadata={
                    **(job.get("client_metadata") or {}),
                    "agent": {
                        "id": job.get("agent_id"),
                        "label": job.get("agent_label"),
                    },
                },
            )


def load_agent_profiles(args: argparse.Namespace) -> dict[str, AgentProfile]:
    if args.agent_url:
        configs = [
            {
                "id": args.agent_id,
                "label": args.agent_label,
                "description": args.agent_description,
                "url": args.agent_url,
                "protocol": args.agent_protocol,
                "action_map": [item.strip() for item in args.action_map.split(",") if item.strip()],
            }
        ]
    elif args.agents_file.is_file():
        payload = json.loads(args.agents_file.read_text(encoding="utf-8"))
        configs = payload.get("agents", [])
    else:
        configs = [
            {
                "id": "demo",
                "label": "Demo sequence",
                "description": "Deterministic actions for checking the phone workflow.",
                "demo_actions": [
                    item.strip() for item in args.demo_actions.split(",") if item.strip()
                ],
            }
        ]

    profiles: dict[str, AgentProfile] = {}
    repo_root = Path(__file__).resolve().parents[2]
    for config in configs:
        profile_id = str(config.get("id", "")).strip()
        if not profile_id or profile_id in profiles:
            raise ValueError(f"Invalid or duplicate agent id: {profile_id!r}")
        guidance = GuidanceConfig(
            float(config.get("forward_distance", args.forward_distance)),
            float(config.get("backward_distance", args.backward_distance)),
            float(config.get("turn_degrees", args.turn_degrees)),
        )
        if config.get("blocked_reason"):
            agent = BlockedAgent(str(config["blocked_reason"]))
            capabilities = {
                "protocol": config.get("protocol", "navverse"),
                "observation": config.get("observation", "rgbd_pose"),
            }
        elif config.get("url"):
            http_agent = HTTPAgent(
                url=str(config["url"]),
                protocol=str(config.get("protocol", "navverse")),
                action_map=list(config.get("action_map") or ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]),
                timeout_s=args.agent_timeout,
                guidance=guidance,
                reset_url=str(config["reset_url"]) if config.get("reset_url") else None,
            )
            launch_config = config.get("launch")
            if launch_config:
                script = repo_root / str(launch_config["script"])
                launch = LaunchSpec(
                    env_name=str(launch_config.get("env_name", "")),
                    python=(repo_root / str(launch_config["python"])) if launch_config.get("python") else None,
                    script=script,
                    args=tuple(
                        os.path.expandvars(str(item))
                        for item in launch_config.get("args", [])
                    ),
                    health_url=str(launch_config["health_url"]),
                    startup_timeout_s=float(launch_config.get("startup_timeout_s", 900)),
                    environment={
                        str(key): str(value)
                        for key, value in (launch_config.get("environment") or {}).items()
                    },
                )
                agent = ManagedHTTPAgent(
                    profile_id=profile_id,
                    delegate=http_agent,
                    launch=launch,
                    repo_root=repo_root,
                    log_dir=args.log_dir,
                )
            else:
                agent = http_agent
            capabilities = {
                "protocol": config.get("protocol", "navverse"),
                "observation": config.get("observation", "rgb"),
                "endpoint": str(config["url"]),
            }
        else:
            demo_actions = list(config.get("demo_actions") or ["MOVE_FORWARD", "STOP"])
            agent = DemoAgent(demo_actions, guidance)
            capabilities = {"protocol": "demo", "observation": "rgb"}
        profiles[profile_id] = AgentProfile(
            id=profile_id,
            label=str(config.get("label") or profile_id),
            description=str(config.get("description") or "Local navigation agent"),
            capabilities=capabilities,
            agent=agent,
        )
    if not profiles:
        raise ValueError("No agent profiles are configured")
    return profiles


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outbound bridge from Sites to a local navigation agent")
    parser.add_argument("--site-url", default=os.environ.get("REAL_WORLD_EXP_SERVER_SITE_URL", ""))
    parser.add_argument("--bridge-token", default=os.environ.get("REAL_WORLD_EXP_SERVER_BRIDGE_TOKEN", ""))
    parser.add_argument(
        "--sites-bypass-token",
        default=os.environ.get("REAL_WORLD_EXP_SERVER_SITES_BYPASS_TOKEN", ""),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("navverse_tools/real_world_exp_server/runs"))
    parser.add_argument("--agents-file", type=Path, default=Path("navverse_tools/real_world_exp_server/agents.json"))
    parser.add_argument("--log-dir", type=Path, default=Path("navverse_tools/real_world_exp_server/logs"))
    parser.add_argument("--agent-url", default="")
    parser.add_argument("--agent-id", default="custom-local")
    parser.add_argument("--agent-label", default="Custom local agent")
    parser.add_argument("--agent-description", default="Agent supplied with --agent-url")
    parser.add_argument(
        "--agent-protocol", choices=("navverse", "longnav", "json"), default="navverse"
    )
    parser.add_argument("--action-map", default="STOP,MOVE_FORWARD,TURN_LEFT,TURN_RIGHT")
    parser.add_argument("--agent-timeout", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=0.75)
    parser.add_argument("--model-idle-timeout", type=float, default=900.0)
    parser.add_argument("--forward-distance", type=float, default=0.25)
    parser.add_argument("--backward-distance", type=float, default=0.25)
    parser.add_argument("--turn-degrees", type=float, default=30.0)
    parser.add_argument(
        "--demo-actions", default="MOVE_FORWARD,TURN_LEFT,MOVE_FORWARD,TURN_RIGHT,STOP"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    env_file = Path(os.environ.get("REAL_WORLD_EXP_SERVER_ENV_FILE", "navverse_tools/real_world_exp_server/.sites-bridge.env"))
    load_env_file(env_file)
    args = parse_args(argv)
    if not args.site_url or not args.bridge_token:
        raise SystemExit("--site-url and --bridge-token (or matching environment variables) are required")
    agents = load_agent_profiles(args)
    client = SitesClient(args.site_url, args.bridge_token, args.sites_bypass_token, args.agent_timeout)
    bridge = Bridge(
        client,
        EpisodeStore(args.output_dir, allow_concurrent=False),
        agents,
        args.poll_interval,
        args.model_idle_timeout,
    )
    atexit.register(bridge.shutdown)
    bridge.run()


if __name__ == "__main__":
    main()
