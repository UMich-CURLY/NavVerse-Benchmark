from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import mimetypes
import os
import ssl
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from navverse_tools.real_world_exp_server.agent import AgentError, DemoAgent, GuidanceConfig, HTTPAgent
from navverse_tools.real_world_exp_server.store import EpisodeStore, StoreError


STATIC_ROOT = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 16 * 1024 * 1024


class RealWorldExperimentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, store, agent, token, cors_origins):
        super().__init__(address, handler)
        self.store = store
        self.agent = agent
        self.token = token
        self.cors_origins = frozenset(cors_origins)
        self.step_lock = threading.Lock()


class RequestHandler(BaseHTTPRequestHandler):
    server: RealWorldExperimentServer
    protocol_version = "HTTP/1.1"

    def do_OPTIONS(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/api/") or not self._cors_origin():
            self._error(HTTPStatus.FORBIDDEN, "Cross-origin request is not allowed")
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self._security_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "auth_required": bool(self.server.token),
                    "secure_context": isinstance(self.server.socket, ssl.SSLSocket),
                },
            )
            return
        if path.startswith("/api/"):
            if not self._authorized():
                return
            if path == "/api/episodes":
                self._json(HTTPStatus.OK, {"episodes": self.server.store.list()})
                return
            prefix = "/api/episodes/"
            if path.startswith(prefix):
                try:
                    episode = self.server.store.get(unquote(path[len(prefix) :]))
                    self._json(HTTPStatus.OK, {"episode": episode})
                except StoreError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                return
            self._error(HTTPStatus.NOT_FOUND, "API route not found")
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            self._error(HTTPStatus.NOT_FOUND, "Route not found")
            return
        if not self._authorized():
            return
        try:
            payload = self._read_json()
            if path == "/api/episodes":
                self._start_episode(payload)
                return
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "episodes"]:
                episode_id, operation = unquote(parts[2]), parts[3]
                if operation == "step":
                    self._step(episode_id, payload)
                    return
                if operation == "finish":
                    self._finish(episode_id, payload)
                    return
            self._error(HTTPStatus.NOT_FOUND, "API route not found")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except StoreError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except AgentError as exc:
            self._error(HTTPStatus.BAD_GATEWAY, str(exc))
        except Exception as exc:
            self.log_error("Unhandled request error: %r", exc)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error")

    def _start_episode(self, payload: dict[str, Any]) -> None:
        try:
            hfov = float(payload.get("hfov", 79.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("hfov must be a number") from exc
        if not 1.0 <= hfov <= 179.0:
            raise ValueError("hfov must be between 1 and 179 degrees")
        episode = self.server.store.start(
            episode_id=str(payload.get("episode_id", "")),
            instruction=str(payload.get("instruction", "")),
            hfov=hfov,
            client_metadata=self._object(payload.get("client_metadata")),
        )
        self._json(HTTPStatus.CREATED, {"episode": episode})

    def _step(self, episode_id: str, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "")).strip()
        if not request_id or len(request_id) > 160:
            raise ValueError("request_id is required and must be at most 160 characters")
        with self.server.step_lock:
            episode = self.server.store.get(episode_id)
            for existing in episode["steps"]:
                if existing["request_id"] == request_id:
                    self._json(HTTPStatus.OK, {"step": existing, "episode": episode})
                    return
            image_bytes = self._decode_image(payload.get("image"))
            sensor = self._object(payload.get("sensor"))
            attachments = self._object(payload.get("attachments"))
            observation_assets = {
                name: self._decode_png(value)
                for name, value in attachments.items()
                if name in {"depth", "confidence"}
            }
            observation = sensor.get("arkit") if isinstance(sensor.get("arkit"), dict) else None
            action = self.server.agent.act(
                image_bytes=image_bytes,
                instruction=episode["instruction"],
                episode_id=episode["episode_id"],
                reset=not episode["steps"],
                hfov=episode["hfov"],
                observation=observation,
                depth_bytes=observation_assets.get("depth"),
                confidence_bytes=observation_assets.get("confidence"),
            )
            step = self.server.store.add_step(
                episode_id=episode_id,
                request_id=request_id,
                image_bytes=image_bytes,
                captured_at=self._optional_string(payload.get("captured_at"), 120),
                sensor=sensor,
                action=action,
                observation_assets=observation_assets,
            )
        self._json(HTTPStatus.CREATED, {"step": step})

    def _finish(self, episode_id: str, payload: dict[str, Any]) -> None:
        episode = self.server.store.finish(
            episode_id=episode_id,
            reason=str(payload.get("reason", "")),
            notes=str(payload.get("notes", "")),
            sensor=self._object(payload.get("sensor")),
            client_time=self._optional_string(payload.get("client_time"), 120),
        )
        self._json(HTTPStatus.OK, {"episode": episode})

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError(f"Request body must be between 1 and {MAX_BODY_BYTES} bytes")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object")
        return value

    @staticmethod
    def _decode_image(value: Any) -> bytes:
        if not isinstance(value, str):
            raise ValueError("image must be a base64 string or data URL")
        encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        try:
            image = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image is not valid base64") from exc
        if not image or len(image) > 12 * 1024 * 1024:
            raise ValueError("decoded image must be between 1 byte and 12 MiB")
        if not image.startswith(b"\xff\xd8"):
            raise ValueError("image must be JPEG")
        return image

    @staticmethod
    def _decode_png(value: Any) -> bytes:
        if not isinstance(value, str):
            raise ValueError("observation attachment must be a base64 string or data URL")
        encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        try:
            image = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("observation attachment is not valid base64") from exc
        if not image or len(image) > 16 * 1024 * 1024:
            raise ValueError("observation attachment must be between 1 byte and 16 MiB")
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("observation attachment must be PNG")
        return image

    def _authorized(self) -> bool:
        if not self.server.token:
            return True
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.token}"
        if hmac.compare_digest(supplied, expected):
            return True
        self._json(
            HTTPStatus.UNAUTHORIZED,
            {"error": "A valid bearer token is required"},
            extra_headers={"WWW-Authenticate": "Bearer"},
        )
        return False

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else unquote(path.lstrip("/"))
        target = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT not in target.parents or not target.is_file():
            target = STATIC_ROOT / "index.html" if "." not in Path(relative).name else None
        if target is None or not target.is_file():
            self._error(HTTPStatus.NOT_FOUND, "Static asset not found")
            return
        content = target.read_bytes()
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self._security_headers()
        self.end_headers()
        self.wfile.write(content)

    def _json(
        self,
        status: HTTPStatus,
        value: Any,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Permissions-Policy",
            "camera=(self), accelerometer=(self), gyroscope=(self), magnetometer=(self)",
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
            "connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'",
        )
        cors_origin = self._cors_origin()
        if cors_origin:
            self.send_header("Access-Control-Allow-Origin", cors_origin)
            self.send_header("Vary", "Origin")

    def _cors_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        if not origin:
            return None
        if "*" in self.server.cors_origins or origin in self.server.cors_origins:
            return origin
        return None

    @staticmethod
    def _object(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("sensor/client_metadata must be a JSON object")
        return value

    @staticmethod
    def _optional_string(value: Any, limit: int) -> str | None:
        if value is None:
            return None
        return str(value)[:limit]

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NavVerse real-world experiment server web server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", type=Path, default=Path("navverse_tools/real_world_exp_server/runs"))
    parser.add_argument("--token", default=os.environ.get("REAL_WORLD_EXP_SERVER_TOKEN", ""))
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="Allowed browser origin; repeat as needed, or use '*' with a required token",
    )
    parser.add_argument("--certfile", type=Path)
    parser.add_argument("--keyfile", type=Path)
    parser.add_argument("--agent-url", default="")
    parser.add_argument(
        "--agent-protocol", choices=("navverse", "longnav", "json"), default="navverse"
    )
    parser.add_argument(
        "--action-map", default="STOP,MOVE_FORWARD,TURN_LEFT,TURN_RIGHT"
    )
    parser.add_argument("--agent-timeout", type=float, default=180.0)
    parser.add_argument("--forward-distance", type=float, default=0.25)
    parser.add_argument("--backward-distance", type=float, default=0.25)
    parser.add_argument("--turn-degrees", type=float, default=30.0)
    parser.add_argument(
        "--demo-actions",
        default="MOVE_FORWARD,TURN_LEFT,MOVE_FORWARD,TURN_RIGHT,STOP",
        help="Actions used when --agent-url is omitted",
    )
    parser.add_argument("--allow-concurrent", action="store_true")
    return parser.parse_args(argv)


def build_server(args: argparse.Namespace) -> RealWorldExperimentServer:
    if bool(args.certfile) != bool(args.keyfile):
        raise ValueError("--certfile and --keyfile must be provided together")
    guidance = GuidanceConfig(
        forward_distance_m=args.forward_distance,
        backward_distance_m=args.backward_distance,
        turn_degrees=args.turn_degrees,
    )
    action_map = [item.strip() for item in args.action_map.split(",") if item.strip()]
    if not action_map:
        raise ValueError("--action-map must not be empty")
    if args.agent_url:
        agent = HTTPAgent(
            url=args.agent_url,
            protocol=args.agent_protocol,
            action_map=action_map,
            timeout_s=args.agent_timeout,
            guidance=guidance,
        )
    else:
        demo_actions = [item.strip() for item in args.demo_actions.split(",") if item.strip()]
        agent = DemoAgent(demo_actions, guidance)
    store = EpisodeStore(args.output_dir, allow_concurrent=args.allow_concurrent)
    server = RealWorldExperimentServer(
        (args.host, args.port),
        RequestHandler,
        store=store,
        agent=agent,
        token=args.token,
        cors_origins=args.cors_origin,
    )
    if args.certfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.certfile, args.keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    server = build_server(args)
    scheme = "https" if args.certfile else "http"
    mode = f"agent={args.agent_url}" if args.agent_url else "demo agent"
    print(f"Real-world experiment server: {scheme}://{args.host}:{server.server_port} ({mode})")
    print(f"Episode output: {args.output_dir.resolve()}")
    if args.host not in {"127.0.0.1", "localhost"} and not args.certfile:
        print("Warning: phone camera/IMU usually require HTTPS; see navverse_tools/real_world_exp_server/README.md")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
