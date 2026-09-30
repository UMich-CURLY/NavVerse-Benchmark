import base64
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from navverse_tools.real_world_exp_server.server import build_server, parse_args


JPEG = b"\xff\xd8phone-test\xff\xd9"


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        args = parse_args(
            [
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--output-dir",
                self.temp_dir.name,
                "--token",
                "test-token",
                "--cors-origin",
                "https://phone.example",
                "--demo-actions",
                "MOVE_FORWARD,STOP",
            ]
        )
        self.server = build_server(args)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp_dir.cleanup()

    def request(self, method, path, payload=None, token="test-token"):
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read())

    def test_health_does_not_require_token(self):
        status, body = self.request("GET", "/api/health", token="")
        self.assertEqual(status, 200)
        self.assertTrue(body["auth_required"])

    def test_mobile_app_is_served(self):
        with urllib.request.urlopen(self.base_url + "/", timeout=3) as response:
            page = response.read().decode()
        self.assertIn("NavVerse Real-World Experiment Server", page)
        self.assertIn('id="camera"', page)
        self.assertIn('id="serverUrl"', page)

    def test_configured_origin_gets_cors_headers(self):
        request = urllib.request.Request(
            self.base_url + "/api/health",
            headers={"Origin": "https://phone.example"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            self.assertEqual(
                response.headers["Access-Control-Allow-Origin"], "https://phone.example"
            )

        preflight = urllib.request.Request(
            self.base_url + "/api/episodes",
            headers={
                "Origin": "https://phone.example",
                "Access-Control-Request-Method": "POST",
            },
            method="OPTIONS",
        )
        with urllib.request.urlopen(preflight, timeout=3) as response:
            self.assertEqual(response.status, 204)
            self.assertIn("Authorization", response.headers["Access-Control-Allow-Headers"])

    def test_episode_round_trip_and_idempotent_step(self):
        status, body = self.request(
            "POST",
            "/api/episodes",
            {"episode_id": "test/episode", "instruction": "Find the chair", "hfov": 79},
        )
        self.assertEqual(status, 201)
        episode_id = body["episode"]["episode_id"]
        self.assertEqual(episode_id, "test_episode")
        step_payload = {
            "request_id": "request-1",
            "captured_at": "2026-08-05T00:00:00Z",
            "image": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode(),
            "sensor": {
                "orientation": {"heading_deg": 10},
                "arkit": {
                    "intrinsics": {"rgb": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                    "pose": {"camera_to_world": [[1, 0, 0, 0]]},
                },
            },
            "attachments": {
                "depth": "data:image/png;base64,"
                + base64.b64encode(b"\x89PNG\r\n\x1a\ndepth").decode(),
            },
        }
        first_status, first = self.request(
            "POST", f"/api/episodes/{episode_id}/step", step_payload
        )
        second_status, second = self.request(
            "POST", f"/api/episodes/{episode_id}/step", step_payload
        )
        self.assertEqual(first_status, 201)
        self.assertEqual(second_status, 200)
        self.assertEqual(first["step"], second["step"])
        self.assertEqual(first["step"]["action"], "MOVE_FORWARD")
        self.assertTrue(
            (Path(self.temp_dir.name) / episode_id / "frames" / "0000.depth.png").is_file()
        )

        status, finished = self.request(
            "POST",
            f"/api/episodes/{episode_id}/finish",
            {
                "reason": "success",
                "client_time": "2026-08-05T00:00:02Z",
                "sensor": {"imu_odometry": {"path_distance_m": 1.25}},
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(finished["episode"]["status"], "finished")
        result_path = Path(self.temp_dir.name) / episode_id / "result.json"
        result = json.loads(result_path.read_text())
        self.assertTrue(result["success"])
        self.assertEqual(result["action_history"], ["MOVE_FORWARD"])
        self.assertEqual(result["imu_odometry"]["path_distance_m"], 1.25)

    def test_bad_token_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("GET", "/api/episodes", token="wrong")
        self.assertEqual(caught.exception.code, 401)

    def test_only_one_active_episode_by_default(self):
        self.request("POST", "/api/episodes", {"episode_id": "one", "instruction": "Go"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("POST", "/api/episodes", {"episode_id": "two", "instruction": "Go"})
        self.assertEqual(caught.exception.code, 409)


if __name__ == "__main__":
    unittest.main()
