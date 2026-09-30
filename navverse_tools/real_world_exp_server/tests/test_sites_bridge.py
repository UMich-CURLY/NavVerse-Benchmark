import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from navverse_tools.real_world_exp_server.agent import DemoAgent, GuidanceConfig
from navverse_tools.real_world_exp_server.sites_bridge import (
    AgentProfile,
    Bridge,
    load_agent_profiles,
    load_env_file,
    parse_args,
)
from navverse_tools.real_world_exp_server.store import EpisodeStore


JPEG = b"\xff\xd8bridge-test\xff\xd9"
PNG = b"\x89PNG\r\n\x1a\nbridge-test"


class FakeSitesClient:
    site_url = "https://phone.example"

    def __init__(self):
        self.results = []
        self.uploaded_videos = []

    def image(self, job_id):
        self.last_image_job_id = job_id
        return JPEG

    def asset(self, job_id, name):
        self.last_asset = (job_id, name)
        return PNG

    def complete(self, job_id, result):
        self.results.append((job_id, result))

    def fail(self, job_id, error):
        self.results.append((job_id, {"error": error}))

    def register_agents(self, profiles):
        self.registered = list(profiles)

    def register_runs(self, runs):
        self.registered_runs = runs

    def upload_video(self, episode_id, video_path):
        self.uploaded_videos.append((episode_id, Path(video_path)))

    def fail_video_request(self, request_id, error):
        self.failed_video_request = (request_id, error)


class FakeManagedAgent:
    def __init__(self):
        self.ready = False
        self.stop_reasons = []
        self.idle_timeout = None
        self.reset_count = 0

    def status(self):
        return {"state": "ready" if self.ready else "stopped", "selectable": True}

    def ensure_ready(self, on_state_change):
        self.ready = True
        on_state_change()

    def schedule_idle_shutdown(self, timeout_s, on_state_change):
        self.idle_timeout = timeout_s
        on_state_change()

    def stop(self, reason):
        was_ready = self.ready
        self.ready = False
        self.stop_reasons.append(reason)
        return was_ready

    def reset(self):
        self.reset_count += 1
        return True


class BridgeTest(unittest.TestCase):
    def test_default_profiles_are_gpu0_managed_and_sgvln_is_blocked(self):
        args = parse_args([])
        profiles = load_agent_profiles(args)
        self.assertEqual(profiles["longnav"].agent.status()["gpu"], "0")
        self.assertEqual(profiles["uninavid"].agent.status()["gpu"], "0")
        self.assertEqual(profiles["longnav"].agent.delegate.reset_url, "http://127.0.0.1:8000/reset")
        self.assertEqual(profiles["uninavid"].agent.delegate.reset_url, "http://127.0.0.1:8300/reset")
        self.assertEqual(profiles["internvla-n1"].agent.status()["gpu"], "0")
        self.assertEqual(profiles["internvla-n1"].agent.delegate.guidance.turn_degrees, 15.0)
        self.assertEqual(profiles["poliformer"].agent.status()["gpu"], "0")
        self.assertEqual(
            profiles["poliformer"].agent.delegate.reset_url,
            "http://127.0.0.1:8400/reset",
        )
        self.assertEqual(profiles["sg-vln"].agent.status()["state"], "blocked")
        self.assertFalse(profiles["sg-vln"].agent.status()["selectable"])
        self.assertIn("unverified", profiles["sg-vln"].registration()["label"])
        self.assertIn("WARNING", profiles["poliformer"].registration()["description"])

        longnav_args = profiles["longnav"].agent.launch.args
        self.assertIn("Aasdfip/hm3d_rpp_ke_standard-checkpoint_231", longnav_args)
        repo = Path(__file__).resolve().parents[3]
        self.assertEqual(profiles["longnav"].agent.launch.python, repo / "baselines/longnav_r1/.venv/bin/python")
        self.assertEqual(profiles["longnav"].agent.launch.script, repo / "baselines/longnav_r1/serve_longnav.py")
        self.assertEqual(set(profiles), {"longnav", "uninavid", "internvla-n1", "poliformer", "sg-vln", "demo"})

    def test_warmup_starts_selected_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeSitesClient()
            managed = FakeManagedAgent()
            profile = AgentProfile("managed", "Managed", "Test", {}, managed)
            bridge = Bridge(client, EpisodeStore(Path(directory)), {"managed": profile}, 0.01)
            bridge.process(
                {
                    "id": 7,
                    "episode_id": "warmup-episode",
                    "agent_id": "managed",
                    "kind": "warmup",
                }
            )
            self.assertTrue(managed.ready)
            self.assertEqual(client.results, [(7, {"warmup_ack": True})])

    def test_warmup_stops_a_different_managed_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeSitesClient()
            selected = FakeManagedAgent()
            previous = FakeManagedAgent()
            previous.ready = True
            bridge = Bridge(
                client,
                EpisodeStore(Path(directory)),
                {
                    "selected": AgentProfile("selected", "Selected", "Test", {}, selected),
                    "previous": AgentProfile("previous", "Previous", "Test", {}, previous),
                },
                0.01,
            )
            bridge.process(
                {
                    "id": 7,
                    "episode_id": "warmup-episode",
                    "agent_id": "selected",
                    "kind": "warmup",
                }
            )
            self.assertTrue(selected.ready)
            self.assertFalse(previous.ready)
            self.assertIn("switched to selected", previous.stop_reasons[-1])

    def test_load_env_file_only_accepts_real_world_exp_server_keys(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {}, clear=True):
            path = Path(directory) / "bridge.env"
            path.write_text(
                "REAL_WORLD_EXP_SERVER_SITE_URL=https://phone.example\nIGNORED=value\n",
                encoding="utf-8",
            )
            load_env_file(path)
            self.assertEqual(os.environ["REAL_WORLD_EXP_SERVER_SITE_URL"], "https://phone.example")
            self.assertNotIn("IGNORED", os.environ)

    def test_step_and_finish_are_saved_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeSitesClient()
            store = EpisodeStore(Path(directory))
            bridge = Bridge(
                client,
                store,
                {
                    "demo": AgentProfile(
                        id="demo",
                        label="Demo",
                        description="Test agent",
                        capabilities={"observation": "rgb"},
                        agent=DemoAgent(["MOVE_FORWARD"], GuidanceConfig()),
                    )
                },
                poll_interval_s=0.01,
            )
            base = {
                "episode_id": "sites_episode",
                "instruction": "Find the chair",
                "agent_id": "demo",
                "agent_label": "Demo",
                "hfov": 79,
                "client_metadata": {},
            }
            bridge.process(
                {
                    **base,
                    "id": 1,
                    "request_id": "step-1",
                    "kind": "step",
                    "captured_at": "2026-08-05T00:00:00Z",
                    "sensor": {
                        "arkit": {
                            "assets": {"depth": True, "confidence": True},
                            "intrinsics": {"rgb": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                            "pose": {"camera_to_world": [[1, 0, 0, 0]]},
                        }
                    },
                }
            )
            self.assertEqual(client.results[0][1]["action"], "MOVE_FORWARD")
            self.assertEqual(store.get("sites_episode")["steps"][0]["request_id"], "step-1")
            episode_root = Path(directory) / "sites_episode" / "frames"
            self.assertEqual((episode_root / "0000.depth.png").read_bytes(), PNG)
            self.assertEqual((episode_root / "0000.confidence.png").read_bytes(), PNG)
            self.assertTrue((episode_root / "0000.intrinsics.json").is_file())
            self.assertTrue((episode_root / "0000.pose.json").is_file())

            with patch(
                "navverse_tools.real_world_exp_server.sites_bridge.render_episode_video",
                return_value={"path": "episode.mp4", "frames": 1},
            ):
                bridge.process(
                    {
                        **base,
                        "id": 2,
                        "request_id": "finish-1",
                        "kind": "finish",
                        "termination": {"reason": "success", "sensor": {}},
                    }
                )
            self.assertEqual(store.get("sites_episode")["status"], "finished")
            self.assertEqual(client.results[-1], (2, {"finish_ack": True}))
            self.assertEqual(client.uploaded_videos, [])
            self.assertEqual(client.registered_runs[0]["episode_id"], "sites_episode")
            self.assertFalse(client.registered_runs[0]["video"]["available"])

    def test_local_runs_are_indexed_and_video_is_uploaded_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeSitesClient()
            store = EpisodeStore(Path(directory))
            store.start(
                episode_id="indexed_run",
                instruction="Find the desk",
                hfov=79,
                client_metadata={"agent": {"id": "demo", "label": "Demo"}},
            )
            store.finish(
                episode_id="indexed_run",
                reason="success",
                notes="",
                sensor={},
                client_time=None,
            )
            video_path = Path(directory) / "indexed_run" / "episode.mp4"
            video_path.write_bytes(b"video")
            bridge = Bridge(client, store, {}, 0.01)

            runs = bridge.local_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["agent"], {"id": "demo", "label": "Demo"})
            self.assertEqual(runs[0]["num_steps"], 0)
            self.assertEqual(runs[0]["video"], {"available": True, "size_bytes": 5})
            self.assertEqual(client.uploaded_videos, [])

            bridge.process_video_request({"id": 4, "episode_id": "indexed_run"})
            self.assertEqual(client.uploaded_videos, [("indexed_run", video_path)])

    def test_finish_schedules_default_idle_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeSitesClient()
            managed = FakeManagedAgent()
            store = EpisodeStore(Path(directory))
            store.start(episode_id="finished", instruction="Find it", hfov=79, client_metadata={})
            bridge = Bridge(
                client,
                store,
                {"managed": AgentProfile("managed", "Managed", "Test", {}, managed)},
                0.01,
            )
            with patch(
                "navverse_tools.real_world_exp_server.sites_bridge.render_episode_video",
                return_value={"path": "episode.mp4", "frames": 0},
            ):
                bridge.process(
                    {
                        "id": 9,
                        "episode_id": "finished",
                        "agent_id": "managed",
                        "kind": "finish",
                        "termination": {"reason": "success", "sensor": {}},
                    }
                )
            self.assertEqual(managed.idle_timeout, 900.0)
            self.assertEqual(managed.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
