import io
import json
import tempfile
import unittest
from pathlib import Path

import imageio.v3 as iio
from PIL import Image

from navverse_tools.real_world_exp_server.store import EpisodeStore
from navverse_tools.real_world_exp_server.video import render_episode_video


class EpisodeVideoTest(unittest.TestCase):
    def test_render_episode_video_and_record_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = EpisodeStore(root)
            store.start(
                episode_id="video_episode",
                instruction="Find the closet",
                hfov=79,
                client_metadata={"agent": {"id": "longnav", "label": "LongNav"}},
            )
            buffer = io.BytesIO()
            Image.new("RGB", (320, 240), (40, 90, 150)).save(buffer, format="JPEG")
            store.add_step(
                episode_id="video_episode",
                request_id="step-1",
                image_bytes=buffer.getvalue(),
                captured_at="2026-08-05T00:00:00Z",
                sensor={"orientation": {"heading_deg": 42.5}},
                action={
                    "action": "MOVE_FORWARD",
                    "label": "forward",
                    "agent_response": {
                        "action_probs": [0.1, 0.7, 0.1, 0.1],
                        "latency_ms": 125,
                    },
                },
            )
            store.finish(
                episode_id="video_episode",
                reason="success",
                notes="",
                sensor={},
                client_time="2026-08-05T00:00:01Z",
            )

            video = render_episode_video(root / "video_episode", fps=2)
            result = store.update_result("video_episode", {"video": video})

            output = root / "video_episode" / "episode.mp4"
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(iio.immeta(output)["fps"], 2.0)
            self.assertEqual(result["video"]["path"], "episode.mp4")
            self.assertEqual(json.loads((root / "video_episode" / "result.json").read_text())["video"]["frames"], 1)


if __name__ == "__main__":
    unittest.main()
