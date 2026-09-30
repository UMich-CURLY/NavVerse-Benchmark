import unittest
from unittest.mock import Mock, patch

from navverse_tools.real_world_exp_server.agent import (
    AgentError,
    GuidanceConfig,
    HTTPAgent,
    encode_multipart,
    normalize_agent_response,
)


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.guidance = GuidanceConfig(forward_distance_m=0.5, turn_degrees=45)
        self.action_map = ["STOP", "FORWARD", "TURN_LEFT_SMALL", "TURN_RIGHT_SMALL"]

    def test_action_id_is_normalized(self):
        action = normalize_agent_response({"action_id": 2}, self.action_map, self.guidance)
        self.assertEqual(action["action"], "TURN_LEFT")
        self.assertEqual(action["target_delta_deg"], -45)

    def test_trajectory_becomes_relative_guidance(self):
        action = normalize_agent_response(
            {"trajectory": [[0, 0], [1, 1]]}, self.action_map, self.guidance
        )
        self.assertEqual(action["action"], "FOLLOW_TRAJECTORY")
        self.assertAlmostEqual(action["target_delta_deg"], 45)
        self.assertAlmostEqual(action["distance_m"], 1.414, places=3)

    def test_invalid_action_is_rejected(self):
        with self.assertRaises(AgentError):
            normalize_agent_response({"action": "JUMP"}, self.action_map, self.guidance)

    def test_multipart_contains_fields_and_image(self):
        body, content_type = encode_multipart(
            {"reset": "true"},
            "rgb",
            b"jpeg",
            extra_files={"depth": ("depth.png", "image/png", b"png")},
        )
        self.assertIn(b'name="reset"', body)
        self.assertIn(b'name="rgb"', body)
        self.assertIn(b"jpeg", body)
        self.assertIn(b'name="depth"', body)
        self.assertIn(b"image/png", body)
        self.assertIn("boundary=", content_type)

    def test_http_agent_reset_uses_the_configured_endpoint(self):
        agent = HTTPAgent(
            url="http://127.0.0.1:9000/act",
            reset_url="http://127.0.0.1:9000/reset",
            protocol="navverse",
            action_map=self.action_map,
            timeout_s=1,
            guidance=self.guidance,
        )
        response = Mock()
        response.read.return_value = b'{"ok": true}'
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch("navverse_tools.real_world_exp_server.agent.urllib.request.urlopen", return_value=context) as urlopen:
            self.assertTrue(agent.reset())
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:9000/reset")
        self.assertEqual(request.method, "POST")

    def test_http_agent_forwards_arkit_depth_and_metadata(self):
        agent = HTTPAgent(
            url="http://127.0.0.1:9000/act",
            protocol="navverse",
            action_map=self.action_map,
            timeout_s=1,
            guidance=self.guidance,
        )
        response = Mock()
        response.read.return_value = b'{"action": "FORWARD"}'
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch("navverse_tools.real_world_exp_server.agent.urllib.request.urlopen", return_value=context) as urlopen:
            result = agent.act(
                image_bytes=b"jpeg",
                instruction="go",
                episode_id="episode",
                reset=True,
                hfov=72,
                observation={"source": "arkit", "depth": {"scale_m": 0.001}},
                depth_bytes=b"depth-png",
                confidence_bytes=b"confidence-png",
            )
        body = urlopen.call_args.args[0].data
        self.assertEqual(result["action"], "MOVE_FORWARD")
        self.assertIn(b'name="observation_json"', body)
        self.assertIn(b'name="depth"', body)
        self.assertIn(b"depth-png", body)
        self.assertIn(b'name="confidence"', body)


if __name__ == "__main__":
    unittest.main()
