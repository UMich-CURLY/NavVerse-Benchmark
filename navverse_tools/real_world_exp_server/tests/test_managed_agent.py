import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from navverse_tools.real_world_exp_server.managed_agent import LaunchSpec, ManagedHTTPAgent


class ManagedAgentTest(unittest.TestCase):
    def test_launch_forces_physical_gpu0(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "serve.py"
            script.write_text("print('unused')\n", encoding="utf-8")
            launch = LaunchSpec(
                env_name="navverse",
                script=script,
                args=("--port", "9999"),
                health_url="http://127.0.0.1:9999/health",
                startup_timeout_s=1,
            )
            process = Mock()
            process.poll.return_value = None
            agent = ManagedHTTPAgent("test", Mock(), launch, root, root / "logs")
            with (
                patch.object(agent, "_healthy", side_effect=[False, True]),
                patch("navverse_tools.real_world_exp_server.managed_agent.subprocess.Popen", return_value=process) as popen,
            ):
                agent.ensure_ready()
            environment = popen.call_args.kwargs["env"]
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], "0")
            agent._process = None
            agent._close_process()

    def test_idle_shutdown_terminates_the_managed_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            process.pid = 1234
            process.poll.return_value = None
            agent = ManagedHTTPAgent(
                "test",
                Mock(),
                LaunchSpec("navverse", root / "serve.py", (), "http://127.0.0.1/health"),
                root,
                root / "logs",
            )
            agent._process = process
            with patch("navverse_tools.real_world_exp_server.managed_agent.os.killpg") as killpg:
                agent.schedule_idle_shutdown(0.01)
                time.sleep(0.05)
            killpg.assert_called_once_with(1234, 15)
            self.assertIsNone(agent._process)

    def test_same_agent_warmup_cancels_idle_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            process.pid = 1234
            process.poll.return_value = None
            agent = ManagedHTTPAgent(
                "test",
                Mock(),
                LaunchSpec("navverse", root / "serve.py", (), "http://127.0.0.1/health"),
                root,
                root / "logs",
            )
            agent._process = process
            with (
                patch.object(agent, "_healthy", return_value=True),
                patch("navverse_tools.real_world_exp_server.managed_agent.os.killpg") as killpg,
            ):
                agent.schedule_idle_shutdown(0.02)
                agent.ensure_ready()
                time.sleep(0.05)
            killpg.assert_not_called()
            agent._process = None
            agent._close_process()

    def test_reset_delegates_without_stopping_the_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            process.poll.return_value = None
            delegate = Mock()
            delegate.reset.return_value = True
            agent = ManagedHTTPAgent(
                "test",
                delegate,
                LaunchSpec("navverse", root / "serve.py", (), "http://127.0.0.1/health"),
                root,
                root / "logs",
            )
            agent._process = process
            with patch.object(agent, "_healthy", return_value=True):
                self.assertTrue(agent.reset())
            delegate.reset.assert_called_once_with()
            self.assertIs(agent._process, process)
            agent._process = None
            agent._close_process()


if __name__ == "__main__":
    unittest.main()
