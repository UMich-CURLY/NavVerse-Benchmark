import errno
import os
import signal
import socket
import subprocess
import time
import torch
import ray
from ray.util.queue import Queue as _RayQueue
from .mp_utils import TaskQueue


def RayQueue(maxsize=0):
    return _RayQueue(maxsize=maxsize, actor_options={"num_cpus": 0})


@ray.remote(num_cpus=0)
class _EventBackend:
    def __init__(self):
        self._flag = False

    def set(self):
        self._flag = True

    def is_set(self):
        return self._flag


class RayEvent:
    def __init__(self):
        self._backend = _EventBackend.remote()

    def set(self):
        ray.get(self._backend.set.remote())

    def is_set(self):
        return ray.get(self._backend.is_set.remote())


class MockValue:
    def __init__(self, typecode, value):
        self.value = value


class RayLocalManager:
    def dict(self):
        return {}

    def list(self):
        return []

    def Value(self, typecode, value):
        return MockValue(typecode, value)


@ray.remote(num_cpus=0)
class _TaskQueueBackend(TaskQueue):
    def __init__(self, args):
        super().__init__(args, RayLocalManager())


class RayTaskQueue:
    def __init__(self, args):
        self._actor = _TaskQueueBackend.remote(args)

    def add_tasks(self, tasks):
        ray.get(self._actor.add_tasks.remote(tasks))

    def get_task(self, worker_id, last_scene_id=None):
        return ray.get(self._actor.get_task.remote(worker_id, last_scene_id))

    def reserve_task(self, worker_id):
        return ray.get(self._actor.reserve_task.remote(worker_id))

    def reserve_tasks(self, worker_id, max_tasks, last_scene_id=None):
        return ray.get(self._actor.reserve_tasks.remote(worker_id, max_tasks, last_scene_id))

    def get_tasks(self, worker_id, max_tasks, last_scene_id=None):
        return ray.get(self._actor.get_tasks.remote(worker_id, max_tasks, last_scene_id))

    def assign_task(self, worker_id, task):
        return ray.get(self._actor.assign_task.remote(worker_id, task))

    def requeue_task(self, task, front=True):
        return ray.get(self._actor.requeue_task.remote(task, front))

    def release_task(self, worker_id, episode_label=None):
        return ray.get(self._actor.release_task.remote(worker_id, episode_label))

    def release_tasks(self, worker_id):
        return ray.get(self._actor.release_tasks.remote(worker_id))

    def complete_task(self, worker_id, episode_label=None):
        return ray.get(self._actor.complete_task.remote(worker_id, episode_label))

    def empty(self):
        return ray.get(self._actor.empty.remote())


@ray.remote(num_cpus=0)
class _ProcessBackend:
    def __init__(self, target, args, kwargs):
        self.target = target
        self.args = args
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs

    def run(self):
        self.target(*self.args, **self.kwargs)


class RayProcess:
    def __init__(self, target, args=(), name=None, **kwargs):
        self.name = name
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self._actor = _ProcessBackend.remote(target, args, kwargs)
        self._future = None

    def start(self):
        self._future = self._actor.run.remote()

    def terminate(self):
        ray.kill(self._actor)

    def join(self, timeout=None):
        if self._future is None:
            return

        try:
            ray.get(self._future, timeout=timeout)
        except ray.exceptions.GetTimeoutError:
            pass
        except Exception as e:
            print(f"Process {self.name} crashed: {e}")


def best_effort_remote_call(logger, actor_handle, method_name, timeout_s=5.0):
    if actor_handle is None:
        return True
    try:
        method = getattr(actor_handle, method_name)
        ref = method.remote()
        ray.get(ref, timeout=timeout_s)
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning(f"{method_name} timed out or failed: {exc}")
        return False


def best_effort_kill(logger, actor_handle, actor_name):
    if actor_handle is None:
        return
    try:
        ray.kill(actor_handle, no_restart=True)
        if logger is not None:
            logger.info(f"Killed {actor_name}.")
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Failed to kill {actor_name}: {exc}")


def terminate_process_group(pid, timeout_s=3.0):
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass

    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)


def is_tcp_port_available(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return False
            raise
    return True


def wait_for_tcp_port_release(host, port, timeout_s=15.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_tcp_port_available(host, port):
            return True
        time.sleep(0.2)
    return is_tcp_port_available(host, port)


def tcp_listener_pids(host, port):
    try:
        result = subprocess.run(
            ["ss", "-ltnp", f"( sport = :{int(port)} )"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    pids = []
    for token in result.stdout.replace(",", " ").replace(")", " ").split():
        if not token.startswith("pid="):
            continue
        try:
            pid = int(token.split("=", 1)[1])
        except ValueError:
            continue
        if pid not in pids:
            pids.append(pid)
    return pids


def process_cmdline(pid):
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            data = f.read()
    except OSError:
        return ""
    return data.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def process_pids_matching_cmdline(required_cmd_substrings):
    required_cmd_substrings = tuple(required_cmd_substrings)
    pids = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        cmdline = process_cmdline(pid)
        if cmdline and all(text in cmdline for text in required_cmd_substrings):
            pids.append(pid)
    return pids


def terminate_processes_matching_cmdline(required_cmd_substrings, timeout_s=15.0):
    for pid in process_pids_matching_cmdline(required_cmd_substrings):
        terminate_process_group(pid, timeout_s=timeout_s)
    return not process_pids_matching_cmdline(required_cmd_substrings)


def terminate_tcp_port_listeners(host, port, timeout_s=15.0, required_cmd_substrings=None):
    del host
    pids = tcp_listener_pids("127.0.0.1", port)
    if not pids:
        return True
    required_cmd_substrings = tuple(required_cmd_substrings or ())
    for pid in pids:
        if pid == os.getpid():
            continue
        cmdline = process_cmdline(pid)
        if required_cmd_substrings and not any(text in cmdline for text in required_cmd_substrings):
            continue
        terminate_process_group(pid, timeout_s=timeout_s)
    return wait_for_tcp_port_release("127.0.0.1", port, timeout_s=timeout_s)


def can_connect(host, port, timeout=1.0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def socket_request_responds(host, port, request, timeout=1.0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.sendall(request.encode())
        header = bytearray()
        while len(header) < 8:
            chunk = sock.recv(8 - len(header))
            if not chunk:
                return False
            header.extend(chunk)
    return True


def find_available_port_block(host, start_port, block_size, blocked_ports=None):
    blocked_ports = set(blocked_ports or [])
    port = start_port
    while True:
        candidate_ports = [port + offset for offset in range(block_size)]
        if any(candidate_port in blocked_ports for candidate_port in candidate_ports):
            port += 1
            continue
        if all(is_tcp_port_available(host, candidate_port) for candidate_port in candidate_ports):
            return candidate_ports
        port += 1


def ray_ref_ready(ref, timeout_s=0.0):
    ready, _ = ray.wait([ref], timeout=timeout_s)
    return len(ready) > 0

def detect_first_gpu_total_vram_gb():
    if torch.cuda.device_count() <= 0:
        raise RuntimeError("torch.cuda did not report any GPUs in the driver process.")

    first_gpu_total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if first_gpu_total_vram_gb <= 0:
        raise RuntimeError("Failed to detect first GPU VRAM in the driver process.")
    return first_gpu_total_vram_gb
