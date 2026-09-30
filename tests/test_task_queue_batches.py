import json
from types import SimpleNamespace

import pytest

from baselines.utils.mp_utils import TaskQueue


class LocalManager:
    def dict(self):
        return {}

    def list(self):
        return []

    def Value(self, _, value):
        return SimpleNamespace(value=value)


def make_queue(tmp_path, completed=()):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "result.json").write_text(
        json.dumps([{"episode_label": label} for label in completed])
    )
    queue = TaskQueue(SimpleNamespace(dump_location=tmp_path, exp_name="run"), LocalManager())
    queue.add_tasks(
        {"scene_id": scene, "episode_label": f"{scene}_{index}"}
        for scene, count in (("A", 3), ("B", 2))
        for index in range(count)
    )
    return queue


def test_batch_claims_stay_in_one_scene_and_recover(tmp_path):
    queue = make_queue(tmp_path)
    first = queue.reserve_tasks(0, 2)
    second = queue.reserve_tasks(1, 3)
    assert [task["episode_label"] for task in first] == ["A_0", "A_1"]
    assert [task["episode_label"] for task in second] == ["B_0", "B_1"]
    assert queue.get_tasks(0, 3) == first
    assert len(queue) == 5
    assert not queue.requeue_task(first[1])
    assert not queue.complete_task(0, "B_0")
    assert queue.complete_task(0, "A_0")
    assert len(queue) == 4

    released = queue.release_tasks(0)
    assert released == [first[1]]
    assert queue.requeue_task(released[0])
    assert [task["episode_label"] for task in queue.reserve_tasks(0, 2, "A")] == ["A_1", "A_2"]
    assert queue.complete_task(0, "A_2")
    assert queue.complete_task(0, "A_1")
    assert queue.complete_task(1, "B_0")
    assert queue.complete_task(1, "B_1")
    assert queue.empty()
    assert set(queue.result_labels) == {"A_0", "A_1", "A_2", "B_0", "B_1"}


def test_batch_conversion_and_single_task_interface(tmp_path):
    queue = make_queue(tmp_path, completed=("A_0",))
    assert queue.reserve_task(0)["episode_label"] == "A_1"
    assert [task["episode_label"] for task in queue.reserve_tasks(0, 2)] == ["A_1", "A_2"]
    with pytest.raises(RuntimeError, match="batch reservation"):
        queue.get_task(0)
    with pytest.raises(RuntimeError, match="batch reservation"):
        queue.assign_task(0, {"scene_id": "B", "episode_label": "B_0"})
    assert queue.release_task(0, "A_1")
    assert [task["episode_label"] for task in queue.get_tasks(0, 2)] == ["A_2"]
    assert queue.complete_task(0, "A_2")
    assert queue.reserve_task(1)["episode_label"] == "B_0"
    assert queue.get_task(1)["episode_label"] == "B_0"


def test_batch_rejects_invalid_size(tmp_path):
    queue = make_queue(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        queue.reserve_tasks(0, 0)
