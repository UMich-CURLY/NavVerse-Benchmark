from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.utils import wandb_logging


class FakeTable:
    def __init__(self, data=None, columns=None, **_kwargs):
        self.data = list(data or [])
        self.columns = list(columns or [])

    def add_data(self, *row):
        self.data.append(list(row))


class FakePlot:
    @staticmethod
    def line(table, x, y, **kwargs):
        return {"table": table, "x": x, "y": y, **kwargs}


class FakeWandb:
    Table = FakeTable
    plot = FakePlot()

    def __init__(self):
        self.logs = []
        self.finished = 0

    @staticmethod
    def Video(path, caption, format):
        return {"path": path, "caption": caption, "format": format}

    def log(self, payload):
        self.logs.append(payload)

    def finish(self):
        self.finished += 1


def make_args(tmp_path, *, canonical=False, labels=("episode_a", "episode_b")):
    episode_folder = tmp_path / "episodes"
    episode_folder.mkdir(exist_ok=True)
    (episode_folder / "test_set.txt").write_text("\n".join(labels) + "\n")
    return SimpleNamespace(
        agent="Dummy",
        canonical_video_index=canonical,
        canonical_video_prefix="canonical_video",
        dump_location=str(tmp_path / "dump"),
        episode_folder=str(episode_folder),
        episode_labels="test",
        exp_name="run",
        log_file=str(tmp_path / "wandb.log"),
        wandb=True,
        wandb_upload_videos=True,
        wandb_test_index_video_prefix="episode_video_test_index",
    )


def make_result(tmp_path, label, *, with_video=True):
    result = {
        "episode_label": label,
        "scene_id": label.rsplit("_", 1)[0],
        "success": 1,
        "spl": 0.5,
    }
    if with_video:
        video = tmp_path / f"{label}_modern_Dummy.mp4"
        video.write_bytes(b"video")
        result["video_save_paths"] = [str(video)]
    return result


def run_thread(monkeypatch, args, fake_wandb, results=(), summary=None):
    run = SimpleNamespace(
        summary=summary or {}, entity="entity", project="project", id="run-id"
    )
    monkeypatch.setattr(wandb_logging, "setup_tee", lambda _path: None)
    monkeypatch.setattr(wandb_logging, "ignore_sigint_in_child", lambda: None)
    monkeypatch.setattr(wandb_logging, "init_wandb", lambda _args: (fake_wandb, run))
    result_queue = queue.Queue()
    for result in results:
        result_queue.put(result)
    quit_event = threading.Event()
    quit_event.set()
    wandb_logging.save_result_thread_func(args, result_queue, quit_event)


def logged_keys(fake_wandb):
    return {key for payload in fake_wandb.logs for key in payload}


def test_default_run_uploads_each_episode_without_canonical_array(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    fake = FakeWandb()
    results = [make_result(tmp_path, label) for label in ("episode_a", "episode_b")]

    run_thread(monkeypatch, args, fake, results)

    video_payloads = [
        payload
        for payload in fake.logs
        if "episode_video_test_index/modern" in payload
    ]
    assert len(video_payloads) == 2
    assert not any(key.startswith("canonical_video/") for key in logged_keys(fake))
    assert fake.finished == 1


def test_incomplete_canonical_run_publishes_no_array(tmp_path, monkeypatch):
    args = make_args(tmp_path, canonical=True)
    fake = FakeWandb()

    run_thread(monkeypatch, args, fake, [make_result(tmp_path, "episode_a")])

    assert not any(key.startswith("canonical_video/") for key in logged_keys(fake))
    assert "episode_video_test_index/modern" not in logged_keys(fake)


def test_complete_canonical_array_is_sorted_by_manifest_test_index(
    tmp_path, monkeypatch
):
    labels = ("episode_b", "episode_a")
    args = make_args(tmp_path, canonical=True, labels=labels)
    fake = FakeWandb()
    results = [make_result(tmp_path, label) for label in reversed(labels)]

    run_thread(monkeypatch, args, fake, results)

    payload = next(
        item for item in fake.logs if "canonical_video/modern" in item
    )
    paths = [Path(video["path"]).name for video in payload["canonical_video/modern"]]
    assert paths == [
        "episode_b_modern_Dummy.mp4",
        "episode_a_modern_Dummy.mp4",
    ]
    table = payload["canonical_video/index_table"]
    assert [row[1] for row in table.data] == [0, 1]
    assert [row[3] for row in table.data] == [1, 2]


def test_canonical_array_rejects_missing_video_file(tmp_path):
    args = make_args(tmp_path, canonical=True, labels=("episode_a",))
    fake = FakeWandb()
    missing = tmp_path / "missing.mp4"
    video_index = {
        "modern": [
            {
                "index": 0,
                "episode_index": 1,
                "episode_test_index": 1,
                "episode_label": "episode_a",
                "path": str(missing),
                "caption": "episode_a",
            }
        ]
    }

    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        wandb_logging.wandb_log_indexed_videos(fake, args, video_index, 1)


def test_resume_backfills_only_history_missing_from_remote_summary(
    tmp_path, monkeypatch
):
    labels = ("episode_b", "episode_c", "episode_a")
    args = make_args(tmp_path, labels=labels)
    completion_order = ("episode_a", "episode_b", "episode_c")
    results = [
        make_result(tmp_path, label, with_video=False) for label in completion_order
    ]
    result_path = Path(args.dump_location) / args.exp_name / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(results))
    fake = FakeWandb()

    run_thread(
        monkeypatch,
        args,
        fake,
        summary={"progress/completed_episodes": 2},
    )

    metric_payloads = [
        payload for payload in fake.logs if "progress/completed_episodes" in payload
    ]
    assert [payload["progress/completed_episodes"] for payload in metric_payloads] == [3]
    result_table = metric_payloads[0]["episode/results_table"]
    label_column = result_table.columns.index("episode_label")
    assert [row[label_column] for row in result_table.data] == list(labels)
    chart = metric_payloads[0]["episode_charts/success"]
    test_index_column = chart["table"].columns.index("episode_test_index")
    assert [row[test_index_column] for row in chart["table"].data] == [1, 2, 3]
