import argparse
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import wandb
from utils.wandb_logging import (
    append_result_videos_to_index,
    wandb_log_indexed_videos,
    wandb_log_result_videos_by_test_index,
)


def load_results(result_path):
    if not result_path.exists():
        return []
    return json.loads(result_path.read_text())


def build_video_index(args, results):
    video_index = {}
    for episode_index, result in enumerate(results, start=1):
        append_result_videos_to_index(args, result, episode_index, video_index)
    return video_index


def filter_video_index(video_index, video_kind):
    if not video_kind:
        return video_index
    keep = {kind.strip() for kind in video_kind.split(",") if kind.strip()}
    return {kind: items for kind, items in video_index.items() if kind in keep}


def tmux_session_alive(session_name):
    if not session_name:
        return False
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Upload saved episode videos to an existing W&B run.")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default="navverse-benchmark")
    parser.add_argument("--entity", default="liandancoin-university-of-michigan")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--dump-location", default="./dump")
    parser.add_argument("--episode-labels", default="")
    parser.add_argument("--episode-folder", default="navverse_data/episodes")
    parser.add_argument("--media-prefix", default="episode_video")
    parser.add_argument("--test-index-media-prefix", default="episode_video_test_index")
    parser.add_argument("--video-kind", default="")
    parser.add_argument("--per-episode", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--watch-session", default="")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    result_path = Path(cli_args.dump_location) / cli_args.exp_name / "result.json"
    upload_args = SimpleNamespace(
        agent=cli_args.agent,
        episode_folder=cli_args.episode_folder,
        episode_labels=cli_args.episode_labels,
        wandb_upload_videos=True,
        wandb_video_prefix=cli_args.media_prefix,
        wandb_test_index_video_prefix=cli_args.test_index_media_prefix,
    )
    run = wandb.init(
        entity=cli_args.entity,
        project=cli_args.project,
        id=cli_args.run_id,
        resume="allow",
        name=cli_args.exp_name,
    )
    try:
        run.define_metric("episode/index")
        run.define_metric("episode/test_index")
        run.define_metric("episode_test_index")
        run.define_metric(f"{cli_args.media_prefix}/*", step_metric="episode/test_index")
        run.define_metric(f"{cli_args.test_index_media_prefix}/*", step_metric="episode_test_index")
    except Exception as exc:
        print(f"[wandb-video-uploader] failed to define metrics: {exc}", flush=True)

    uploaded_video_paths = set()
    finished_idle_cycles = 0
    print(
        f"[wandb-video-uploader] watching {result_path} for run {cli_args.run_id}",
        flush=True,
    )
    while True:
        try:
            results = load_results(result_path)
        except Exception as exc:
            print(f"[wandb-video-uploader] failed to read results: {exc}", flush=True)
            time.sleep(cli_args.poll_seconds)
            continue

        if cli_args.video_kind:
            video_kinds = {kind.strip() for kind in cli_args.video_kind.split(",") if kind.strip()}
        else:
            video_kinds = None

        if cli_args.per_episode:
            uploaded_count = 0
            for completed_count, result in enumerate(results, start=1):
                uploaded_count += wandb_log_result_videos_by_test_index(
                    wandb,
                    upload_args,
                    result,
                    completed_count,
                    uploaded_video_paths,
                    video_kinds,
                )
        else:
            video_index = filter_video_index(
                build_video_index(upload_args, results),
                cli_args.video_kind,
            )
            uploaded_count = wandb_log_indexed_videos(
                wandb,
                upload_args,
                video_index,
                len(results),
                uploaded_video_paths,
            )
        if uploaded_count:
            print(
                f"[wandb-video-uploader] uploaded {uploaded_count} new videos at completed={len(results)}",
                flush=True,
            )

        if cli_args.once:
            break
        if cli_args.watch_session and not tmux_session_alive(cli_args.watch_session):
            finished_idle_cycles += 1
            if finished_idle_cycles >= 2:
                break
        else:
            finished_idle_cycles = 0
        time.sleep(cli_args.poll_seconds)

    wandb.finish()
    print("[wandb-video-uploader] finished", flush=True)


if __name__ == "__main__":
    main()
