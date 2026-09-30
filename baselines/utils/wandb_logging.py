import json
import os
import signal
import time

from baselines.utils.logging import setup_tee

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected in benchmark runs.
    np = None


WANDB_PROGRESS_INTERVAL_S = 10.0
WANDB_LOG_PROGRESS_CHARTS = True


def _step_data_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _step_data_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_step_data_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _queue_empty_exception(exc):
    return exc.__class__.__name__ in {"Empty", "QueueEmpty"}


def save_step_data_thread_func(args, step_data_queue, quit_event):
    setup_tee(args.log_file)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    output_dir = os.path.join(args.dump_location, args.exp_name, "step_data")
    os.makedirs(output_dir, exist_ok=True)
    handles = {}
    try:
        while not quit_event.is_set() or not step_data_queue.empty():
            drained = 0
            while True:
                try:
                    step_data = step_data_queue.get(block=False)
                except Exception as exc:
                    if _queue_empty_exception(exc):
                        break
                    print(f"[step_data] queue read failed: {exc}")
                    break

                episode_label = str(step_data.get("episode_label", "unknown"))
                path = os.path.join(output_dir, f"{episode_label}.jsonl")
                handle = handles.get(path)
                if handle is None:
                    handle = open(path, "a", buffering=1)
                    handles[path] = handle
                handle.write(json.dumps(_step_data_jsonable(step_data), ensure_ascii=True) + "\n")
                drained += 1
                if step_data.get("final_step") or step_data.get("action") == 0:
                    handle.flush()
            if drained == 0:
                time.sleep(0.1)
    finally:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        print("[step_data] save thread finished")


def _jsonable_config(args):
    config = {}
    for key, value in vars(args).items():
        try:
            json.dumps(value)
            config[key] = value
        except TypeError:
            config[key] = str(value)
    return config


def init_wandb(args):
    if not getattr(args, "wandb", False):
        return None, None
    try:
        import wandb
    except Exception as exc:
        print(f"[wandb] disabled because wandb import failed: {exc}")
        return None, None

    init_kwargs = {
        "entity": "liandancoin-university-of-michigan",
        "project": args.wandb_project,
        "name": args.wandb_run_name or args.exp_name,
        "config": _jsonable_config(args),
    }
    if os.environ.get("WANDB_RUN_ID"):
        init_kwargs["id"] = os.environ["WANDB_RUN_ID"]
        init_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[wandb] disabled because wandb init failed: {exc}")
        return None, None
    try:
        run.define_metric("episode/index")
        run.define_metric("episode/test_index")
        run.define_metric("episode_test_index")
        run.define_metric("episode/*", step_metric="episode/index")
        run.define_metric("aggregate/*", step_metric="episode/index")
        run.define_metric("progress/completed_episodes", step_metric="episode/index")
        run.define_metric("episode_video/*", step_metric="episode/test_index")
        run.define_metric("episode_video_test_index/*", step_metric="episode_test_index")
    except Exception as exc:
        print(f"[wandb] failed to define custom metrics: {exc}")
    print(f"[wandb] logging run to {run.url if run is not None else 'local/offline run'}")
    return wandb, run


EPISODE_METRIC_COLUMNS = [
    "success",
    "soft_success",
    "spl",
    "soft_spl",
    "distance_to_goal",
    "oracle_navigation_error",
    "oracle_success",
    "step",
    "sim_duration",
    "episode_duration",
    "path_length",
    "collision",
    "collision_rate",
    "covered_area",
    "exploration_efficiency",
    "distance_to_wall",
    "average_distance_to_wall",
    "closest_goal",
    "navigable_surface",
    "navigable_surface_ratio",
]

EPISODE_TABLE_COLUMNS = [
    "completed_episode",
    "episode_test_index",
    "episode_label",
    "scene_id",
    "termination_reason",
    "video_path",
    "instruction",
] + EPISODE_METRIC_COLUMNS

EPISODE_CHART_METRICS = [
    "success",
    "soft_success",
    "spl",
    "soft_spl",
    "distance_to_goal",
    "oracle_navigation_error",
    "collision_rate",
    "step",
    "sim_duration",
]

EPISODE_AGGREGATE_METRICS = [
    "success",
    "soft_success",
    "spl",
    "soft_spl",
]


def _episode_set_path(args):
    episode_labels = getattr(args, "episode_labels", "")
    episode_folder = getattr(args, "episode_folder", "")
    if not episode_labels or not episode_folder or "," in episode_labels:
        return None
    return os.path.join(episode_folder, f"{episode_labels}_set.txt")


def _load_episode_test_index_map(args):
    cached = getattr(args, "_wandb_episode_test_index_map", None)
    if cached is not None:
        return cached

    labels = []
    episode_labels = getattr(args, "episode_labels", "")
    if episode_labels and "," in episode_labels:
        labels = [label.strip() for label in episode_labels.split(",") if label.strip()]
    else:
        set_path = _episode_set_path(args)
        if set_path and os.path.exists(set_path):
            with open(set_path, "r") as f:
                labels = [line.strip() for line in f if line.strip()]

    index_map = {label: idx for idx, label in enumerate(labels, start=1)}
    try:
        setattr(args, "_wandb_episode_test_index_map", index_map)
    except Exception:
        pass
    return index_map


def episode_test_index(args, episode_label, fallback_index):
    if not episode_label:
        return fallback_index
    return _load_episode_test_index_map(args).get(episode_label, fallback_index)


def result_video_paths(result):
    paths = []
    raw_paths = result.get("video_save_paths", [])
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]
    paths.extend(raw_paths)
    legacy_path = result.get("video_save_path")
    if legacy_path:
        paths.append(legacy_path)

    seen = set()
    unique_paths = []
    for path in paths:
        if path and "_rgb_" not in os.path.basename(path) and path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def result_table_row(args, result, completed_count):
    video_paths = result_video_paths(result)
    label = result.get("episode_label", "")
    row = {
        "completed_episode": completed_count,
        "episode_test_index": episode_test_index(args, label, completed_count),
        "episode_label": label,
        "scene_id": result.get("scene_id", ""),
        "termination_reason": result.get("termination_reason", ""),
        "video_path": video_paths[0] if video_paths else "",
        "instruction": result.get("instruction", ""),
    }
    for key in EPISODE_METRIC_COLUMNS:
        row[key] = result.get(key)
    return [row.get(column) for column in EPISODE_TABLE_COLUMNS]


def episode_rows_as_dicts(episode_rows):
    dict_rows = []
    for row in episode_rows:
        dict_rows.append({
            column: row[idx] if idx < len(row) else None
            for idx, column in enumerate(EPISODE_TABLE_COLUMNS)
        })
    return dict_rows


def episode_rows_by_test_index(episode_rows):
    test_index_column = EPISODE_TABLE_COLUMNS.index("episode_test_index")
    return sorted(episode_rows, key=lambda row: row[test_index_column])


def numeric_values(rows, metric):
    values = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def mean_or_none(values):
    if not values:
        return None
    return sum(values) / len(values)


def add_episode_scalar_metrics(payload, rows, latest_row):
    for metric in EPISODE_AGGREGATE_METRICS:
        value = latest_row.get(metric)
        if value is not None:
            payload[f"episode/{metric}"] = value

        mean_value = mean_or_none(numeric_values(rows, metric))
        if mean_value is not None:
            payload[f"aggregate/{metric}_mean"] = mean_value

    aliases = {
        "SR": "success",
        "soft_SR": "soft_success",
        "SPL": "spl",
        "soft_SPL": "soft_spl",
    }
    for alias, metric in aliases.items():
        mean_value = mean_or_none(numeric_values(rows, metric))
        if mean_value is not None:
            payload[f"aggregate/{alias}"] = mean_value


def episode_video_kind(args, result, video_path):
    stem = os.path.basename(video_path)
    suffix = f"_{args.agent}.mp4"
    if stem.endswith(suffix):
        stem = stem[:-len(suffix)]
    episode_label = result.get("episode_label", "")
    if episode_label and stem.startswith(f"{episode_label}_"):
        stem = stem[len(episode_label) + 1:]
    return stem or "video"


def _format_metric_value(value, digits=3, unit=""):
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        text = str(int(number))
    else:
        text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return f"{text}{unit}"


def result_video_caption(result, test_index, episode_label):
    metrics = [
        f"success={_format_metric_value(result.get('success'), digits=0)}",
        f"spl={_format_metric_value(result.get('spl'))}",
        f"sim={_format_metric_value(result.get('sim_duration'), unit='s')}",
        f"oracle_err={_format_metric_value(result.get('oracle_navigation_error'), unit='m')}",
    ]
    return f"{test_index:04d} {episode_label}\n" + " | ".join(metrics)


def append_result_videos_to_index(args, result, completed_count, video_index):
    episode_label = result.get("episode_label", "") or f"episode_{completed_count:04d}"
    test_index = episode_test_index(args, episode_label, completed_count)
    for video_path in result_video_paths(result):
        kind = episode_video_kind(args, result, video_path)
        items = video_index.setdefault(kind, [])
        if any(item["path"] == video_path for item in items):
            continue
        items.append({
            "index": len(items),
            "episode_index": completed_count,
            "episode_test_index": test_index,
            "episode_label": episode_label,
            "path": video_path,
            "caption": result_video_caption(result, test_index, episode_label),
        })


def _sorted_video_items(items):
    return sorted(
        items,
        key=lambda item: (
            item.get("episode_test_index") or item.get("episode_index") or item.get("index", 0),
            item.get("episode_label") or "",
            item.get("path") or "",
        ),
    )


def _indexed_video_kind(wandb, kind, items, uploaded_video_paths):
    videos = []
    table_rows = []
    missing_paths = []
    uploaded_paths = []
    for array_index, item in enumerate(_sorted_video_items(items)):
        video_path = item["path"]
        table_rows.append([
            kind,
            array_index,
            item["episode_index"],
            item.get("episode_test_index", item["episode_index"]),
            item["episode_label"],
            video_path,
        ])
        if not os.path.exists(video_path):
            missing_paths.append(video_path)
        elif uploaded_video_paths is None or video_path not in uploaded_video_paths:
            videos.append(
                wandb.Video(video_path, caption=item["caption"], format="mp4")
            )
            uploaded_paths.append(video_path)
    return videos, table_rows, missing_paths, uploaded_paths


def wandb_log_indexed_videos(wandb, args, video_index, completed_count, uploaded_video_paths=None):
    if wandb is None or not getattr(args, "wandb_upload_videos", True):
        return 0
    if not video_index:
        return 0

    video_prefix = getattr(
        args,
        "canonical_video_prefix",
        getattr(args, "wandb_video_prefix", "episode_video"),
    )
    payload = {"episode/index": completed_count}
    table_rows = []
    missing_paths = []
    uploaded_this_call = []
    for kind, items in sorted(video_index.items()):
        videos, kind_rows, kind_missing, kind_uploaded = _indexed_video_kind(
            wandb, kind, items, uploaded_video_paths
        )
        table_rows.extend(kind_rows)
        missing_paths.extend(kind_missing)
        uploaded_this_call.extend(kind_uploaded)
        if videos:
            payload[f"{video_prefix}/{kind}"] = videos

    if missing_paths:
        raise FileNotFoundError(
            "Missing video files required for W&B upload: " + ", ".join(missing_paths[:10])
        )
    if len(payload) == 1:
        return 0

    payload[f"{video_prefix}/index_table"] = wandb.Table(
        data=table_rows,
        columns=[
            "video_key",
            "array_index",
            "completed_episode",
            "episode_test_index",
            "episode_label",
            "video_path",
        ],
    )
    wandb.log(payload)
    if uploaded_video_paths is not None:
        uploaded_video_paths.update(uploaded_this_call)
    return len(uploaded_this_call)


def wandb_log_result_videos_by_test_index(
    wandb,
    args,
    result,
    completed_count,
    uploaded_video_paths=None,
    video_kinds=None,
):
    if wandb is None or not getattr(args, "wandb_upload_videos", True):
        return 0

    episode_label = result.get("episode_label", "") or f"episode_{completed_count:04d}"
    test_index = episode_test_index(args, episode_label, completed_count)
    video_prefix = getattr(args, "wandb_test_index_video_prefix", "episode_video_test_index")
    uploaded_count = 0
    for video_path in result_video_paths(result):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Missing video file required for W&B upload: {video_path}")
        if uploaded_video_paths is not None and video_path in uploaded_video_paths:
            continue
        kind = episode_video_kind(args, result, video_path)
        if video_kinds is not None and kind not in video_kinds:
            continue
        caption = result_video_caption(result, test_index, episode_label)
        payload = {
            "episode/index": completed_count,
            "episode/test_index": test_index,
            "episode/label": episode_label,
            "episode_test_index": test_index,
            "episode_label": episode_label,
            f"{video_prefix}/{kind}": wandb.Video(video_path, caption=caption, format="mp4"),
        }
        wandb.log(payload)
        if uploaded_video_paths is not None:
            uploaded_video_paths.add(video_path)
        uploaded_count += 1
    return uploaded_count


def wandb_log_result(wandb, args, result, completed_count, episode_rows):
    if wandb is None:
        return

    episode_rows.append(result_table_row(args, result, completed_count))
    episode_dict_rows = episode_rows_as_dicts(episode_rows)
    latest_row = episode_dict_rows[-1]
    sorted_rows = episode_rows_by_test_index(episode_rows)
    result_table = wandb.Table(data=sorted_rows, columns=EPISODE_TABLE_COLUMNS)
    chart_table = wandb.Table(data=sorted_rows, columns=EPISODE_TABLE_COLUMNS)
    payload = {
        "episode/index": completed_count,
        "episode/test_index": latest_row.get("episode_test_index"),
        "episode/label": latest_row.get("episode_label"),
        "episode_test_index": latest_row.get("episode_test_index"),
        "episode_label": latest_row.get("episode_label"),
        "progress/completed_episodes": completed_count,
        "episode/results_table": result_table,
    }
    add_episode_scalar_metrics(payload, episode_dict_rows, latest_row)

    for metric in EPISODE_CHART_METRICS:
        try:
            payload[f"episode_charts/{metric}"] = wandb.plot.line(
                chart_table,
                "episode_test_index",
                metric,
                title=f"{metric} by episode",
            )
        except Exception as exc:
            print(f"[wandb] failed to log episode chart {metric}: {exc}")

    wandb.log(payload)


PROGRESS_COLUMNS = [
    "elapsed_s",
    "worker",
    "worker_index",
    "worker_id",
    "role",
    "gpu",
    "port",
    "episode",
    "scene_id",
    "status",
    "progress_value",
    "agent_step",
    "sim_frame_count",
    "sim_fps",
    "sim_to_wall_ratio",
    "sim_total_time_s",
    "restart_count",
    "last_action",
    "last_message",
    "updated_ago_s",
]


def make_progress_rows(snapshot):
    elapsed_s = float(snapshot.get("elapsed_s") or 0.0)
    now_s = float(snapshot.get("wall_time_s") or time.time())
    rows = []
    worker_states = snapshot.get("worker_states", {})
    for raw_worker_id, state in sorted(worker_states.items(), key=lambda item: int(item[0])):
        worker_id = int(raw_worker_id)
        episode = state.get("assigned_episode") or ""
        scene_id = state.get("scene_id") or ""
        port = int(state.get("sim_port") or 0)
        restart_count = int(state.get("restart_count") or 0)
        updated_at_s = float(state.get("updated_at_s") or now_s)
        updated_ago_s = max(0.0, now_s - updated_at_s)
        agent_step = int(state.get("agent_step") or 0)
        sim_frame_count = int(state.get("sim_frame_count") or 0)
        sim_fps = float(state.get("sim_fps") or 0.0)
        sim_to_wall_ratio = float(state.get("sim_to_wall_ratio") or 0.0)
        sim_total_time_s = float(state.get("sim_total_time_s") or 0.0)

        rows.append([
            elapsed_s,
            f"sim-{worker_id}",
            worker_id * 2,
            worker_id,
            "sim",
            state.get("sim_cuda_visible_devices") or "",
            port,
            episode,
            scene_id,
            state.get("sim_status") or "",
            sim_frame_count,
            agent_step,
            sim_frame_count,
            sim_fps,
            sim_to_wall_ratio,
            sim_total_time_s,
            restart_count,
            "",
            state.get("last_message") or "",
            updated_ago_s,
        ])
        rows.append([
            elapsed_s,
            f"agent-{worker_id}",
            worker_id * 2 + 1,
            worker_id,
            "agent",
            state.get("agent_cuda_visible_devices") or "",
            port,
            episode,
            scene_id,
            state.get("agent_status") or "",
            agent_step,
            agent_step,
            sim_frame_count,
            sim_fps,
            sim_to_wall_ratio,
            sim_total_time_s,
            restart_count,
            state.get("agent_last_action") or "",
            state.get("last_message") or "",
            updated_ago_s,
        ])
    return rows


def drain_queue_nowait(queue_obj):
    if queue_obj is None:
        return []
    items = []
    while True:
        try:
            items.append(queue_obj.get_nowait())
        except Exception:
            break
    return items


def queue_is_empty(queue_obj):
    if queue_obj is None:
        return True
    try:
        return queue_obj.empty()
    except Exception:
        return True


def add_progress_charts(wandb, payload, history_table):
    if not WANDB_LOG_PROGRESS_CHARTS:
        return
    try:
        payload["progress/thread_progress_chart"] = wandb.plot.line(
            history_table,
            "elapsed_s",
            "progress_value",
            stroke="worker",
            title="Thread progress over wall time",
        )
    except Exception as exc:
        print(f"[wandb] failed to log progress line chart: {exc}")


def wandb_log_progress(wandb, snapshot, history_table=None):
    if wandb is None:
        return history_table
    rows = make_progress_rows(snapshot)
    if not rows:
        return history_table

    current_table = wandb.Table(data=rows, columns=PROGRESS_COLUMNS)
    if history_table is None:
        try:
            history_table = wandb.Table(columns=PROGRESS_COLUMNS, log_mode="INCREMENTAL")
        except TypeError:
            history_table = wandb.Table(columns=PROGRESS_COLUMNS)
    for row in rows:
        history_table.add_data(*row)
    payload = {
        "progress/thread_status": current_table,
        "progress/thread_timeline_table": history_table,
    }
    add_progress_charts(wandb, payload, history_table)

    wandb.log(payload)
    return history_table


def ignore_sigint_in_child():
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        pass


def preload_existing_results(args, result_path):
    completed_count = 0
    episode_rows = []
    video_index = {}
    existing_results = []
    if not os.path.exists(result_path):
        return completed_count, episode_rows, video_index, existing_results
    try:
        with open(result_path, "r") as f:
            existing_results = json.load(f)
        for existing_result in existing_results:
            completed_count += 1
            episode_rows.append(result_table_row(args, existing_result, completed_count))
            append_result_videos_to_index(args, existing_result, completed_count, video_index)
    except Exception as exc:
        print(f"[wandb] failed to preload existing results for resumed run: {exc}")
    return completed_count, episode_rows, video_index, existing_results


def wandb_completed_episode_count(wandb, run):
    def summary_count(summary):
        for key in ("progress/completed_episodes", "episode/index"):
            try:
                value = summary.get(key)
            except Exception:
                continue
            if value is not None:
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    continue
        return 0

    count = summary_count(getattr(run, "summary", {}))
    if count or not os.environ.get("WANDB_RUN_ID"):
        return count

    try:
        remote_run = wandb.Api().run(f"{run.entity}/{run.project}/{run.id}")
        return summary_count(remote_run.summary)
    except Exception as exc:
        print(f"[wandb] failed to read resumed run summary: {exc}")
        return 0


def append_results_to_disk(result_path, new_results):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    all_results = []
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            all_results = json.load(f)
    all_results.extend(new_results)
    with open(result_path, "w+") as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved result to {result_path}")


def handle_new_results(
    wandb,
    args,
    new_results,
    episode_rows,
    video_index,
    completed_count,
    uploaded_video_paths,
):
    canonical_video_index = getattr(args, "canonical_video_index", False)
    for result in new_results:
        completed_count += 1
        append_result_videos_to_index(args, result, completed_count, video_index)
        wandb_log_result(wandb, args, result, completed_count, episode_rows)
        if not canonical_video_index:
            wandb_log_result_videos_by_test_index(
                wandb,
                args,
                result,
                completed_count,
                uploaded_video_paths,
            )
    return completed_count


def backfill_existing_results(wandb, run, args, existing_results):
    already_logged_count = min(
        len(existing_results),
        wandb_completed_episode_count(wandb, run),
    )
    print(
        "[wandb] resumed result history: "
        f"{already_logged_count} already logged, "
        f"{len(existing_results) - already_logged_count} to backfill"
    )
    episode_rows = []
    video_index = {}
    uploaded_video_paths = set()
    canonical_video_index = getattr(args, "canonical_video_index", False)
    for existing_index, existing_result in enumerate(existing_results, start=1):
        append_result_videos_to_index(args, existing_result, existing_index, video_index)
        if existing_index <= already_logged_count:
            episode_rows.append(result_table_row(args, existing_result, existing_index))
            uploaded_video_paths.update(result_video_paths(existing_result))
            continue
        wandb_log_result(wandb, args, existing_result, existing_index, episode_rows)
        if not canonical_video_index:
            wandb_log_result_videos_by_test_index(
                wandb,
                args,
                existing_result,
                existing_index,
                uploaded_video_paths,
            )
    return episode_rows, video_index, uploaded_video_paths


def finalize_wandb_videos(wandb, args, video_index, completed_count):
    if not getattr(args, "canonical_video_index", False):
        return
    expected_count = len(_load_episode_test_index_map(args))
    if expected_count == 0 or completed_count != expected_count:
        print(
            "[wandb] skipped canonical video array for incomplete run: "
            f"{completed_count}/{expected_count} episodes"
        )
        return
    wandb_log_indexed_videos(wandb, args, video_index, completed_count)


def save_result_thread_func(args, result_queue, quit_event, progress_queue=None):
    setup_tee(args.log_file)
    ignore_sigint_in_child()
    wandb, run = init_wandb(args)
    result_path = f"{args.dump_location}/{args.exp_name}/result.json"
    completed_count, episode_rows, video_index, existing_results = preload_existing_results(args, result_path)
    progress_history_table = None
    uploaded_video_paths = set()

    if wandb is not None and existing_results:
        episode_rows, video_index, uploaded_video_paths = backfill_existing_results(
            wandb, run, args, existing_results
        )

    try:
        while (
            not quit_event.is_set()
            or not result_queue.empty()
            or not queue_is_empty(progress_queue)
        ):
            new_results = drain_queue_nowait(result_queue)
            progress_snapshots = drain_queue_nowait(progress_queue)
            if len(new_results) > 0:
                append_results_to_disk(result_path, new_results)
                completed_count = handle_new_results(
                    wandb,
                    args,
                    new_results,
                    episode_rows,
                    video_index,
                    completed_count,
                    uploaded_video_paths,
                )
            for snapshot in progress_snapshots:
                progress_history_table = wandb_log_progress(wandb, snapshot, progress_history_table)
            time.sleep(0.1)
    finally:
        if wandb is not None:
            try:
                finalize_wandb_videos(wandb, args, video_index, completed_count)
            finally:
                wandb.finish()
    print("Save result thread finished.")
