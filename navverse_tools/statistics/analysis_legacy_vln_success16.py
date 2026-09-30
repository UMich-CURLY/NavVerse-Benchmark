#!/usr/bin/env python3
"""Recompute legacy benchmark tables with the 1.6 m success criterion."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUCCESS_THRESHOLD_M = 1.6
METHOD_ORDER = ("UniNaVid", "InternVLA-N1", "StreamVLN")
PAPER_METHOD_ORDER = (
    "SGImagineNav",
    "PoliFormer",
    "UniNaVid",
    "LongNav-R1",
    "InternVLA-N1",
    "StreamVLN",
)
METHOD_CATEGORY = {
    "SGImagineNav": "Modular",
    "PoliFormer": "RL",
    "UniNaVid": "VLA",
    "LongNav-R1": "VLA-RL",
    "InternVLA-N1": "VLA",
    "StreamVLN": "VLA",
}
BENCHMARK_ORDER = ("Legacy", "VLN")
DOMAIN_ORDER = ("All", "Indoor", "Outdoor", "In2Out")
TASK_ORDER = ("All", "ObjNav", "PlaceNav")
DIFFICULTY_ORDER = ("Easy", "Medium", "Hard")


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "navverse_tools").is_dir() and (candidate / "episodes").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the NavVerse-Benchmark repository root")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label(record: dict[str, Any]) -> str:
    return f"{record['scene_id']}_{record['episode_id']}"


def _domain(scene_id: str) -> str:
    if scene_id.startswith("gr"):
        return "Indoor"
    if scene_id.startswith("vc_"):
        return "Outdoor"
    if scene_id.startswith("innout_"):
        return "In2Out"
    raise ValueError(f"Unknown scene domain: {scene_id}")


def _task(goal: dict[str, Any]) -> str:
    instance = str(goal.get("instance", ""))
    store_prefixes = ("/World/store/", "/World/ground/terrain/store/")
    return "PlaceNav" if instance.startswith(store_prefixes) else "ObjNav"


def _difficulty(shortest_path: float) -> str:
    if shortest_path < 10.0:
        return "Easy"
    if shortest_path < 30.0:
        return "Medium"
    return "Hard"


def _xyz_distance(a: list[Any], b: list[Any]) -> float:
    return sum((float(b[index]) - float(a[index])) ** 2 for index in range(3)) ** 0.5


def _initial_distance_to_goal(episode: dict[str, Any]) -> float:
    start_position = episode["start_position"]
    candidates: list[tuple[float, int]] = []
    for goal_index, goal in enumerate(episode["goals"]):
        reference_path = goal.get("reference_path")
        if not isinstance(reference_path, list) or not reference_path:
            continue
        path_distance_to_goal = 0.0
        for reverse_index in range(len(reference_path)):
            index = len(reference_path) - reverse_index - 1
            point = reference_path[index]
            if reverse_index > 0:
                path_distance_to_goal += _xyz_distance(point, reference_path[index + 1])
            candidates.append(
                (_xyz_distance(start_position, point) + path_distance_to_goal, goal_index)
            )
    if not candidates:
        raise ValueError(f"Episode has no usable reference path: {episode.get('scene_id')}")
    closest_distance, closest_goal_index = min(candidates, key=lambda item: item[0])
    radius = float(episode["goals"][closest_goal_index].get("radius") or 0.0)
    return max(1e-5, closest_distance - radius)


def _load_ground_truth(episode_dir: Path) -> dict[str, dict[str, Any]]:
    ground_truth: dict[str, dict[str, Any]] = {}
    for path in sorted(episode_dir.glob("*.json")):
        payload = _read_json(path)
        episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
        for episode in episodes:
            label = f"{episode['scene_id']}_{episode['episode_id']}"
            if label in ground_truth:
                raise ValueError(f"Duplicate ground-truth label: {label}")
            ground_truth[label] = episode
    if not ground_truth:
        raise ValueError(f"No ground-truth episodes found under {episode_dir}")
    return ground_truth


def _read_manifest(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate labels in {path}")
    return labels


def _fix_records(
    records: list[dict[str, Any]], ground_truth: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixed: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in records:
        label = _label(source)
        if label in seen:
            raise ValueError(f"Duplicate result label: {label}")
        seen.add(label)
        episode = ground_truth.get(label)
        if episode is None:
            raise KeyError(f"Missing ground truth for {label}")
        closest_goal_idx = int(episode["closest_goal_idx"])
        goals = episode["goals"]
        if not 0 <= closest_goal_idx < len(goals):
            raise IndexError(f"Invalid closest_goal_idx for {label}: {closest_goal_idx}")
        goal = goals[closest_goal_idx]
        shortest_path = float(goal["path_length"])
        actual_path = float(source.get("path_length", 0.0))
        distance = float(source["distance_to_goal"])
        success = int(distance < SUCCESS_THRESHOLD_M)
        denominator = max(actual_path, shortest_path)
        spl = success * shortest_path / denominator if denominator > 0 else 0.0
        initial_distance = _initial_distance_to_goal(episode)
        fix2_denominator = max(actual_path, initial_distance)
        spl_fix2 = (
            success * initial_distance / fix2_denominator if fix2_denominator > 0 else 0.0
        )
        soft_success = max(0.0, min(1.0, 1.0 - distance / shortest_path)) if shortest_path > 0 else 0.0
        soft_spl = soft_success * shortest_path / denominator if denominator > 0 else 0.0

        record = dict(source)
        for key in ("success", "spl", "soft_success", "soft_spl"):
            if key in source:
                record[f"{key}_original"] = source[key]
        record.update(
            {
                "success": success,
                "spl": spl,
                "soft_success": soft_success,
                "soft_spl": soft_spl,
                "success_threshold_m": SUCCESS_THRESHOLD_M,
                "gt_shortest_path": shortest_path,
                "fix2_start_distance": initial_distance,
                "spl_fix2": spl_fix2,
                "goal_task": _task(goal),
                "scene_domain": _domain(str(source["scene_id"])),
                "difficulty": _difficulty(shortest_path),
            }
        )
        fixed.append(record)
        audit.append(
            {
                "label": label,
                "old_success": int(bool(source.get("success", 0))),
                "new_success": success,
                "old_spl": float(source.get("spl", 0.0)),
                "new_spl": spl,
            }
        )
    return fixed, audit


def _metric_row(
    records: list[dict[str, Any]], spl_key: str = "spl"
) -> dict[str, Any]:
    count = len(records)
    return {
        "N": count,
        "SR": sum(float(row["success"]) for row in records) / count if count else None,
        "SPL": sum(float(row[spl_key]) for row in records) / count if count else None,
    }


def _filter(
    records: Iterable[dict[str, Any]],
    labels: set[str] | None = None,
    task: str = "All",
    domain: str = "All",
    difficulty: str | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        if labels is not None and _label(record) not in labels:
            continue
        if task != "All" and record["goal_task"] != task:
            continue
        if domain != "All" and record["scene_domain"] != domain:
            continue
        if difficulty is not None and record["difficulty"] != difficulty:
            continue
        selected.append(record)
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = ["| " + " | ".join(map(cell, headers)) + " |"]
    output.append("| " + " | ".join("---" for _ in headers) + " |")
    output.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(output)


def _wide_breakdown(
    rows: list[dict[str, Any]], benchmark: str, axis: str, values: tuple[str, ...]
) -> str:
    headers = ["Task", "Method"]
    for value in values:
        headers.extend([f"{value} N", f"{value} SR (%)", f"{value} SPL (%)"])
    table_rows: list[list[Any]] = []
    lookup = {
        (row["Benchmark"], row["Method"], row["Task"], row[axis]): row
        for row in rows
        if row["Benchmark"] == benchmark
    }
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            output: list[Any] = [task, method]
            for value in values:
                row = lookup.get((benchmark, method, task, value))
                output.extend(["--", "--", "--"] if row is None else [row["N"], _pct(row["SR"]), _pct(row["SPL"])])
            table_rows.append(output)
    return _markdown_table(headers, table_rows)


def _paper_methods(task: str) -> tuple[str, ...]:
    return METHOD_ORDER if task == "VLN" else PAPER_METHOD_ORDER


def _paper_best_values(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    best: dict[tuple[str, str, str], float] = {}
    for row in rows:
        for metric in ("SR", "SPL"):
            value = row[metric]
            if value is None:
                continue
            key = (row["Task"], row["Domain"], metric)
            best[key] = max(best.get(key, float("-inf")), float(value))
    return best


def _paper_metric_cell(
    row: dict[str, Any] | None,
    metric: str,
    best: dict[tuple[str, str, str], float],
    latex: bool = False,
) -> str:
    if row is None or row[metric] is None:
        return "--"
    value = float(row[metric])
    text = _pct(value)
    is_best = abs(value - best[(row["Task"], row["Domain"], metric)]) < 1e-12
    if not is_best:
        return text
    return rf"\textbf{{{text}}}" if latex else f"**{text}**"


def _paper_wide_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Task", "Method", "Category", "N"]
    for domain in DOMAIN_ORDER:
        headers.extend([f"{domain} SR (%)", f"{domain} SPL (%)"])
    lookup = {(row["Task"], row["Method"], row["Domain"]): row for row in rows}
    best = _paper_best_values(rows)
    table_rows: list[list[Any]] = []
    for task in ("ObjNav", "PlaceNav", "VLN"):
        for method in _paper_methods(task):
            all_row = lookup[(task, method, "All")]
            output: list[Any] = [task, method, METHOD_CATEGORY[method], all_row["N"]]
            for domain in DOMAIN_ORDER:
                row = lookup.get((task, method, domain))
                output.extend(
                    [
                        _paper_metric_cell(row, "SR", best),
                        _paper_metric_cell(row, "SPL", best),
                    ]
                )
            table_rows.append(output)
    return _markdown_table(headers, table_rows)


def _paper_latex_table(rows: list[dict[str, Any]]) -> str:
    lookup = {(row["Task"], row["Method"], row["Domain"]): row for row in rows}
    best = _paper_best_values(rows)
    lines = [
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        r"Task & Method & Category & \multicolumn{2}{c}{All Scenes} & \multicolumn{2}{c}{Indoor} & \multicolumn{2}{c}{Outdoor} & \multicolumn{2}{c}{Indoor-to-Outdoor} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r" & & & SR$\uparrow$ & SPL$\uparrow$ & SR$\uparrow$ & SPL$\uparrow$ & SR$\uparrow$ & SPL$\uparrow$ & SR$\uparrow$ & SPL$\uparrow$ \\",
        r"\midrule",
    ]
    for task_index, task in enumerate(("ObjNav", "PlaceNav", "VLN")):
        for method_index, method in enumerate(_paper_methods(task)):
            values = []
            for domain in DOMAIN_ORDER:
                row = lookup.get((task, method, domain))
                values.extend(
                    [
                        _paper_metric_cell(row, "SR", best, latex=True),
                        _paper_metric_cell(row, "SPL", best, latex=True),
                    ]
                )
            task_cell = task if method_index == 0 else ""
            lines.append(
                " & ".join([task_cell, method, METHOD_CATEGORY[method], *values]) + r" \\"
            )
        if task_index < 2:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def generate_statistics(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    repo_root = find_repo_root(repo_root)
    run_root = repo_root / "dump/legacy_internvla_streamvln_7gpu14w_chain_20260805_071857"
    output_dir = output_dir or run_root / "statistics_success16"
    specs = {
        ("Legacy", "UniNaVid"): repo_root / "dump/benchmark_uninavid_20260521_102426_best/result.json",
        ("Legacy", "InternVLA-N1"): run_root / "internvla_n1_legacy_test_objnav/result.json",
        ("Legacy", "StreamVLN"): run_root / "streamvln_legacy_test_objnav/result.json",
        ("VLN", "UniNaVid"): repo_root / "dump2/benchmark_uninavid_vln_20260527_095700/result.json",
        ("VLN", "InternVLA-N1"): run_root / "internvla_n1_legacy_vln/result.json",
        ("VLN", "StreamVLN"): run_root / "streamvln_legacy_vln/result.json",
    }
    paper_legacy_specs = {
        "SGImagineNav": repo_root / "dump/sgvln_may24/result.json",
        "PoliFormer": repo_root / "dump/benchmark_poliformer_gpu0123_20260521_210923/result.json",
        "LongNav-R1": repo_root / "dump/longnav_auto_20260522_101625/result.json",
    }
    missing = [
        str(path)
        for path in (*specs.values(), *paper_legacy_specs.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing result files:\n" + "\n".join(missing))

    episode_root = repo_root / "navverse_data" / "episodes"
    ground_truth = _load_ground_truth(episode_root / "legacy")
    manifests = {
        "Legacy": _read_manifest(episode_root / "benchmark450_set.txt"),
        "VLN": _read_manifest(episode_root / "legacy_vln_set.txt"),
    }
    manifest_sets = {key: set(value) for key, value in manifests.items()}
    datasets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    audits: dict[tuple[str, str], list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for (benchmark, method), path in specs.items():
        raw = _read_json(path)
        if not isinstance(raw, list):
            raise TypeError(f"Expected a list in {path}")
        fixed, audit = _fix_records(raw, ground_truth)
        labels = {_label(row) for row in fixed}
        extras = sorted(labels - manifest_sets[benchmark])
        if extras:
            raise ValueError(f"{path} has labels outside {benchmark} manifest: {extras[:5]}")
        datasets[(benchmark, method)] = fixed
        audits[(benchmark, method)] = audit
        fixed_path = output_dir / "fixed_results" / f"{benchmark.lower()}_{method.lower().replace('-', '_')}_result_success16.json"
        _write_json(fixed_path, fixed)
        sources.append(
            {
                "benchmark": benchmark,
                "method": method,
                "path": str(path.relative_to(repo_root)),
                "sha256": _sha256(path),
                "result_count": len(fixed),
                "manifest_count": len(manifests[benchmark]),
                "missing_manifest_labels": len(manifest_sets[benchmark] - labels),
                "fixed_result": str(fixed_path.relative_to(repo_root)),
            }
        )

    paper_datasets = dict(datasets)
    paper_sources: list[dict[str, Any]] = []
    for method, path in paper_legacy_specs.items():
        raw = _read_json(path)
        if not isinstance(raw, list):
            raise TypeError(f"Expected a list in {path}")
        fixed, _ = _fix_records(raw, ground_truth)
        labels = {_label(row) for row in fixed}
        if labels != manifest_sets["Legacy"]:
            raise ValueError(
                f"{path} does not exactly match the Legacy manifest: "
                f"missing={len(manifest_sets['Legacy'] - labels)}, "
                f"extra={len(labels - manifest_sets['Legacy'])}"
            )
        paper_datasets[("Legacy", method)] = fixed
        fixed_path = (
            output_dir
            / "fixed_results"
            / f"legacy_{method.lower().replace('-', '_')}_result_success16.json"
        )
        _write_json(fixed_path, fixed)
        paper_sources.append(
            {
                "benchmark": "Legacy",
                "method": method,
                "path": str(path.relative_to(repo_root)),
                "sha256": _sha256(path),
                "result_count": len(fixed),
                "fixed_result": str(fixed_path.relative_to(repo_root)),
            }
        )

    fair_labels: dict[str, set[str]] = {}
    for benchmark in BENCHMARK_ORDER:
        sets = [
            {_label(row) for row in datasets[(benchmark, method)]}
            for method in METHOD_ORDER
        ]
        fair_labels[benchmark] = set.intersection(*sets)
        if not fair_labels[benchmark]:
            raise ValueError(f"No common labels for {benchmark}")

    overall_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_ORDER:
        for method in METHOD_ORDER:
            records = datasets[(benchmark, method)]
            fair = _filter(records, labels=fair_labels[benchmark])
            overall_rows.append({"Benchmark": benchmark, "Method": method, **_metric_row(fair)})
            native_rows.append(
                {
                    "Benchmark": benchmark,
                    "Method": method,
                    "N": len(records),
                    "Manifest N": len(manifests[benchmark]),
                    "Missing": len(manifest_sets[benchmark] - {_label(row) for row in records}),
                    **{key: value for key, value in _metric_row(records).items() if key != "N"},
                }
            )
            audit = audits[(benchmark, method)]
            audit_rows.append(
                {
                    "Benchmark": benchmark,
                    "Method": method,
                    "N": len(audit),
                    "Old successes": sum(row["old_success"] for row in audit),
                    "New successes": sum(row["new_success"] for row in audit),
                    "Changed": sum(row["old_success"] != row["new_success"] for row in audit),
                    "Old SR": sum(row["old_success"] for row in audit) / len(audit),
                    "New SR": sum(row["new_success"] for row in audit) / len(audit),
                    "Old SPL": sum(row["old_spl"] for row in audit) / len(audit),
                    "New SPL": sum(row["new_spl"] for row in audit) / len(audit),
                }
            )
            for task in TASK_ORDER:
                for domain in DOMAIN_ORDER:
                    subset = _filter(fair, task=task, domain=domain)
                    if subset:
                        breakdown_rows.append(
                            {"Benchmark": benchmark, "Method": method, "Task": task, "Domain": domain, **_metric_row(subset)}
                        )

    paper_rows: list[dict[str, Any]] = []
    for task in ("ObjNav", "PlaceNav", "VLN"):
        benchmark = "VLN" if task == "VLN" else "Legacy"
        for method in _paper_methods(task):
            records = paper_datasets[(benchmark, method)]
            task_records = records if task == "VLN" else _filter(records, task=task)
            for domain in DOMAIN_ORDER:
                subset = _filter(task_records, domain=domain)
                if subset:
                    paper_rows.append(
                        {
                            "Task": task,
                            "Method": method,
                            "Category": METHOD_CATEGORY[method],
                            "Domain": domain,
                            **_metric_row(subset, spl_key="spl_fix2"),
                        }
                    )
                for difficulty in DIFFICULTY_ORDER:
                    subset = _filter(fair, task=task, difficulty=difficulty)
                    if subset:
                        difficulty_rows.append(
                            {
                                "Benchmark": benchmark,
                                "Method": method,
                                "Task": task,
                                "Difficulty": difficulty,
                                **_metric_row(subset),
                            }
                        )

    _write_csv(output_dir / "overall_common_episodes.csv", overall_rows)
    _write_csv(output_dir / "native_coverage.csv", native_rows)
    _write_csv(output_dir / "task_domain_common_episodes.csv", breakdown_rows)
    _write_csv(output_dir / "difficulty_common_episodes.csv", difficulty_rows)
    _write_csv(output_dir / "success_change_audit.csv", audit_rows)
    _write_csv(output_dir / "paper_task_scene_fix2.csv", paper_rows)

    paper_table = _paper_wide_table(paper_rows)
    paper_content = [
        "# Paper-Style Task and Scene Comparison",
        "",
        "Success is `distance_to_goal < 1.6 m`. SPL follows the historical `result_fix2.json` convention: `success * start_distance / max(path_length, start_distance)`.",
        "",
        paper_table,
    ]
    (output_dir / "paper_task_scene_fix2.md").write_text(
        "\n".join(paper_content) + "\n", encoding="utf-8"
    )
    (output_dir / "paper_task_scene_fix2.tex").write_text(
        _paper_latex_table(paper_rows), encoding="utf-8"
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "success_definition": f"distance_to_goal < {SUCCESS_THRESHOLD_M}",
        "spl_definition": "success * gt_shortest_path / max(path_length, gt_shortest_path)",
        "difficulty_definition": {"Easy": "L* < 10m", "Medium": "10m <= L* < 30m", "Hard": "L* >= 30m"},
        "fair_episode_counts": {key: len(value) for key, value in fair_labels.items()},
        "sources": sources,
        "paper_legacy_sources": paper_sources,
        "overall_common_episodes": overall_rows,
        "native_coverage": native_rows,
        "task_domain_common_episodes": breakdown_rows,
        "difficulty_common_episodes": difficulty_rows,
        "success_change_audit": audit_rows,
        "paper_task_scene_fix2": paper_rows,
    }
    _write_json(output_dir / "metrics.json", report)

    content = [
        "# Legacy and VLN Statistics with the 1.6 m Success Standard",
        "",
        "Success is strictly `distance_to_goal < 1.6 m`. SPL is recomputed as `success * L* / max(path_length, L*)`, where `L*` is the selected legacy goal's ground-truth shortest path.",
        "",
        "Both primary comparisons use the 450 episode labels shared by all three methods and exactly matching their corresponding legacy manifests.",
        "",
        "## Paper-Style ObjNav / PlaceNav / VLN Comparison",
        "",
        "This table follows the historical `result_fix2.json` SPL convention used by the May 25 paper table. ObjNav and PlaceNav are split from each method's 450-episode Legacy run; VLN uses the three methods with dedicated 450-episode VLN runs.",
        "",
        paper_table,
        "",
        "## Fair Overall Comparison",
        "",
        _markdown_table(
            ["Benchmark", "Method", "N", "SR (%)", "SPL (%)"],
            [[row["Benchmark"], row["Method"], row["N"], _pct(row["SR"]), _pct(row["SPL"])] for row in overall_rows],
        ),
        "",
        "## Native Coverage and Metrics",
        "",
        _markdown_table(
            ["Benchmark", "Method", "N", "Manifest N", "Missing", "SR (%)", "SPL (%)"],
            [[row["Benchmark"], row["Method"], row["N"], row["Manifest N"], row["Missing"], _pct(row["SR"]), _pct(row["SPL"])] for row in native_rows],
        ),
    ]
    for benchmark in BENCHMARK_ORDER:
        content.extend(
            [
                "",
                f"## {benchmark}: Task and Scene Domain (Common Episodes)",
                "",
                _wide_breakdown(breakdown_rows, benchmark, "Domain", DOMAIN_ORDER),
                "",
                f"## {benchmark}: Difficulty (Common Episodes)",
                "",
                _wide_breakdown(difficulty_rows, benchmark, "Difficulty", DIFFICULTY_ORDER),
            ]
        )
    content.extend(
        [
            "",
            "## Success-Standard Change Audit (Native Coverage)",
            "",
            _markdown_table(
                ["Benchmark", "Method", "N", "Old successes", "New successes", "Changed", "Old SR (%)", "New SR (%)", "Old SPL (%)", "New SPL (%)"],
                [
                    [
                        row["Benchmark"], row["Method"], row["N"], row["Old successes"], row["New successes"], row["Changed"],
                        _pct(row["Old SR"]), _pct(row["New SR"]), _pct(row["Old SPL"]), _pct(row["New SPL"]),
                    ]
                    for row in audit_rows
                ],
            ),
        ]
    )
    (output_dir / "tables.md").write_text("\n".join(content) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = generate_statistics()
    print(json.dumps({"fair_episode_counts": result["fair_episode_counts"]}, indent=2))
