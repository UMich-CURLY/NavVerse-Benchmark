import json
import numpy as np
import os
import glob
import argparse

THRESH = 1.6


def load_gt_path_lengths(episode_dir):
    gt_map = {}
    pattern = os.path.join(episode_dir, "*.json")
    files = glob.glob(pattern)
    for path in files:
        if "test_generator" in path:
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for ep in data:
                scene_id = ep.get("scene_id")
                ep_id = ep.get("episode_id")
                goals = ep.get("goals")
                if scene_id is not None and ep_id is not None and goals:
                    lens = [
                        float(g["path_length"])
                        for g in goals
                        if g.get("path_length") is not None
                    ]
                    if not lens:
                        continue
                    l_opt = min(lens)
                    label = f"{scene_id}_{ep_id}"
                    gt_map[label] = l_opt
        except:
            pass
    return gt_map


def get_nav_type(ep_label):
    if not isinstance(ep_label, str):
        return "object"
    return "place" if "_store" in ep_label.lower() else "object"


def get_difficulty(length):
    if length < 10:
        return "Easy"
    if length <= 30:
        return "Medium"
    return "Hard"


def safe_float(x):
    try:
        return float(x)
    except:
        return None


def compute_metrics(records, gt_map, filter_func=None):
    # filter_func(ep_label, length) -> bool
    succs = []
    spls = []

    for r in records:
        ep = r.get("episode_label")
        if not ep:
            continue

        l_opt = gt_map.get(ep)
        if filter_func and (l_opt is None or not filter_func(ep, l_opt)):
            continue

        d = safe_float(r.get("distance_to_goal"))
        if d is None:
            continue

        s = 1.0 if d < THRESH else 0.0

        p = safe_float(r.get("path_length"))
        if p is None:
            p = 0.0

        if s == 0.0:
            spl = 0.0
        elif l_opt is None:
            continue
        else:
            denom = max(p, l_opt)
            spl = l_opt / denom if denom > 0 else 1.0

        succs.append(s)
        spls.append(spl)

    if not succs:
        return 0.0, 0.0
    return np.mean(succs) * 100, np.mean(spls) * 100


def load_file(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            for k in ["records", "results", "episodes"]:
                if isinstance(d.get(k), list):
                    return d[k]
    except:
        pass
    return []


def main(episode_dir, paths):
    gt_map = load_gt_path_lengths(episode_dir)

    # Structure: Task -> Method -> FileKey
    tasks = [
        ("ObjNav", ["PoliFormer", "UniNavid", "SGImgineNav"]),
        ("PlaceNav", ["PoliFormer", "UniNavid", "SGImgineNav"]),
        ("VLN", ["UniNavid"]),  # Special case for VLN file
    ]

    print(r"\begin{table*}[!t]")
    print(r"\centering")
    print(r"\setlength\tabcolsep{3pt}")
    print(
        r"\caption{Comparison of methods across tasks and difficulties. Easy ($<10$m), Medium ($10-30$m), Hard ($>30$m).}"
    )
    print(r"\label{tab:result_task_difficulty}")
    print(r"\begin{small}")
    print(r"\begin{tabular}{lcccccccc}")
    print(r"\toprule")
    print(
        r"\multirow{2}{*}{Task} & \multirow{2}{*}{Method} & \multirow{2}{*}{Category} & \multicolumn{2}{c}{\textbf{Easy}} & \multicolumn{2}{c}{\textbf{Medium}} & \multicolumn{2}{c}{\textbf{Hard}}\\"
    )
    print(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    print(
        r"&&& SR $\uparrow$ & SPL $\uparrow$ & SR $\uparrow$ & SPL $\uparrow$ & SR $\uparrow$ & SPL $\uparrow$ \\"
    )
    print(r"\midrule")

    categories = {"PoliFormer": "RL", "UniNavid": "VLA", "SGImgineNav": "Modular"}

    for task, methods in tasks:
        print(f"\\multirow{{{len(methods)}}}{{*}}{{{task}}}")

        for method in methods:
            if task == "VLN":
                file_key = "VLN_UniNavid"
                cat = "VLA"
            else:
                file_key = method
                cat = categories.get(method, "")

            records = load_file(paths.get(file_key))

            # Filter by task type
            if task == "ObjNav":
                recs = [
                    r
                    for r in records
                    if get_nav_type(r.get("episode_label")) == "object"
                ]
            elif task == "PlaceNav":
                recs = [
                    r
                    for r in records
                    if get_nav_type(r.get("episode_label")) == "place"
                ]
            else:  # VLN
                recs = records  # Assume all records in VLN file are VLN task

            # Compute stats
            # Easy
            sr_easy, spl_easy = compute_metrics(recs, gt_map, lambda ep, l: l < 10)

            # Medium
            sr_med, spl_med = compute_metrics(recs, gt_map, lambda ep, l: 10 <= l <= 30)

            # Hard
            sr_hard, spl_hard = compute_metrics(recs, gt_map, lambda ep, l: l > 30)

            print(f"& {method} & {cat}")
            print(f"& {sr_easy:.2f} & {spl_easy:.2f}")
            print(f"& {sr_med:.2f} & {spl_med:.2f}")
            print(f"& {sr_hard:.2f} & {spl_hard:.2f} \\\\")

        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{small}")
    print(r"\end{table*}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a benchmark LaTeX table")
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--poliformer", required=True)
    parser.add_argument("--uninavid", required=True)
    parser.add_argument("--sg-imagine-nav", required=True)
    parser.add_argument("--vln-uninavid", required=True)
    args = parser.parse_args()
    main(
        args.episode_dir,
        {
            "PoliFormer": args.poliformer,
            "UniNavid": args.uninavid,
            "SGImgineNav": args.sg_imagine_nav,
            "VLN_UniNavid": args.vln_uninavid,
        },
    )
