import json
import numpy as np
import os
import glob
import argparse

THRESH = 1.6  # success = (distance_to_goal < THRESH)


# ----------------------------
# 1. Load Ground Truth Path Lengths
# ----------------------------
def load_gt_path_lengths(episode_dir):
    """
    Load all episode files and build a map: {episode_label} -> optimal_path_length.
    episode_label is assumed to be f"{scene_id}_{episode_id}".
    """
    gt_map = {}
    # Find all json files
    pattern = os.path.join(episode_dir, "*.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} episode files in {episode_dir}")

    count = 0
    for path in files:
        # skip non-episode files if any
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
                    # Collect path lengths from all goals
                    lens = []
                    for g in goals:
                        pl = g.get("path_length")
                        if pl is not None:
                            lens.append(float(pl))

                    if not lens:
                        continue

                    # Use min path length (distance to closest goal)
                    l_opt = min(lens)

                    # Construct label
                    label = f"{scene_id}_{ep_id}"
                    gt_map[label] = l_opt
                    count += 1

        except Exception as e:
            print(f"Error reading {path}: {e}")

    print(f"Loaded {count} ground truth path lengths.")
    return gt_map


# ----------------------------
# Helpers
# ----------------------------
def get_split_from_episode_label(ep_label: str):
    """Map episode_label prefix to split name."""
    if not isinstance(ep_label, str):
        return None
    s = ep_label.strip().lower()
    if s.startswith("gr"):
        return "indoor"
    if s.startswith("vc"):
        return "outdoor"
    if "innout" in s:
        return "innout"
    return "other"


def get_nav_type_from_episode_label(ep_label: str):
    """
    Rule:
    - episode_label contains '_store' -> place navigation
    - otherwise                      -> object navigation
    """
    if not isinstance(ep_label, str):
        return None
    s = ep_label.strip().lower()
    if "_store" in s:
        return "place"
    return "object"


def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return float(x)
    except:
        return None


def compute_success(r, use_success_field=False):
    if use_success_field:
        s = safe_float(r.get("success"))
        if s is None:
            return None
        return 1.0 if s > 0 else 0.0

    d = safe_float(r.get("distance_to_goal"))
    if d is None:
        return None
    return 1.0 if d < THRESH else 0.0


def load_records(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ["records", "results", "episodes"]:
            if k in data and isinstance(data[k], list):
                return data[k]
        raise ValueError(f"Unrecognized dict format in {json_path}")
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {json_path}")
    return data


def index_by_episode(records):
    idx = {}
    for r in records:
        ep = r.get("episode_label")
        if ep is None:
            continue
        idx[ep] = r
    return idx


def filter_eps(ep_list, split=None, nav_type=None):
    out = []
    for ep in ep_list:
        sp = get_split_from_episode_label(ep)
        nt = get_nav_type_from_episode_label(ep)
        if split is not None and sp != split:
            continue
        if nav_type is not None and nt != nav_type:
            continue
        out.append(ep)
    return out


# ----------------------------
# Metrics Calculation
# ----------------------------
def calc_metrics(ep_list, idx, gt_map, use_success_field=False):
    succs = []
    spls = []

    missing_gt_count = 0

    for ep in ep_list:
        r = idx.get(ep)
        if r is None:
            continue

        s = compute_success(r, use_success_field=use_success_field)
        if s is None:
            continue  # Skip records without distance/success

        # Agent path length
        p = safe_float(r.get("path_length"))
        if p is None:
            p = 0.0  # Should probably skip, but assume 0 movement?

        # Ground truth path length
        l = gt_map.get(ep)

        if l is None:
            # If we don't have GT, we can only compute SPL if success=0 (then SPL=0)
            if s == 0.0:
                spl = 0.0
            else:
                missing_gt_count += 1
                # Treat as missing this data point for SPL?
                # Or count as 0? Typically we drop or warn.
                # Let's drop it from the average for now.
                continue
        else:
            if s == 0.0:
                spl = 0.0
            else:
                denom = max(p, l)
                spl = l / denom if denom > 0 else 1.0

        succs.append(s)
        spls.append(spl)

    if not succs:
        return float("nan"), float("nan"), 0, missing_gt_count

    return float(np.mean(succs)), float(np.mean(spls)), len(succs), missing_gt_count


# ----------------------------
# Main
# ----------------------------
def compare_two_results_on_common_eps(
    path_a, path_b, episode_dir, name_a="A", name_b="B", use_success_field=False
):
    # 1. Load GT
    gt_map = load_gt_path_lengths(episode_dir)

    # 2. Load Results
    rec_a = load_records(path_a)
    rec_b = load_records(path_b)

    idx_a = index_by_episode(rec_a)
    idx_b = index_by_episode(rec_b)

    eps_a = set(idx_a.keys())
    eps_b = set(idx_b.keys())
    common = sorted(eps_a.intersection(eps_b))

    print(f"\n{name_a}: episodes={len(eps_a)}")
    print(f"{name_b}: episodes={len(eps_b)}")
    print(f"COMMON episodes={len(common)}")

    groups = [
        ("indoor", "object"),
        ("indoor", "place"),
        ("indoor", None),
        ("outdoor", "object"),
        ("outdoor", "place"),
        ("outdoor", None),
        ("innout", "object"),
        ("innout", "place"),
        ("innout", None),
        ("all", "object"),
        ("all", "place"),
        ("all", None),
    ]

    print("\n=== Success Rate & SPL (Recalculated) ===")
    print(f"Threshold: < {THRESH}m")

    for sp, nt in groups:
        if sp == "all":
            sub = common
        else:
            sub = filter_eps(common, split=sp, nav_type=None)

        if nt is not None:
            sub = filter_eps(sub, split=None if sp == "all" else sp, nav_type=nt)

        sr_a, spl_a, n_a, m_a = calc_metrics(
            sub, idx_a, gt_map, use_success_field=use_success_field
        )
        sr_b, spl_b, n_b, m_b = calc_metrics(
            sub, idx_b, gt_map, use_success_field=use_success_field
        )

        label = f"{sp:6s} | {('merged' if nt is None else nt):6s}"
        print(
            f"[{label}] n={len(sub):4d} | {name_a}: SR={sr_a * 100:.2f} SPL={spl_a * 100:.2f} (n={n_a}) | {name_b}: SR={sr_b * 100:.2f} SPL={spl_b * 100:.2f} (n={n_b})"
        )
        if m_a > 0 or m_b > 0:
            print(
                f"          (Warning: Missing GT path lengths for successful eps: {name_a}={m_a}, {name_b}={m_b})"
            )


# ----------------------------
# USAGE
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two result files on shared episodes"
    )
    parser.add_argument("result_a")
    parser.add_argument("result_b")
    parser.add_argument("--episode-dir", required=True)
    args = parser.parse_args()
    compare_two_results_on_common_eps(args.result_a, args.result_b, args.episode_dir)
