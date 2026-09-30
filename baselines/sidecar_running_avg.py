"""Periodically push running-average metrics to an active wandb run's summary.

Usage:
    python baselines/sidecar_running_avg.py \\
        --run liandancoin-university-of-michigan/gs-world-object-token/6xn91lwr \\
        --result_path dump/<exp_name>/result.json [--interval 30]
"""
import argparse
import json
import os
import time

import wandb


METRICS = [
    "success",
    "soft_success",
    "spl",
    "soft_spl",
    "oracle_success",
    "collision",
    "collision_rate",
    "step",
    "sim_duration",
    "episode_duration",
    "distance_to_goal",
    "oracle_navigation_error",
    "path_length",
]


def compute_averages(results):
    summary = {"running_avg/n_episodes": len(results)}
    for m in METRICS:
        vals = [r.get(m) for r in results if r.get(m) is not None]
        if vals:
            summary[f"running_avg/{m}"] = sum(vals) / len(vals)
    n_timeout = sum(1 for r in results if r.get("termination_reason") == "time_out")
    n_success = sum(1 for r in results if r.get("success", 0) > 0)
    summary["running_avg/timeout_rate"] = n_timeout / len(results) if results else 0.0
    summary["running_avg/success_count"] = n_success
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="entity/project/run_id")
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    api = wandb.Api()
    last_n = -1
    while True:
        try:
            if not os.path.exists(args.result_path):
                print(f"[sidecar] waiting for {args.result_path}")
            else:
                with open(args.result_path) as f:
                    results = json.load(f)
                if len(results) != last_n and results:
                    summary = compute_averages(results)
                    run = api.run(args.run)
                    run.summary.update(summary)
                    run.update()
                    last_n = len(results)
                    sr = summary.get("running_avg/success", 0.0)
                    spl = summary.get("running_avg/spl", 0.0)
                    soft_sr = summary.get("running_avg/soft_success", 0.0)
                    soft_spl = summary.get("running_avg/soft_spl", 0.0)
                    print(f"[sidecar] n={len(results)} SR={sr:.3f} SPL={spl:.3f} "
                          f"soft_SR={soft_sr:.3f} soft_SPL={soft_spl:.3f}")
        except Exception as exc:
            print(f"[sidecar] error: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
