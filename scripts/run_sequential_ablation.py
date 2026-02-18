#!/usr/bin/env python3
"""
Sequential Ablation Runner
===========================
Runs ablation groups one-at-a-time, automatically locking the best config
from each group into the baseline before proceeding to the next.

This is the rigorous "greedy one-factor-at-a-time" approach for papers.

Usage:
    # Run the full sequential ablation (A1 → A2 → A3 → A4 → A5)
    python scripts/run_sequential_ablation.py

    # Start from a specific group (e.g., if A1 is already done)
    python scripts/run_sequential_ablation.py --start-from A2_observation

    # Dry-run: show the plan without executing
    python scripts/run_sequential_ablation.py --dry-run

    # Use a custom ranking metric (default: mean_sec)
    python scripts/run_sequential_ablation.py --metric total_energy_wh
"""

import os
import sys
import copy
import json
import yaml
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_experiments import (
    load_sweep_config, load_base_config, build_run_plan,
    run_single_experiment, _save_master_csv
)

# Ablation order — this defines the sequence
ABLATION_ORDER = [
    "A1_policy",
    "A2_observation",
    "A3_reward",
    "A4_cooldown",
    "A5_num_oai",
]

# Map from sweep override keys to baseline keys
OVERRIDE_TO_BASELINE = {
    "policy": "policy",
    "observation_schema": "observation_schema",
    "qos_lambda": "qos_lambda",
    "type_switch_cost": "type_switch_cost",
    "cooldown_period": "cooldown_period",
    "num_oai_instances": "num_oai_instances",
}


def find_best_config(results_dir, group_name, metric="mean_sec", lower_better=True):
    """
    Read metrics from completed runs and find the best config in a group.
    Returns (config_name, overrides_dict, mean_value, std_value).
    """
    group_dir = os.path.join(results_dir, group_name)
    if not os.path.isdir(group_dir):
        return None

    # Collect metrics from all runs in this group
    rows = []
    for config_name in os.listdir(group_dir):
        config_dir = os.path.join(group_dir, config_name)
        if not os.path.isdir(config_dir):
            continue
        for seed_dir_name in os.listdir(config_dir):
            seed_dir = os.path.join(config_dir, seed_dir_name)
            metrics_file = os.path.join(seed_dir, "metrics.json")
            if not os.path.isfile(metrics_file):
                continue
            with open(metrics_file, "r") as f:
                m = json.load(f)
            if m.get("status") != "success":
                continue
            m["config_name"] = config_name
            rows.append(m)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    if metric not in df.columns:
        print(f"  ⚠ Metric '{metric}' not found in results")
        return None

    # Aggregate by config: mean across seeds
    summary = df.groupby("config_name")[metric].agg(["mean", "std", "count"]).reset_index()

    if lower_better:
        best_idx = summary["mean"].idxmin()
    else:
        best_idx = summary["mean"].idxmax()

    best_row = summary.loc[best_idx]
    best_config = best_row["config_name"]
    best_mean = best_row["mean"]
    best_std = best_row["std"]
    n_seeds = int(best_row["count"])

    # Retrieve the overrides from the sweep config for this config
    return best_config, best_mean, best_std, n_seeds


def get_overrides_for_config(sweep_cfg, group_name, config_name):
    """Get the override dict for a specific config within a group."""
    group = sweep_cfg["ablations"].get(group_name, {})
    for entry in group.get("configs", []):
        if entry["name"] == config_name:
            return {k: v for k, v in entry.items() if k != "name"}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Sequential ablation runner")
    parser.add_argument("--sweep", default="experiments/sweep_config.yaml")
    parser.add_argument("--base-config", default="config.yaml")
    parser.add_argument("--output", default="experiments/results")
    parser.add_argument("--start-from", default=None,
                        help="Skip groups before this one (e.g., A2_observation)")
    parser.add_argument("--metric", default="mean_sec",
                        help="Metric to select best config (default: mean_sec)")
    parser.add_argument("--higher-better", action="store_true",
                        help="If set, higher metric = better (default: lower = better)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    sweep_cfg = load_sweep_config(args.sweep)
    base_config = load_base_config(args.base_config)
    lower_better = not args.higher_better

    # Current baseline (evolves as we lock winners)
    baseline = copy.deepcopy(sweep_cfg["baseline"])

    # Determine which groups to run
    groups_to_run = list(ABLATION_ORDER)
    if args.start_from:
        try:
            start_idx = groups_to_run.index(args.start_from)
            skipped = groups_to_run[:start_idx]
            groups_to_run = groups_to_run[start_idx:]

            # For skipped groups, load their winners to reconstruct the baseline
            print(f"\n{'='*60}")
            print(f"  RECONSTRUCTING BASELINE FROM PREVIOUS GROUPS")
            print(f"{'='*60}")
            for g in skipped:
                result = find_best_config(args.output, g, args.metric, lower_better)
                if result is None:
                    print(f"  ⚠ {g}: No results found — using original baseline value")
                    continue
                best_config, best_mean, best_std, n_seeds = result
                overrides = get_overrides_for_config(sweep_cfg, g, best_config)
                for key, val in overrides.items():
                    if key in OVERRIDE_TO_BASELINE:
                        baseline[OVERRIDE_TO_BASELINE[key]] = val
                print(f"  ✓ {g}: locked '{best_config}' ({args.metric}={best_mean:.4f}±{best_std:.4f})")

        except ValueError:
            print(f"ERROR: Unknown group '{args.start_from}'. Available: {ABLATION_ORDER}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  SEQUENTIAL ABLATION PLAN")
    print(f"  Metric: {args.metric} ({'lower' if lower_better else 'higher'} is better)")
    print(f"  Groups to run: {' → '.join(groups_to_run)}")
    print(f"  Seeds per config: {len(sweep_cfg['seeds'])}")
    print(f"{'='*60}")
    print(f"  Starting baseline: {baseline}")

    if args.dry_run:
        print(f"\n  DRY RUN — would execute:")
        for g in groups_to_run:
            group = sweep_cfg["ablations"].get(g, {})
            configs = [c["name"] for c in group.get("configs", [])]
            n_runs = len(configs) * len(sweep_cfg["seeds"])
            print(f"    {g}: {configs} × {len(sweep_cfg['seeds'])} seeds = {n_runs} runs")
        total = sum(
            len(sweep_cfg["ablations"].get(g, {}).get("configs", [])) * len(sweep_cfg["seeds"])
            for g in groups_to_run
        )
        print(f"\n  Total runs: {total}")
        return

    # ── Execute groups sequentially ───────────────────────────────
    all_results = []
    decisions_log = []

    for group_name in groups_to_run:
        group = sweep_cfg["ablations"].get(group_name)
        if group is None:
            print(f"\n  ⚠ Group '{group_name}' not found in sweep config, skipping")
            continue

        variable = group.get("variable", "unknown")
        configs = group.get("configs", [])
        n_runs = len(configs) * len(sweep_cfg["seeds"])

        print(f"\n{'='*60}")
        print(f"  GROUP: {group_name} ({variable})")
        print(f"  Baseline: {baseline}")
        print(f"  Configs: {[c['name'] for c in configs]} × {len(sweep_cfg['seeds'])} seeds = {n_runs} runs")
        print(f"{'='*60}")

        # Build and execute runs for this group
        # Temporarily update sweep_cfg baseline for build_run_plan
        sweep_cfg_copy = copy.deepcopy(sweep_cfg)
        sweep_cfg_copy["baseline"] = baseline

        runs = build_run_plan(sweep_cfg_copy, group_filter=group_name)

        for i, run_info in enumerate(runs, 1):
            run_dir = os.path.join(args.output, run_info["run_id"])
            metrics_file = os.path.join(run_dir, "metrics.json")

            # Resume support: skip if already completed
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    existing = json.load(f)
                if existing.get("status") == "success":
                    print(f"  [{i}/{len(runs)}] SKIP (exists): {run_info['run_id']}")
                    existing.update({
                        "run_id": run_info["run_id"],
                        "group": run_info["group"],
                        "config_name": run_info["config_name"],
                        "seed": run_info["seed"],
                    })
                    all_results.append(existing)
                    continue

            print(f"\n  [{i}/{len(runs)}] Running: {run_info['run_id']}")
            metrics = run_single_experiment(run_info, base_config, sweep_cfg_copy, args.output)
            metrics.update({
                "run_id": run_info["run_id"],
                "group": run_info["group"],
                "config_name": run_info["config_name"],
                "variable": run_info["variable"],
                "seed": run_info["seed"],
            })
            metrics.update(run_info["overrides"])
            all_results.append(metrics)

            # Incremental save
            _save_master_csv(all_results, args.output)

        # ── Analyze this group and pick winner ────────────────────
        result = find_best_config(args.output, group_name, args.metric, lower_better)

        if result is None:
            print(f"\n  ✗ No successful results for {group_name} — cannot pick winner")
            continue

        best_config, best_mean, best_std, n_seeds = result
        overrides = get_overrides_for_config(sweep_cfg, group_name, best_config)

        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  WINNER: {best_config:<35} │")
        print(f"  │  {args.metric}: {best_mean:.4f} ± {best_std:.4f} (n={n_seeds})     │")
        print(f"  └─────────────────────────────────────────────┘")

        # Lock winner into baseline
        for key, val in overrides.items():
            if key in OVERRIDE_TO_BASELINE:
                old_val = baseline.get(OVERRIDE_TO_BASELINE[key])
                baseline[OVERRIDE_TO_BASELINE[key]] = val
                print(f"  → Locked: {OVERRIDE_TO_BASELINE[key]} = {val} (was {old_val})")

        decisions_log.append({
            "group": group_name,
            "variable": variable,
            "winner": best_config,
            f"{args.metric}_mean": round(best_mean, 6),
            f"{args.metric}_std": round(best_std, 6),
            "n_seeds": n_seeds,
            "locked_overrides": overrides,
        })

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SEQUENTIAL ABLATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Decision chain:")
    for d in decisions_log:
        print(f"    {d['group']:20s} → {d['winner']:25s} "
              f"({args.metric}={d[f'{args.metric}_mean']:.4f}±{d[f'{args.metric}_std']:.4f})")

    print(f"\n  Final best configuration:")
    for key, val in baseline.items():
        print(f"    {key}: {val}")

    # Save decisions
    decisions_path = os.path.join(args.output, "ablation_decisions.json")
    with open(decisions_path, "w") as f:
        json.dump({
            "metric": args.metric,
            "lower_better": lower_better,
            "decisions": decisions_log,
            "final_baseline": baseline,
        }, f, indent=2)
    print(f"\n  ✓ Decisions saved to {decisions_path}")

    # Save final master CSV
    _save_master_csv(all_results, args.output)
    print(f"  ✓ Master CSV saved to {os.path.join(args.output, 'all_experiments.csv')}")


if __name__ == "__main__":
    main()
