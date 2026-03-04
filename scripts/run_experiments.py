#!/usr/bin/env python3
"""
Experiment Runner — Systematic Ablation Study
=============================================
Reads sweep_config.yaml, generates all (config × seed) combinations,
trains and evaluates each run, and saves structured results.

Usage:
    # Run ALL experiments (55 runs)
    python -m scripts.run_experiments

    # Run a single ablation group
    python -m scripts.run_experiments --group A1_policy

    # Run a specific config within a group
    python -m scripts.run_experiments --group A2_observation --config hybrid

    # Dry-run: print all planned runs without executing
    python -m scripts.run_experiments --dry-run

    # Resume: skip runs whose output dirs already exist
    python -m scripts.run_experiments --resume
"""

import os
import sys
import copy
import json
import time
import argparse
import datetime
import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_sweep_config(path="experiments/sweep_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_base_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(config, overrides, baseline):
    """Apply ablation overrides to a deep copy of the base config."""
    cfg = copy.deepcopy(config)

    # Merge baseline first, then ablation-specific overrides
    merged = {**baseline, **overrides}

    # Map sweep keys to their locations in the full config
    key_map = {
        "policy":              ("agent", "policy"),
        "observation_schema":  ("environment", "observation_schema"),
        "qos_lambda":          ("reward", "qos_lambda"),
        "type_switch_cost":    ("reward", "type_switch_cost"),
        "scale_up_cost_per_inst":   ("reward", "scale_up_cost_per_inst"),
        "scale_down_cost_per_inst": ("reward", "scale_down_cost_per_inst"),
        "cooldown_period":     ("environment", "cooldown_period"),
        "num_oai_instances":   ("environment", "num_oai_instances"),
        "performance_threshold": ("environment", "performance_threshold"),
    }

    for key, value in merged.items():
        if key in key_map:
            section, param = key_map[key]
            cfg[section][param] = value
        elif key == "name":
            continue  # metadata, not a config value
        else:
            print(f"  ⚠ Unknown override key: {key}")

    return cfg


def build_run_plan(sweep_cfg, group_filter=None, config_filter=None):
    """
    Build the full list of (group, config_name, overrides, seed) tuples.
    Returns a list of dicts with all run metadata.
    """
    baseline = sweep_cfg["baseline"]
    seeds = sweep_cfg["seeds"]
    ablations = sweep_cfg["ablations"]

    runs = []
    for group_name, group in ablations.items():
        if group_filter and group_name != group_filter:
            continue

        for cfg_entry in group["configs"]:
            cfg_name = cfg_entry["name"]
            if config_filter and cfg_name != config_filter:
                continue

            for seed in seeds:
                run_id = f"{group_name}/{cfg_name}/seed_{seed}"
                runs.append({
                    "run_id": run_id,
                    "group": group_name,
                    "config_name": cfg_name,
                    "variable": group.get("variable", "unknown"),
                    "description": group.get("description", ""),
                    "overrides": {k: v for k, v in cfg_entry.items() if k != "name"},
                    "seed": seed,
                    "baseline": baseline,
                })

    return runs


def run_single_experiment(run_info, base_config, sweep_cfg, output_root):
    """Train and evaluate a single experiment run."""
    from src import utils, agent, tf_config  # noqa
    from src.environment import ManualCooldownEnv
    from src.callbacks import SecLoggingCallback, ResourceUsageCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback
    import math

    run_id = run_info["run_id"]
    seed = run_info["seed"]

    # Build config with overrides
    config = apply_overrides(base_config, run_info["overrides"], run_info["baseline"])
    config["training"]["seed"] = seed
    config["training"]["total_timesteps"] = sweep_cfg.get("total_timesteps", 300000)

    # Set up output directory
    exp_dir = os.path.join(output_root, run_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Redirect model save paths into experiment dir
    config["paths"]["best_model_save_path"] = exp_dir
    config["paths"]["log_dir"] = os.path.join(exp_dir, "tb_logs")
    config["paths"]["results_dir"] = os.path.join(exp_dir, "eval")
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)

    # Save the exact config used
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Save run metadata
    meta = {
        "run_id": run_id,
        "group": run_info["group"],
        "config_name": run_info["config_name"],
        "variable": run_info["variable"],
        "seed": seed,
        "overrides": run_info["overrides"],
        "started_at": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(exp_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Set seeds for reproducibility
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"  RUN: {run_id}")
    print(f"  Overrides: {run_info['overrides']}")
    print(f"  Seed: {seed}")
    print(f"  Output: {exp_dir}")
    print(f"{'='*60}")

    t0 = time.time()

    try:
        # 1. Load data and create environments
        precomputed_data = utils.load_and_preprocess_data(config)
        train_env, eval_env = utils.create_vectorized_envs(config, precomputed_data)
        train_env = VecMonitor(train_env)
        eval_env = VecMonitor(eval_env)

        # 2. Create agent
        model = agent.create_agent(train_env, config)

        # 3. Callbacks
        traincf = config["training"]
        eval_freq_base = traincf["eval_freq_denom"]
        eval_freq = max(math.ceil(eval_freq_base / train_env.num_envs), 500)

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=exp_dir,
            log_path=exp_dir,
            eval_freq=eval_freq,
            n_eval_episodes=traincf["n_eval_episodes"],
            deterministic=traincf["deterministic_eval"],
            render=False,
        )

        sec_cb = SecLoggingCallback(
            log_dir=exp_dir,
            tb_prefix="train",
            verbose=traincf.get("verbose", 1),
        )

        res_cb = ResourceUsageCallback(
            log_dir=exp_dir,
            tb_prefix="sys",
            log_every_n_steps=traincf.get("log_every_n_steps", 1000),
            verbose=traincf.get("verbose", 1),
        )

        # 4. Train
        model.learn(
            total_timesteps=traincf["total_timesteps"],
            callback=[sec_cb, eval_cb, res_cb],
            progress_bar=False,  # disable per-run progress bar in batch mode
        )

        # 5. Save final model and normalization stats
        final_model_path = os.path.join(exp_dir, "final_model")
        vecnorm_path = os.path.join(exp_dir, "vec_normalize_stats.pkl")
        model.save(final_model_path)
        train_env.save(vecnorm_path)

        train_time = time.time() - t0

        # 6. Quick evaluation metrics (inline, no separate process)
        metrics = _quick_evaluate(model, eval_env, config, precomputed_data)
        metrics["train_time_s"] = round(train_time, 2)
        metrics["status"] = "success"

        # Save metrics
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        meta["finished_at"] = datetime.datetime.now().isoformat()
        meta["status"] = "success"
        meta["train_time_s"] = round(train_time, 2)
        with open(os.path.join(exp_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ✓ Completed in {train_time:.1f}s | "
              f"Mean SEC={metrics.get('mean_sec', 'N/A'):.4f} | "
              f"QoS Viol={metrics.get('qos_violation_rate', 'N/A'):.3f}")

        return metrics

    except Exception as e:
        train_time = time.time() - t0
        error_msg = str(e)
        print(f"  ✗ FAILED after {train_time:.1f}s: {error_msg}")

        meta["finished_at"] = datetime.datetime.now().isoformat()
        meta["status"] = "failed"
        meta["error"] = error_msg
        with open(os.path.join(exp_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return {"status": "failed", "error": error_msg, "train_time_s": round(train_time, 2)}


def _quick_evaluate(model, eval_env, config, precomputed_data):
    """
    Run a single evaluation episode and compute summary metrics.
    Lighter than the full evaluate.py — just returns numbers.
    """
    from src.evaluate import predict_dpdk_from_lookup, predict_oai_from_lookup
    import pandas as pd

    env_cfg = config["environment"]
    reward_cfg = config["reward"]
    thr = env_cfg["performance_threshold"]
    eps = float(reward_cfg.get("sec_eps_mbps", 1e-6))
    K = int(env_cfg.get("num_oai_instances", 1))

    is_recurrent = getattr(model.policy, "is_recurrent", False)

    obs = eval_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    state = None
    episode_start = np.ones((eval_env.num_envs,), dtype=bool)

    results = []
    while True:
        if is_recurrent:
            action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)

        step_out = eval_env.step(action)
        if len(step_out) == 4:
            obs, _, dones, infos = step_out
        else:
            obs, _, terminated, truncated, infos = step_out
            dones = np.logical_or(terminated, truncated)

        if isinstance(obs, tuple):
            obs = obs[0]

        episode_start = np.asarray(dones, dtype=bool).reshape(-1)
        results.append(infos[0].copy())

        if episode_start[0]:
            break

    if not results:
        return {"status": "no_results"}

    df = pd.DataFrame(results)

    # Core metrics
    traffic = pd.to_numeric(df.get("traffic"), errors="coerce").to_numpy()
    power = pd.to_numeric(df.get("power"), errors="coerce").to_numpy()
    perf = pd.to_numeric(df.get("performance"), errors="coerce").to_numpy()

    sec = power / np.maximum(traffic, eps)
    qos_violations = (perf < thr).sum()

    # Switching stats
    type_switches = int((df.get("type_switch_penalty", 0) > 0).sum()) if "type_switch_penalty" in df else 0
    scale_events = int((df.get("scale_penalty", 0) != 0).sum()) if "scale_penalty" in df else 0

    # Baselines (always-DPDK, always-OAI)
    dpdk_lookup = precomputed_data["test"]["dpdk_lookup"]
    oai_lookup = precomputed_data["test"]["oai_lookup"]

    dpdk_pow_arr = np.array([predict_dpdk_from_lookup(dpdk_lookup, float(t))[1] for t in traffic])
    dpdk_sec = dpdk_pow_arr / np.maximum(traffic, eps)

    STEP_HOURS = 0.25
    energy_policy = float(power.sum() * STEP_HOURS)
    energy_dpdk = float(dpdk_pow_arr.sum() * STEP_HOURS)

    metrics = {
        "n_steps": len(df),
        "mean_power": round(float(np.nanmean(power)), 4),
        "mean_sec": round(float(np.nanmean(sec)), 6),
        "mean_performance": round(float(np.nanmean(perf)), 4),
        "qos_violations": int(qos_violations),
        "qos_violation_rate": round(float(qos_violations / len(df)), 4),
        "type_switches": type_switches,
        "scale_events": scale_events,
        "total_energy_wh": round(energy_policy, 4),
        "total_reward": round(float(pd.to_numeric(df.get("reward"), errors="coerce").sum()), 4),
        # Baselines for comparison
        "baseline_dpdk_mean_sec": round(float(np.nanmean(dpdk_sec)), 6),
        "baseline_dpdk_energy_wh": round(energy_dpdk, 4),
        "energy_saved_vs_dpdk_pct": round(100.0 * (energy_dpdk - energy_policy) / max(energy_dpdk, 1e-9), 2),
    }

    # Add per-k OAI baselines
    for k in range(1, K + 1):
        oai_pow_arr = np.array([predict_oai_from_lookup(oai_lookup, float(t), k)[1] for t in traffic])
        energy_oai = float(oai_pow_arr.sum() * STEP_HOURS)
        metrics[f"baseline_{k}xoai_energy_wh"] = round(energy_oai, 4)
        metrics[f"energy_saved_vs_{k}xoai_pct"] = round(
            100.0 * (energy_oai - energy_policy) / max(energy_oai, 1e-9), 2
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--sweep", default="experiments/sweep_config.yaml",
                        help="Path to sweep config YAML")
    parser.add_argument("--base-config", default="config.yaml",
                        help="Path to base project config.yaml")
    parser.add_argument("--output", default="experiments/results",
                        help="Root output directory for all runs")
    parser.add_argument("--group", default=None,
                        help="Run only this ablation group (e.g., A1_policy)")
    parser.add_argument("--config", default=None,
                        help="Run only this config within a group (e.g., lstm)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned runs without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose output dirs already contain metrics.json")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    sweep_cfg = load_sweep_config(args.sweep)
    base_config = load_base_config(args.base_config)

    runs = build_run_plan(sweep_cfg, args.group, args.config)

    print(f"\n{'='*60}")
    print(f"  ABLATION EXPERIMENT SWEEP")
    print(f"  Total planned runs: {len(runs)}")
    print(f"  Output root: {args.output}")
    print(f"{'='*60}")

    # Print summary table
    groups = {}
    for r in runs:
        g = r["group"]
        if g not in groups:
            groups[g] = {"configs": set(), "seeds": set()}
        groups[g]["configs"].add(r["config_name"])
        groups[g]["seeds"].add(r["seed"])

    for g, info in groups.items():
        print(f"  {g}: {len(info['configs'])} configs × {len(info['seeds'])} seeds = {len(info['configs'])*len(info['seeds'])} runs")

    if args.dry_run:
        print(f"\n--- DRY RUN: Planned Runs ---")
        for i, r in enumerate(runs, 1):
            print(f"  [{i:3d}] {r['run_id']}  overrides={r['overrides']}")
        print(f"\nTotal: {len(runs)} runs. Use without --dry-run to execute.")
        return

    # Execute runs
    all_results = []
    skipped = 0
    failed = 0
    succeeded = 0

    for i, run_info in enumerate(runs, 1):
        run_dir = os.path.join(args.output, run_info["run_id"])
        metrics_file = os.path.join(run_dir, "metrics.json")

        # Resume support
        if args.resume and os.path.exists(metrics_file):
            print(f"  [{i}/{len(runs)}] SKIP (exists): {run_info['run_id']}")
            with open(metrics_file, "r") as f:
                existing = json.load(f)
            existing["run_id"] = run_info["run_id"]
            existing["group"] = run_info["group"]
            existing["config_name"] = run_info["config_name"]
            existing["seed"] = run_info["seed"]
            all_results.append(existing)
            skipped += 1
            continue

        print(f"\n  [{i}/{len(runs)}] Running: {run_info['run_id']}")
        metrics = run_single_experiment(run_info, base_config, sweep_cfg, args.output)
        metrics["run_id"] = run_info["run_id"]
        metrics["group"] = run_info["group"]
        metrics["config_name"] = run_info["config_name"]
        metrics["variable"] = run_info["variable"]
        metrics["seed"] = run_info["seed"]
        metrics.update(run_info["overrides"])
        all_results.append(metrics)

        if metrics.get("status") == "success":
            succeeded += 1
        else:
            failed += 1

        # Save running master CSV after every run (crash-safe)
        _save_master_csv(all_results, args.output)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE")
    print(f"  Succeeded: {succeeded} | Failed: {failed} | Skipped: {skipped}")
    print(f"  Master CSV: {os.path.join(args.output, 'all_experiments.csv')}")
    print(f"{'='*60}")


def _save_master_csv(results, output_root):
    """Save/overwrite the master experiments CSV — crash-safe incremental saves."""
    import pandas as pd
    os.makedirs(output_root, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_root, "all_experiments.csv")
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
