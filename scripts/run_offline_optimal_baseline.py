#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils
from src.offline_optimal import (
    baseline_style_dataframe,
    brute_force_best_objective,
    compute_offline_optimal,
    solve_offline_optimal_actions,
    summarize_baseline_csv,
    summarize_eval_dataframe,
)


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _extract_timestamp(filename: str) -> str:
    m = re.search(r"(\d{8}-\d{6})", filename)
    return m.group(1) if m else ""


def _latest_eval_csv(run_dir: Path) -> Path:
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing eval directory: {eval_dir}")

    candidates = sorted(eval_dir.glob("evaluation_results*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No evaluation_results*.csv found in: {eval_dir}")

    def key_fn(path: Path):
        ts = _extract_timestamp(path.name)
        return (bool(ts), ts, path.stat().st_mtime)

    return max(candidates, key=key_fn)


def _run_tag(run_dir: Path, explicit_tag: str | None) -> str:
    if explicit_tag:
        return explicit_tag
    return f"{run_dir.parent.name}_{run_dir.name}".replace("/", "_")


def _comparison_row(system: str, source: str, summary: dict[str, Any], run_ref: str | None = None) -> dict[str, Any]:
    return {
        "system": system,
        "source": source,
        "switching_count": int(summary.get("switching_count", 0)),
        "scaling_count": int(summary.get("scaling_count", 0)),
        "avg_qos": float(summary.get("avg_qos", np.nan)),
        "violation_rate": float(summary.get("violation_rate", np.nan)),
        "total_energy_wh": float(summary.get("total_energy_wh", np.nan)),
        "avg_power_w": float(summary.get("avg_power_w", np.nan)),
        "avg_sec": float(summary.get("avg_sec", np.nan)),
        "total_reward": float(summary.get("total_reward", np.nan)),
        "steps": int(summary.get("steps", 0)),
        "run_ref": run_ref,
    }


def _attach_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    offline_row = out[out["system"] == "offline_optimal"]
    if offline_row.empty:
        return out

    off_energy = float(offline_row.iloc[0]["total_energy_wh"])
    off_sec = float(offline_row.iloc[0]["avg_sec"])
    off_reward = float(offline_row.iloc[0]["total_reward"])

    out["energy_gap_vs_offline_wh"] = out["total_energy_wh"] - off_energy
    out["energy_gap_vs_offline_pct"] = 100.0 * (out["total_energy_wh"] - off_energy) / max(off_energy, 1e-12)
    out["avg_sec_gap_vs_offline_pct"] = 100.0 * (out["avg_sec"] - off_sec) / max(off_sec, 1e-12)
    out["reward_gap_vs_offline"] = off_reward - out["total_reward"]
    out.loc[out["total_reward"].isna(), "reward_gap_vs_offline"] = np.nan
    return out


def _sanity_checks(objective_mode: str = "discounted", gamma: float = 0.995) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    rng = np.random.default_rng(42)
    traffic = np.array([90.0, 110.0, 140.0, 125.0, 100.0], dtype=np.float64)
    power = rng.uniform(0.4, 1.0, size=(len(traffic), 3)).astype(np.float64)
    perf = rng.uniform(0.85, 0.98, size=(len(traffic), 3)).astype(np.float64)

    env_cfg_cd0 = {"performance_threshold": 0.90, "cooldown_period": 0}
    rew_cfg = {
        "sec_scale": 100.0,
        "sec_eps_mbps": 1e-6,
        "qos_lambda": 30.0,
        "type_switch_cost": 0.03,
        "scale_up_cost_per_inst": 0.012,
        "scale_down_cost_per_inst": 0.003,
    }

    solved_cd0 = solve_offline_optimal_actions(
        traffic_steps=traffic,
        power_table=power,
        perf_table=perf,
        env_config=env_cfg_cd0,
        reward_config=rew_cfg,
        initial_config=0,
        initial_counter=0,
        objective_mode=objective_mode,
        gamma=gamma,
    )
    brute_cd0 = brute_force_best_objective(
        traffic_steps=traffic,
        power_table=power,
        perf_table=perf,
        env_config=env_cfg_cd0,
        reward_config=rew_cfg,
        initial_config=0,
        initial_counter=0,
        objective_mode=objective_mode,
        gamma=gamma,
    )
    checks["cooldown_zero_matches_bruteforce"] = bool(np.isclose(solved_cd0["objective_value"], brute_cd0, atol=1e-8))

    traffic1 = np.array([50.0, 60.0, 70.0, 65.0], dtype=np.float64)
    power1 = np.array([[0.8], [0.82], [0.84], [0.81]], dtype=np.float64)
    perf1 = np.array([[0.96], [0.95], [0.97], [0.96]], dtype=np.float64)
    env_cfg1 = {"performance_threshold": 0.90, "cooldown_period": 4}
    solved_single = solve_offline_optimal_actions(
        traffic_steps=traffic1,
        power_table=power1,
        perf_table=perf1,
        env_config=env_cfg1,
        reward_config=rew_cfg,
        initial_config=0,
        initial_counter=4,
        objective_mode=objective_mode,
        gamma=gamma,
    )
    checks["single_action_constant_sequence"] = bool(np.all(np.asarray(solved_single["executed_actions"], dtype=int) == 0))

    traffic2 = np.array([70.0, 80.0, 90.0, 100.0], dtype=np.float64)
    pcol = np.array([0.75, 0.76, 0.77, 0.78], dtype=np.float64)
    qcol = np.array([0.95, 0.95, 0.95, 0.95], dtype=np.float64)
    power2 = np.column_stack([pcol, pcol, pcol])
    perf2 = np.column_stack([qcol, qcol, qcol])
    env_cfg2 = {"performance_threshold": 0.90, "cooldown_period": 2}
    rew_cfg2 = dict(rew_cfg)
    rew_cfg2["type_switch_cost"] = 0.05
    rew_cfg2["scale_up_cost_per_inst"] = 0.02
    rew_cfg2["scale_down_cost_per_inst"] = 0.01
    solved_ident = solve_offline_optimal_actions(
        traffic_steps=traffic2,
        power_table=power2,
        perf_table=perf2,
        env_config=env_cfg2,
        reward_config=rew_cfg2,
        initial_config=0,
        initial_counter=2,
        objective_mode=objective_mode,
        gamma=gamma,
    )
    checks["identical_configs_avoid_switching"] = bool(np.all(np.asarray(solved_ident["executed_actions"], dtype=int) == 0))

    checks["all_passed"] = bool(all(bool(v) for v in checks.values()))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute offline-optimal baseline under exact ManualCooldownEnv semantics")
    parser.add_argument("--run-dir", required=True, help="Run directory containing config.yaml and eval outputs")
    parser.add_argument("--ppo-eval-csv", default=None, help="Optional explicit PPO evaluation CSV path")
    parser.add_argument(
        "--rule-csv",
        default="experiments/results/Baselines/baseline_rule_results.csv",
        help="Rule baseline CSV (optional, skipped if missing)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/Baselines",
        help="Output directory for offline-optimal artifacts",
    )
    parser.add_argument("--tag", default=None, help="Optional suffix tag for output file names")
    parser.add_argument(
        "--objective-mode",
        choices=["discounted", "undiscounted"],
        default="discounted",
        help="DP objective type. 'discounted' matches PPO-style discounted return; 'undiscounted' uses plain sum of rewards.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor for discounted objective (defaults to config agent.gamma, then 0.995).",
    )
    args = parser.parse_args()

    run_dir = _resolve_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in run directory: {config_path}")

    config = utils.load_config(str(config_path))
    objective_mode = str(args.objective_mode).strip().lower()
    gamma = float(args.gamma) if args.gamma is not None else float(config.get("agent", {}).get("gamma", 0.995))
    if objective_mode == "discounted" and not 0.0 <= gamma <= 1.0:
        raise ValueError(f"For discounted objective mode, gamma must be in [0, 1], got {gamma}.")

    data_for_env = utils.load_and_preprocess_data(config)
    test_payload = data_for_env["test"]

    offline = compute_offline_optimal(
        env_payload=test_payload,
        env_config=config["environment"],
        reward_config=config["reward"],
        objective_mode=objective_mode,
        gamma=gamma,
    )

    tag = _run_tag(run_dir, args.tag)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    offline_eval_path = output_dir / f"offline_optimal_eval_{tag}.csv"
    offline.evaluation_df.to_csv(offline_eval_path, index=False)

    baseline_df = baseline_style_dataframe(offline.evaluation_df)
    baseline_path = output_dir / f"baseline_offline_optimal_results_{tag}.csv"
    baseline_df.to_csv(baseline_path, index=False)

    sequence_df = pd.DataFrame(
        {
            "t": np.arange(len(offline.requested_actions), dtype=int),
            "requested_action": np.asarray(offline.requested_actions, dtype=int),
            "executed_action": np.asarray(offline.executed_actions, dtype=int),
            "cooldown_blocked": pd.to_numeric(
                offline.evaluation_df.get("cooldown_blocked"), errors="coerce"
            ).fillna(0).astype(bool),
        }
    )
    sequence_path = output_dir / f"baseline_offline_optimal_sequence_{tag}.csv"
    sequence_df.to_csv(sequence_path, index=False)

    summary_row = dict(offline.summary)
    summary_row.update(
        {
            "system": "offline_optimal",
            "source": "DP exact planner",
            "objective_value": float(offline.objective_value),
            "objective_mode": objective_mode,
            "objective_gamma": float(gamma) if objective_mode == "discounted" else np.nan,
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "qos_threshold": float(config["environment"]["performance_threshold"]),
            "cooldown_period": int(config["environment"].get("cooldown_period", 0)),
            "num_oai_instances": int(config["environment"].get("num_oai_instances", 1)),
        }
    )
    summary_path = output_dir / f"baseline_offline_optimal_summary_{tag}.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    ppo_eval_csv = _resolve_path(args.ppo_eval_csv) if args.ppo_eval_csv else _latest_eval_csv(run_dir)
    ppo_df = pd.read_csv(ppo_eval_csv)
    ppo_summary = summarize_eval_dataframe(
        ppo_df,
        performance_threshold=float(config["environment"]["performance_threshold"]),
    )

    rows = [
        _comparison_row(
            system="offline_optimal",
            source="DP exact planner",
            summary=offline.summary,
            run_ref=str(offline_eval_path),
        ),
        _comparison_row(
            system=run_dir.parent.name,
            source="PPO eval CSV",
            summary=ppo_summary,
            run_ref=str(ppo_eval_csv),
        ),
    ]

    rule_csv = _resolve_path(args.rule_csv)
    if rule_csv.exists():
        rule_summary = summarize_baseline_csv(
            rule_csv,
            performance_threshold=float(config["environment"]["performance_threshold"]),
        )
        rows.append(
            _comparison_row(
                system="baseline_rule",
                source="Baseline CSV",
                summary=rule_summary,
                run_ref=str(rule_csv),
            )
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df = _attach_gap_columns(comparison_df)
    comparison_path = output_dir / f"offline_optimal_vs_baselines_{tag}.csv"
    comparison_df.to_csv(comparison_path, index=False)

    sanity = _sanity_checks(objective_mode=objective_mode, gamma=gamma)
    sanity["objective_mode"] = objective_mode
    sanity["objective_gamma"] = float(gamma) if objective_mode == "discounted" else None
    sanity_path = output_dir / f"offline_optimal_sanity_checks_{tag}.json"
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)

    print("=" * 72)
    print("Offline-Optimal Baseline Completed")
    print("=" * 72)
    print(f"Run dir:                 {run_dir}")
    print(f"PPO eval CSV:            {ppo_eval_csv}")
    print(f"Objective mode:          {objective_mode}")
    if objective_mode == "discounted":
        print(f"Objective gamma:         {gamma:.6f}")
    print(f"Offline objective value: {offline.objective_value:.6f}")
    print(f"Offline total reward:    {offline.summary['total_reward']:.6f}")
    print(f"Offline total energy Wh: {offline.summary['total_energy_wh']:.6f}")
    print(f"Offline avg SEC:         {offline.summary['avg_sec']:.6f}")
    print(f"Offline violation rate:  {offline.summary['violation_rate']:.6f}")
    print("Outputs:")
    print(f"  - {offline_eval_path}")
    print(f"  - {baseline_path}")
    print(f"  - {sequence_path}")
    print(f"  - {summary_path}")
    print(f"  - {comparison_path}")
    print(f"  - {sanity_path}")
    print("=" * 72)

    ppo_row = comparison_df[comparison_df["system"] == run_dir.parent.name]
    if not ppo_row.empty:
        p = ppo_row.iloc[0]
        print("PPO gap vs offline-optimal:")
        print(f"  - Energy gap (Wh): {float(p['energy_gap_vs_offline_wh']):.6f}")
        print(f"  - Energy gap (%):  {float(p['energy_gap_vs_offline_pct']):.6f}")
        print(f"  - SEC gap (%):     {float(p['avg_sec_gap_vs_offline_pct']):.6f}")
        if pd.notna(p["reward_gap_vs_offline"]):
            print(f"  - Reward gap:      {float(p['reward_gap_vs_offline']):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
