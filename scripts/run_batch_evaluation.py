#!/usr/bin/env python3
"""
Batch Evaluation Runner
=======================
Run src.evaluate over many experiment run directories.

By default, this targets A1 policy observation runs:
    experiments/results/A1_policy_observation

Usage:
    python -m scripts.run_batch_evaluation

    # custom root
    python -m scripts.run_batch_evaluation --runs-root experiments/results/A2_observation

    # stop on first failure
    python -m scripts.run_batch_evaluation --fail-fast

    # headless plotting (useful on servers)
    python -m scripts.run_batch_evaluation --headless
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A1_ROOT = PROJECT_ROOT / "experiments" / "results" / "A1_policy_observation"


def discover_run_dirs(runs_root: Path) -> list[Path]:
    """Find run dirs that are evaluatable by src.evaluate."""
    run_dirs = sorted({p.parent for p in runs_root.rglob("best_model.zip")})

    valid = []
    for run_dir in run_dirs:
        stats = run_dir / "vec_normalize_stats.pkl"
        if stats.exists():
            valid.append(run_dir)
    return valid


def evaluate_run(run_dir: Path, stamp: bool, headless: bool) -> int:
    """Execute one evaluation subprocess. Returns process exit code."""
    cmd = [sys.executable, "-m", "src.evaluate", "--run_dir", str(run_dir)]
    if stamp:
        cmd.append("--stamp")

    env = os.environ.copy()
    if headless:
        env["MPLBACKEND"] = "Agg"

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run src.evaluate for all runs under a directory")
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_A1_ROOT),
        help="Root directory containing run folders (default: A1 policy observation)",
    )
    parser.add_argument(
        "--stamp",
        action="store_true",
        default=True,
        help="Pass --stamp to src.evaluate (default: enabled)",
    )
    parser.add_argument(
        "--no-stamp",
        action="store_false",
        dest="stamp",
        help="Disable timestamped output filenames",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Set MPLBACKEND=Agg for non-interactive plotting",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first failed run",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = (PROJECT_ROOT / runs_root).resolve()

    if not runs_root.exists():
        print(f"ERROR: runs root does not exist: {runs_root}")
        return 2

    run_dirs = discover_run_dirs(runs_root)
    if not run_dirs:
        print(f"No run directories found under {runs_root}")
        return 1

    print(f"Found {len(run_dirs)} run(s) under {runs_root}")

    failures: list[Path] = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(run_dirs)}] Evaluating: {run_dir}")
        print("=" * 80)

        code = evaluate_run(run_dir, stamp=args.stamp, headless=args.headless)
        if code != 0:
            failures.append(run_dir)
            print(f"✗ Failed (exit code {code}): {run_dir}")
            if args.fail_fast:
                break
        else:
            print(f"✓ Completed: {run_dir}")

    print("\n" + "-" * 80)
    print(f"Completed {len(run_dirs) - len(failures)}/{len(run_dirs)} evaluations")
    if failures:
        print("Failed runs:")
        for run_dir in failures:
            print(f"  - {run_dir}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
