#!/usr/bin/env python3
"""
Experiment Analysis — Aggregate, Compare, and Generate Paper-Ready Tables/Figures
=================================================================================

Reads the master CSV from run_experiments.py and produces:
  1. Per-ablation summary tables (mean ± std across seeds)
  2. Bar plots comparing configs within each ablation group
  3. Statistical significance tests (paired t-tests)
  4. LaTeX-ready tables for the paper
  5. A single "best config" recommendation

Usage:
    python -m scripts.analyze_experiments                          # full analysis
    python -m scripts.analyze_experiments --group A1_policy        # one group
    python -m scripts.analyze_experiments --latex                  # output LaTeX tables
    python -m scripts.analyze_experiments --metric mean_sec        # rank by specific metric
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Metric definitions ──────────────────────────────────────────────

# Metrics to display and their formatting
METRICS = {
    "mean_sec":              {"label": "Mean SEC (W/Mbps)",   "fmt": ".4f", "lower_better": True},
    "mean_power":            {"label": "Mean Power (W)",       "fmt": ".3f", "lower_better": True},
    "total_energy_wh":       {"label": "Energy (Wh)",          "fmt": ".2f", "lower_better": True},
    "qos_violation_rate":    {"label": "QoS Viol. Rate",       "fmt": ".3f", "lower_better": True},
    "type_switches":         {"label": "Type Switches",        "fmt": ".0f", "lower_better": True},
    "total_reward":          {"label": "Total Reward",         "fmt": ".1f", "lower_better": False},
    "energy_saved_vs_dpdk_pct": {"label": "% Saved vs DPDK",  "fmt": ".1f", "lower_better": False},
    "train_time_s":          {"label": "Train Time (s)",       "fmt": ".0f", "lower_better": True},
}

PRIMARY_METRIC = "mean_sec"  # default ranking metric


def load_results(results_dir):
    """Load the master CSV from the experiment sweep."""
    csv_path = os.path.join(results_dir, "all_experiments.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run experiments first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # Filter out failed runs
    if "status" in df.columns:
        n_failed = (df["status"] != "success").sum()
        if n_failed > 0:
            print(f"⚠ Excluding {n_failed} failed runs")
        df = df[df["status"] == "success"].copy()

    print(f"✓ Loaded {len(df)} successful runs from {csv_path}")
    return df


def summarize_group(df, group_name):
    """
    Compute mean ± std for each config within an ablation group.
    Returns a summary DataFrame.
    """
    group_df = df[df["group"] == group_name]
    if group_df.empty:
        return pd.DataFrame()

    metric_cols = [m for m in METRICS if m in group_df.columns]

    # Aggregate across seeds
    agg_funcs = {}
    for m in metric_cols:
        agg_funcs[m] = ["mean", "std", "count"]

    summary = group_df.groupby("config_name").agg(agg_funcs)

    # Flatten multi-level columns
    rows = []
    for config_name in summary.index:
        row = {"config": config_name}
        for m in metric_cols:
            mean_val = summary.loc[config_name, (m, "mean")]
            std_val = summary.loc[config_name, (m, "std")]
            count = summary.loc[config_name, (m, "count")]
            row[f"{m}_mean"] = mean_val
            row[f"{m}_std"] = std_val
            row[f"{m}_str"] = f"{mean_val:{METRICS[m]['fmt']}} ± {std_val:{METRICS[m]['fmt']}}"
            row["n_seeds"] = int(count)
        rows.append(row)

    return pd.DataFrame(rows)


def print_group_table(summary_df, group_name, variable=""):
    """Print a formatted comparison table for one ablation group."""
    if summary_df.empty:
        print(f"\n  No results for {group_name}")
        return

    metric_cols = [m for m in METRICS if f"{m}_str" in summary_df.columns]

    print(f"\n{'='*70}")
    print(f"  {group_name}: {variable}")
    print(f"{'='*70}")

    # Header
    header = f"  {'Config':<25}"
    for m in metric_cols:
        header += f" {METRICS[m]['label']:>20}"
    print(header)
    print("  " + "-" * (25 + 21 * len(metric_cols)))

    # Find best config per metric
    best = {}
    for m in metric_cols:
        if METRICS[m]["lower_better"]:
            best[m] = summary_df[f"{m}_mean"].idxmin()
        else:
            best[m] = summary_df[f"{m}_mean"].idxmax()

    # Rows
    for idx, row in summary_df.iterrows():
        line = f"  {row['config']:<25}"
        for m in metric_cols:
            val = row[f"{m}_str"]
            marker = " *" if idx == best.get(m) else "  "
            line += f" {val:>18}{marker}"
        print(line)

    print(f"  (* = best in column, n={summary_df['n_seeds'].iloc[0]} seeds)")


def significance_test(df, group_name, metric=PRIMARY_METRIC):
    """
    Run pairwise Welch's t-tests between all configs within a group.
    Returns a DataFrame of p-values.
    """
    group_df = df[df["group"] == group_name]
    configs = sorted(group_df["config_name"].unique())

    if len(configs) < 2 or metric not in group_df.columns:
        return pd.DataFrame()

    results = []
    for i, c1 in enumerate(configs):
        for j, c2 in enumerate(configs):
            if j <= i:
                continue
            vals1 = group_df[group_df["config_name"] == c1][metric].dropna().values
            vals2 = group_df[group_df["config_name"] == c2][metric].dropna().values

            if len(vals1) < 2 or len(vals2) < 2:
                continue

            t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            results.append({
                "config_1": c1,
                "config_2": c2,
                "mean_1": f"{np.mean(vals1):.4f}",
                "mean_2": f"{np.mean(vals2):.4f}",
                "t_statistic": round(t_stat, 3),
                "p_value": round(p_val, 4),
                "significance": sig,
            })

    return pd.DataFrame(results)


def plot_group_comparison(df, group_name, metric, output_dir):
    """Generate a bar plot comparing configs within a group."""
    group_df = df[df["group"] == group_name]
    if group_df.empty or metric not in group_df.columns:
        return

    configs = sorted(group_df["config_name"].unique())
    means = [group_df[group_df["config_name"] == c][metric].mean() for c in configs]
    stds = [group_df[group_df["config_name"] == c][metric].std() for c in configs]

    fig, ax = plt.subplots(figsize=(max(6, len(configs) * 1.5), 4))
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Highlight best
    meta = METRICS.get(metric, {})
    if meta.get("lower_better", True):
        best_idx = np.argmin(means)
    else:
        best_idx = np.argmax(means)
    bars[best_idx].set_color("green")
    bars[best_idx].set_alpha(1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel(meta.get("label", metric))
    ax.set_title(f"{group_name}: {meta.get('label', metric)}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{group_name}_{metric}.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {fig_path}")


def plot_multi_metric_radar(summary_df, group_name, output_dir):
    """Radar/spider plot comparing configs across multiple metrics."""
    metric_cols = [m for m in ["mean_sec", "mean_power", "qos_violation_rate", "type_switches"]
                   if f"{m}_mean" in summary_df.columns]

    if len(metric_cols) < 3 or summary_df.empty:
        return

    configs = summary_df["config"].tolist()
    n_metrics = len(metric_cols)

    # Normalize each metric to [0, 1] (lower is better → invert so higher = better on radar)
    values = {}
    for m in metric_cols:
        col = summary_df[f"{m}_mean"].values
        mn, mx = col.min(), col.max()
        if mx - mn < 1e-9:
            values[m] = np.ones_like(col) * 0.5
        else:
            normalized = (col - mn) / (mx - mn)
            if METRICS[m]["lower_better"]:
                normalized = 1 - normalized  # invert so bigger = better
            values[m] = normalized

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, config in enumerate(configs):
        vals = [values[m][i] for m in metric_cols] + [values[metric_cols[0]][i]]
        ax.plot(angles, vals, "o-", label=config, linewidth=1.5, markersize=4)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRICS[m]["label"] for m in metric_cols], size=8)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title(f"{group_name} — Multi-metric Comparison", pad=20)

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{group_name}_radar.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {fig_path}")


def generate_latex_table(summary_df, group_name, variable=""):
    """Generate a LaTeX-formatted table for the paper."""
    if summary_df.empty:
        return ""

    metric_cols = [m for m in METRICS if f"{m}_str" in summary_df.columns]
    # Select key metrics for paper table
    paper_metrics = [m for m in ["mean_sec", "total_energy_wh", "qos_violation_rate",
                                  "type_switches", "energy_saved_vs_dpdk_pct"]
                     if m in metric_cols]

    n_cols = len(paper_metrics) + 1
    col_spec = "l" + "r" * len(paper_metrics)

    lines = []
    lines.append(f"% Table: {group_name} — {variable}")
    lines.append(f"\\begin{{table}}[ht]")
    lines.append(f"\\centering")
    lines.append(f"\\caption{{{variable} comparison (mean $\\pm$ std over {summary_df['n_seeds'].iloc[0]} seeds)}}")
    lines.append(f"\\label{{tab:{group_name.lower()}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(f"\\toprule")

    # Header
    header = "Config"
    for m in paper_metrics:
        header += f" & {METRICS[m]['label']}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Find best per metric
    best = {}
    for m in paper_metrics:
        if METRICS[m]["lower_better"]:
            best[m] = summary_df[f"{m}_mean"].idxmin()
        else:
            best[m] = summary_df[f"{m}_mean"].idxmax()

    # Rows
    for idx, row in summary_df.iterrows():
        line = row["config"].replace("_", "\\_")
        for m in paper_metrics:
            val = row[f"{m}_str"]
            if idx == best.get(m):
                val = f"\\textbf{{{val}}}"
            line += f" & ${val}$"
        line += " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def find_best_overall(df, metric=PRIMARY_METRIC):
    """Find the best configuration across all groups."""
    if metric not in df.columns:
        return None

    meta = METRICS.get(metric, {})
    grouped = df.groupby(["group", "config_name"])[metric].agg(["mean", "std", "count"])

    if meta.get("lower_better", True):
        best_idx = grouped["mean"].idxmin()
    else:
        best_idx = grouped["mean"].idxmax()

    best = grouped.loc[best_idx]
    return {
        "group": best_idx[0],
        "config": best_idx[1],
        "mean": best["mean"],
        "std": best["std"],
        "n_seeds": int(best["count"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", default="experiments/results",
                        help="Root directory with all_experiments.csv")
    parser.add_argument("--group", default=None,
                        help="Analyze only this ablation group")
    parser.add_argument("--metric", default=PRIMARY_METRIC,
                        help=f"Primary metric for ranking (default: {PRIMARY_METRIC})")
    parser.add_argument("--latex", action="store_true",
                        help="Output LaTeX tables")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    df = load_results(args.results_dir)
    groups = sorted(df["group"].unique())

    if args.group:
        groups = [g for g in groups if g == args.group]

    latex_tables = []

    for group_name in groups:
        group_df = df[df["group"] == group_name]
        variable = group_df["variable"].iloc[0] if "variable" in group_df.columns else ""

        # 1. Summary table
        summary = summarize_group(df, group_name)
        print_group_table(summary, group_name, variable)

        # 2. Statistical tests
        sig_df = significance_test(df, group_name, args.metric)
        if not sig_df.empty:
            print(f"\n  Statistical significance ({args.metric}):")
            for _, row in sig_df.iterrows():
                print(f"    {row['config_1']} vs {row['config_2']}: "
                      f"p={row['p_value']:.4f} ({row['significance']})")

        # 3. Plots
        if not args.no_plots:
            for metric in ["mean_sec", "total_energy_wh", "qos_violation_rate"]:
                plot_group_comparison(df, group_name, metric, args.results_dir)
            plot_multi_metric_radar(summary, group_name, args.results_dir)

        # 4. LaTeX
        if args.latex:
            latex = generate_latex_table(summary, group_name, variable)
            latex_tables.append(latex)
            print(f"\n{latex}")

        # Save group summary CSV
        if not summary.empty:
            summary_path = os.path.join(args.results_dir, f"{group_name}_summary.csv")
            summary.to_csv(summary_path, index=False)

    # Overall best
    best = find_best_overall(df, args.metric)
    if best:
        meta = METRICS.get(args.metric, {})
        print(f"\n{'='*70}")
        print(f"  BEST OVERALL ({meta.get('label', args.metric)}):")
        print(f"    Group:  {best['group']}")
        print(f"    Config: {best['config']}")
        print(f"    Value:  {best['mean']:{meta.get('fmt', '.4f')}} ± {best['std']:{meta.get('fmt', '.4f')}}")
        print(f"    Seeds:  {best['n_seeds']}")
        print(f"{'='*70}")

    # Save LaTeX to file
    if args.latex and latex_tables:
        latex_path = os.path.join(args.results_dir, "paper_tables.tex")
        with open(latex_path, "w") as f:
            f.write("\n\n".join(latex_tables))
        print(f"\n✓ LaTeX tables saved to {latex_path}")


if __name__ == "__main__":
    main()
