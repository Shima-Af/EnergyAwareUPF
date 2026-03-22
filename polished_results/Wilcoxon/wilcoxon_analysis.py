#!/usr/bin/env python3
"""
Wilcoxon signed-rank analysis for PPO vs Rule-based and FF vs Recurrent.

Comparisons (× 2 traces = 6 total):
  1. mlp_hybrid   (5 seeds) vs Rule-based   — Main & YouTube
  2. lstm_history (5 seeds) vs Rule-based   — Main & YouTube
  3. mlp_hybrid   vs lstm_history (paired)  — Main & YouTube

Metrics tested per comparison: E_tot (Wh), SEC (W/Mbps), v (violation rate)

Outputs (per comparison):
  report_{tag}.txt          full printed analysis
  tab_{tag}.csv             summary table (W, p, mean, std, CI)
  plot_{tag}.png            scatter plot across seeds
"""

import os, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

HERE   = Path(__file__).parent
ROOT   = HERE.parent.parent

PPO_CSV  = HERE.parent / "PPO" / "tab_ppo_all.csv"
BASE_CSV = HERE.parent / "Baselines" / "tab_baselines_both_traces.csv"

os.makedirs(HERE, exist_ok=True)

ALPHA   = 0.05
METRICS = [
    ("E_tot", "Total Energy (Wh)"),
    ("SEC",   "Mean SEC (W/Mbps)"),
    ("v",     "QoS Violation Rate"),
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

ppo  = pd.read_csv(PPO_CSV)
base = pd.read_csv(BASE_CSV)

# Filter to B1 only (5-seed experiment)
b1 = ppo[ppo["Exp"] == "B1"].copy()

SEEDS = sorted(b1["Seed"].unique())   # ['seed_1024', 'seed_123', ...]

def get_ppo_values(policy, trace, metric):
    rows = b1[(b1["Config"] == policy) & (b1["Trace"] == trace)].sort_values("Seed")
    return rows["Seed"].tolist(), rows[metric].values

def get_rule_value(trace, metric):
    row = base[(base["Controller"] == "Rule-based") & (base["Trace"] == trace)]
    return float(row[metric].values[0])

# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def wilcoxon_one_sample(values, reference, metric_label):
    """One-sample Wilcoxon signed-rank vs a fixed reference."""
    diff = values - reference
    try:
        w, p = stats.wilcoxon(diff, zero_method="wilcox")
    except ValueError:
        w, p = np.nan, np.nan   # all zeros edge case

    n       = len(values)
    mean_v  = np.mean(values)
    std_v   = np.std(values, ddof=1)
    med_v   = np.median(values)
    mean_d  = np.mean(diff)
    std_d   = np.std(diff, ddof=1)
    med_d   = np.median(diff)
    sem_v   = std_v / np.sqrt(n)
    t_crit  = stats.t.ppf(0.975, n - 1)
    ci_lo   = mean_v - t_crit * sem_v
    ci_hi   = mean_v + t_crit * sem_v
    n_better = int(np.sum(diff < 0))
    n_worse  = int(np.sum(diff > 0))
    pct_diff = (mean_d / abs(reference)) * 100 if reference != 0 else np.nan

    return {
        "metric":      metric_label,
        "n":           n,
        "ref":         reference,
        "mean":        mean_v,
        "std":         std_v,
        "median":      med_v,
        "ci_lo":       ci_lo,
        "ci_hi":       ci_hi,
        "mean_diff":   mean_d,
        "med_diff":    med_d,
        "std_diff":    std_d,
        "pct_diff":    pct_diff,
        "n_better":    n_better,
        "n_worse":     n_worse,
        "W":           w,
        "p":           p,
        "sig":         bool(p < ALPHA) if not np.isnan(p) else False,
    }


def wilcoxon_paired(values_a, values_b, label_a, label_b, metric_label):
    """Paired Wilcoxon signed-rank: a - b."""
    diff = values_a - values_b
    try:
        w, p = stats.wilcoxon(diff, zero_method="wilcox")
    except ValueError:
        w, p = np.nan, np.nan

    n        = len(diff)
    mean_d   = np.mean(diff)
    std_d    = np.std(diff, ddof=1)
    med_d    = np.median(diff)
    sem_d    = std_d / np.sqrt(n)
    t_crit   = stats.t.ppf(0.975, n - 1)
    ci_lo    = mean_d - t_crit * sem_d
    ci_hi    = mean_d + t_crit * sem_d
    n_a_better = int(np.sum(diff < 0))
    n_b_better = int(np.sum(diff > 0))
    ref_mean   = np.mean(values_b)
    pct_diff   = (mean_d / abs(ref_mean)) * 100 if ref_mean != 0 else np.nan

    return {
        "metric":      metric_label,
        "n":           n,
        "ref":         ref_mean,
        "mean":        np.mean(values_a),
        "std":         np.std(values_a, ddof=1),
        "median":      np.median(values_a),
        "ci_lo":       np.mean(values_a) - t_crit * (np.std(values_a, ddof=1) / np.sqrt(n)),
        "ci_hi":       np.mean(values_a) + t_crit * (np.std(values_a, ddof=1) / np.sqrt(n)),
        "mean_diff":   mean_d,
        "med_diff":    med_d,
        "std_diff":    std_d,
        "pct_diff":    pct_diff,
        "n_better":    n_a_better,
        "n_worse":     n_b_better,
        "W":           w,
        "p":           p,
        "sig":         bool(p < ALPHA) if not np.isnan(p) else False,
    }

# ---------------------------------------------------------------------------
# Report & plot writers
# ---------------------------------------------------------------------------

def format_report(title, label_a, label_b, trace, seeds, results_by_metric,
                  values_a_by_metric, values_b_is_scalar):
    buf = io.StringIO()
    sep = "=" * 80

    buf.write(f"{sep}\n")
    buf.write(f"{title}\n")
    buf.write(f"Trace: {trace}   |   A: {label_a}   |   B (reference): {label_b}\n")
    buf.write(f"{sep}\n\n")

    for (col, metric_label), r in zip(METRICS, results_by_metric):
        buf.write(f"{'─'*70}\n")
        buf.write(f"METRIC: {metric_label}\n")
        buf.write(f"{'─'*70}\n")

        # per-seed values
        vals_a = values_a_by_metric[col]
        ref    = r["ref"]

        buf.write(f"  Reference ({label_b}): {ref:.6f}\n\n")
        buf.write(f"  {label_a} per seed:\n")
        for s, v in zip(seeds, vals_a):
            diff  = v - ref
            arrow = "▼ BETTER" if diff < 0 else "▲ WORSE "
            buf.write(f"    {s:<12}  {v:11.6f}   diff {diff:+.6f}  {arrow}\n")

        buf.write(f"\n  Summary of {label_a}:\n")
        buf.write(f"    Mean   : {r['mean']:.6f}  ±{r['std']:.6f}\n")
        buf.write(f"    Median : {r['median']:.6f}\n")
        buf.write(f"    95% CI : [{r['ci_lo']:.6f}, {r['ci_hi']:.6f}]\n")

        buf.write(f"\n  Difference ({label_a} − {label_b}):\n")
        buf.write(f"    Mean   : {r['mean_diff']:+.6f}  ({r['pct_diff']:+.2f}%)\n")
        buf.write(f"    Median : {r['med_diff']:+.6f}\n")
        buf.write(f"    n_better (A < B): {r['n_better']}   n_worse (A > B): {r['n_worse']}\n")

        buf.write(f"\n  Wilcoxon signed-rank (H₀: median diff = 0, two-tailed):\n")
        buf.write(f"    W statistic : {r['W']}\n")
        buf.write(f"    p-value     : {r['p']:.4f}\n")
        buf.write(f"    Significant (α={ALPHA}): {'YES ✓' if r['sig'] else 'NO'}\n")
        buf.write(f"    Note: min achievable p with n=5 is 0.0625\n\n")

    buf.write(f"{sep}\n")
    buf.write("QUICK-REFERENCE TABLE\n")
    buf.write(f"{sep}\n")
    rows = []
    for (col, mlabel), r in zip(METRICS, results_by_metric):
        rows.append({
            "Metric":    mlabel,
            f"{label_a} mean": f"{r['mean']:.5f}",
            f"{label_a} std":  f"{r['std']:.5f}",
            f"{label_b}":      f"{r['ref']:.5f}",
            "Δ mean":    f"{r['mean_diff']:+.5f} ({r['pct_diff']:+.2f}%)",
            "W":         f"{r['W']}",
            "p":         f"{r['p']:.4f}",
            f"Sig α={ALPHA}": "Yes" if r["sig"] else "No",
        })
    df_t = pd.DataFrame(rows)
    buf.write(df_t.to_string(index=False))
    buf.write("\n")
    return buf.getvalue()


def make_plot(tag, title, label_a, label_b, trace, seeds,
              results_by_metric, values_a_by_metric):
    seed_nums = [int(s.split("_")[1]) for s in seeds]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{title}\n[{trace} trace]", fontsize=11, fontweight="bold")

    for ax, (col, metric_label), r in zip(axes, METRICS, results_by_metric):
        vals = values_a_by_metric[col]
        ref  = r["ref"]

        ax.scatter(seed_nums, vals, s=100, zorder=3, color="steelblue",
                   label=label_a, alpha=0.85)
        ax.axhline(ref, color="red", linestyle="--", linewidth=1.8,
                   label=f"{label_b} ({ref:.4f})")
        ax.fill_between(
            [min(seed_nums) - 50, max(seed_nums) + 50],
            r["ci_lo"], r["ci_hi"],
            alpha=0.18, color="steelblue", label="95% CI"
        )
        sig_str = f"p={r['p']:.4f} {'*' if r['sig'] else 'ns'}"
        ax.set_title(f"{metric_label}\n{sig_str}", fontsize=10)
        ax.set_xlabel("Seed", fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_xticks(seed_nums)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = HERE / f"plot_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_summary_csv(tag, label_a, label_b, trace, seeds,
                     results_by_metric, values_a_by_metric):
    rows = []
    for (col, mlabel), r in zip(METRICS, results_by_metric):
        rows.append({
            "comparison":  f"{label_a} vs {label_b}",
            "trace":       trace,
            "metric":      mlabel,
            "n_seeds":     r["n"],
            f"{label_a}_mean":   round(r["mean"],  6),
            f"{label_a}_std":    round(r["std"],   6),
            f"{label_a}_median": round(r["median"],6),
            "ci_lo":       round(r["ci_lo"], 6),
            "ci_hi":       round(r["ci_hi"], 6),
            f"{label_b}_ref":    round(r["ref"],   6),
            "mean_diff":   round(r["mean_diff"],  6),
            "median_diff": round(r["med_diff"],   6),
            "pct_diff":    round(r["pct_diff"],   4),
            "n_better":    r["n_better"],
            "n_worse":     r["n_worse"],
            "W":           r["W"],
            "p_value":     round(r["p"], 6),
            f"sig_a{int(ALPHA*100):02d}": r["sig"],
        })
    df = pd.DataFrame(rows)
    path = HERE / f"tab_{tag}.csv"
    df.to_csv(path, index=False)
    return path

# ---------------------------------------------------------------------------
# Run all comparisons
# ---------------------------------------------------------------------------

COMPARISONS = [
    # (tag,  policy_a,       label_a,          label_b,           paired)
    ("mlp_vs_rule_main",     "mlp_hybrid",  "FF-PPO",   "Rule-based", "Main",    False),
    ("mlp_vs_rule_youtube",  "mlp_hybrid",  "FF-PPO",   "Rule-based", "YouTube", False),
    ("lstm_vs_rule_main",    "lstm_history","Rec-PPO",  "Rule-based", "Main",    False),
    ("lstm_vs_rule_youtube", "lstm_history","Rec-PPO",  "Rule-based", "YouTube", False),
    ("mlp_vs_lstm_main",     "mlp_hybrid",  "FF-PPO",   "Rec-PPO",   "Main",    True),
    ("mlp_vs_lstm_youtube",  "mlp_hybrid",  "FF-PPO",   "Rec-PPO",   "YouTube", True),
]

for tag, policy_a, label_a, label_b, trace, paired in COMPARISONS:
    print(f"\nRunning: {tag}")

    seeds_a, _ = get_ppo_values(policy_a, trace, "E_tot")

    values_a_by_metric = {}
    for col, _ in METRICS:
        _, va = get_ppo_values(policy_a, trace, col)
        values_a_by_metric[col] = va

    results_by_metric = []
    for col, mlabel in METRICS:
        va = values_a_by_metric[col]
        if paired:
            _, vb = get_ppo_values("lstm_history", trace, col)
            r = wilcoxon_paired(va, vb, label_a, label_b, mlabel)
        else:
            ref = get_rule_value(trace, col)
            r   = wilcoxon_one_sample(va, ref, mlabel)
        results_by_metric.append(r)

    title = f"{label_a} vs {label_b}"
    report_str = format_report(title, label_a, label_b, trace,
                               seeds_a, results_by_metric, values_a_by_metric,
                               values_b_is_scalar=not paired)

    # Save report
    report_path = HERE / f"report_{tag}.txt"
    report_path.write_text(report_str)

    # Save CSV
    csv_path = save_summary_csv(tag, label_a, label_b, trace,
                                seeds_a, results_by_metric, values_a_by_metric)

    # Save plot
    plot_path = make_plot(tag, title, label_a, label_b, trace,
                          seeds_a, results_by_metric, values_a_by_metric)

    # Print compact summary
    for (col, mlabel), r in zip(METRICS, results_by_metric):
        sig = "* sig" if r["sig"] else "  ns "
        print(f"  {mlabel:<28}  Δ={r['mean_diff']:+.4f}  W={r['W']}  p={r['p']:.4f}  {sig}")

    print(f"  → report: {report_path.name}")
    print(f"  → table:  {csv_path.name}")
    print(f"  → plot:   {plot_path.name}")

print(f"\nDone. All files saved to {HERE}/")
