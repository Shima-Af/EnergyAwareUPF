#!/usr/bin/env python3
"""
Wilcoxon signed-rank analysis for PPO vs Rule-based and FF vs Recurrent.

HOW TO CUSTOMIZE PLOTS:
  Edit the PLOT STYLE section below, then re-run:
      python3 wilcoxon_analysis.py

Comparisons (x 2 traces = 6 total):
  1. mlp_hybrid   (5 seeds) vs Rule-based   — Main & YouTube
  2. lstm_history (5 seeds) vs Rule-based   — Main & YouTube
  3. mlp_hybrid   vs lstm_history (paired)  — Main & YouTube

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

# =============================================================================
# ██ PLOT STYLE CONFIG — edit this section to change plot appearance ██
# =============================================================================

# ── Font stack (matches Elsevier CAS template / notebook) ────────────────
ELSEVIER_SERIF_STACK = [
    'Times New Roman', 'Times', 'Nimbus Roman No9 L',
    'Nimbus Roman', 'Liberation Serif', 'STIXGeneral',
    'DejaVu Serif',
]

PAPER_STYLE = {
    "figure.dpi":        130,
    "savefig.dpi":       300,
    "font.family":       "serif",
    "font.serif":        ELSEVIER_SERIF_STACK,
    "font.size":         10,
    "axes.titlesize":    14,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "mathtext.fontset":  "stix",
    "axes.unicode_minus": False,
}

# ── Paper colors (matches notebook PAPER_COLORS) ─────────────────────────
PAPER_COLORS = {
    "Feed-forward PPO":     "#1f77b4",   # blue
    "Recurrent PPO":        "#2ca02c",   # green
    "Rule-based baseline":  "#ff7f0e",   # orange
    "Static USR (k=1)":     "#b8b7b7",   # gray
}

# ── Figure size (width, height) in inches ─────────────────────────────────
FIGSIZE = (15, 4)

# ── Axis labels ───────────────────────────────────────────────────────────
XLABEL_TEXT = "Seed"

# ── Legend ─────────────────────────────────────────────────────────────────
LEGEND_FONTSIZE   = 9
LEGEND_LOC        = "upper center"     # on top of each subplot
LEGEND_NCOL       = 3                  # 3 items side by side
LEGEND_FRAMEALPHA = 0.85

# ── Scatter points ────────────────────────────────────────────────────────
SCATTER_COLOR = "#1f77b4"     # FF-PPO blue (default for A vs baseline)
SCATTER_SIZE  = 100
SCATTER_ALPHA = 0.85

# ── Reference line (baseline) ────────────────────────────────────────────
REF_LINE_COLOR = "#ff7f0e"    # Rule-based orange
REF_LINE_STYLE = "--"
REF_LINE_WIDTH = 1.8

# ── Confidence interval band ─────────────────────────────────────────────
CI_BAND_COLOR = "#1f77b4"
CI_BAND_ALPHA = 0.18

# ── Grid ──────────────────────────────────────────────────────────────────
GRID_ALPHA = 0.3

# ── Subplot titles (metric name, placed above legend) ───────────────────
SUBPLOT_TITLE_FONTSIZE = 11
SUBPLOT_TITLE_PAD = 32        # points above axes top → clears the legend

# ── Output resolution ────────────────────────────────────────────────────
SAVE_DPI = 300

# =============================================================================
# ██ END PLOT STYLE CONFIG — you should not need to edit below this line ██
# =============================================================================

plt.rcParams.update(PAPER_STYLE)

HERE   = Path(__file__).parent
ROOT   = HERE.parent.parent

PPO_CSV  = HERE.parent / "PPO" / "tab_ppo_all.csv"
BASE_CSV = HERE.parent / "Baselines" / "tab_baselines_both_traces.csv"

os.makedirs(HERE, exist_ok=True)

ALPHA   = 0.05
METRICS = [
    ("E_tot", "Total Energy [Wh]"),
    ("SEC",   "Mean SEC [W/Mbps]"),
    ("v",     "QoS Violation Rate"),
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

ppo  = pd.read_csv(PPO_CSV)
base = pd.read_csv(BASE_CSV)

b1 = ppo[ppo["Exp"] == "B1"].copy()
SEEDS = sorted(b1["Seed"].unique())

def get_ppo_values(policy, trace, metric):
    rows = b1[(b1["Config"] == policy) & (b1["Trace"] == trace)].sort_values("Seed")
    return rows["Seed"].tolist(), rows[metric].values

def get_rule_value(trace, metric):
    row = base[(base["Controller"] == "Rule-based") & (base["Trace"] == trace)]
    return float(row[metric].values[0])

# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def wilcoxon_one_sample(values, reference, metric_label):
    diff = values - reference
    try:
        w, p = stats.wilcoxon(diff, zero_method="wilcox")
    except ValueError:
        w, p = np.nan, np.nan

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
        "metric": metric_label, "n": n, "ref": reference,
        "mean": mean_v, "std": std_v, "median": med_v,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "mean_diff": mean_d, "med_diff": med_d, "std_diff": std_d,
        "pct_diff": pct_diff, "n_better": n_better, "n_worse": n_worse,
        "W": w, "p": p,
        "sig": bool(p < ALPHA) if not np.isnan(p) else False,
    }


def wilcoxon_paired(values_a, values_b, label_a, label_b, metric_label):
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
    ref_mean = np.mean(values_b)
    pct_diff = (mean_d / abs(ref_mean)) * 100 if ref_mean != 0 else np.nan

    return {
        "metric": metric_label, "n": n, "ref": ref_mean,
        "mean": np.mean(values_a), "std": np.std(values_a, ddof=1),
        "median": np.median(values_a),
        "ci_lo": np.mean(values_a) - t_crit * (np.std(values_a, ddof=1) / np.sqrt(n)),
        "ci_hi": np.mean(values_a) + t_crit * (np.std(values_a, ddof=1) / np.sqrt(n)),
        "mean_diff": mean_d, "med_diff": med_d, "std_diff": std_d,
        "pct_diff": pct_diff,
        "n_better": int(np.sum(diff < 0)), "n_worse": int(np.sum(diff > 0)),
        "W": w, "p": p,
        "sig": bool(p < ALPHA) if not np.isnan(p) else False,
    }

# ---------------------------------------------------------------------------
# Report & plot writers
# ---------------------------------------------------------------------------

def format_report(title, label_a, label_b, trace, seeds, results_by_metric,
                  values_a_by_metric, values_b_is_scalar):
    buf = io.StringIO()
    sep = "=" * 80

    buf.write(f"{sep}\n{title}\n")
    buf.write(f"Trace: {trace}   |   A: {label_a}   |   B (reference): {label_b}\n")
    buf.write(f"{sep}\n\n")

    for (col, metric_label), r in zip(METRICS, results_by_metric):
        buf.write(f"{'─'*70}\nMETRIC: {metric_label}\n{'─'*70}\n")
        vals_a = values_a_by_metric[col]
        ref    = r["ref"]
        buf.write(f"  Reference ({label_b}): {ref:.6f}\n\n  {label_a} per seed:\n")
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
        buf.write(f"    W statistic : {r['W']}\n    p-value     : {r['p']:.4f}\n")
        buf.write(f"    Significant (α={ALPHA}): {'YES ✓' if r['sig'] else 'NO'}\n")
        buf.write(f"    Note: min achievable p with n=5 is 0.0625\n\n")

    buf.write(f"{sep}\nQUICK-REFERENCE TABLE\n{sep}\n")
    rows = []
    for (col, mlabel), r in zip(METRICS, results_by_metric):
        rows.append({
            "Metric": mlabel,
            f"{label_a} mean": f"{r['mean']:.5f}",
            f"{label_a} std":  f"{r['std']:.5f}",
            f"{label_b}":      f"{r['ref']:.5f}",
            "Δ mean": f"{r['mean_diff']:+.5f} ({r['pct_diff']:+.2f}%)",
            "W": f"{r['W']}", "p": f"{r['p']:.4f}",
            f"Sig α={ALPHA}": "Yes" if r["sig"] else "No",
        })
    buf.write(pd.DataFrame(rows).to_string(index=False))
    buf.write("\n")
    return buf.getvalue()


def make_plot(tag, title, label_a, label_b, trace, seeds,
              results_by_metric, values_a_by_metric):
    seed_nums = [int(s.split("_")[1]) for s in seeds]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    # Pick colors based on which model is label_a
    if "FF" in label_a:
        sc_color = PAPER_COLORS.get("Feed-forward PPO", SCATTER_COLOR)
    elif "Rec" in label_a:
        sc_color = PAPER_COLORS.get("Recurrent PPO", SCATTER_COLOR)
    else:
        sc_color = SCATTER_COLOR

    if "Rule" in label_b:
        ref_color = PAPER_COLORS.get("Rule-based baseline", REF_LINE_COLOR)
    elif "Rec" in label_b:
        ref_color = PAPER_COLORS.get("Recurrent PPO", REF_LINE_COLOR)
    else:
        ref_color = REF_LINE_COLOR

    for ax, (col, metric_label), r in zip(axes, METRICS, results_by_metric):
        vals = values_a_by_metric[col]
        ref  = r["ref"]

        ax.scatter(seed_nums, vals, s=SCATTER_SIZE, zorder=3,
                   color=sc_color, label=label_a, alpha=SCATTER_ALPHA)
        ax.axhline(ref, color=ref_color, linestyle=REF_LINE_STYLE,
                   linewidth=REF_LINE_WIDTH, label=f"{label_b} ({ref:.4f})")
        ax.fill_between(
            [min(seed_nums) - 50, max(seed_nums) + 50],
            r["ci_lo"], r["ci_hi"],
            alpha=CI_BAND_ALPHA, color=sc_color, label="95% CI"
        )
        sig_str = f"p={r['p']:.4f} {'*' if r['sig'] else 'ns'}"
        ax.set_xlabel(XLABEL_TEXT)
        ax.set_ylabel(metric_label)
        ax.set_xticks(seed_nums)
        ax.legend(fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL, loc="lower center",
                  framealpha=LEGEND_FRAMEALPHA, bbox_to_anchor=(0.5, 1.02),
                  borderaxespad=0)
        ax.set_title(f"{metric_label}  ({sig_str})",
                     fontsize=SUBPLOT_TITLE_FONTSIZE, pad=SUBPLOT_TITLE_PAD)
        ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout(rect=[0, 0, 1, 0.90])   # room for legends + titles on top
    path = HERE / f"plot_{tag}.png"
    plt.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    return path


def save_summary_csv(tag, label_a, label_b, trace, seeds,
                     results_by_metric, values_a_by_metric):
    rows = []
    for (col, mlabel), r in zip(METRICS, results_by_metric):
        rows.append({
            "comparison": f"{label_a} vs {label_b}", "trace": trace,
            "metric": mlabel, "n_seeds": r["n"],
            f"{label_a}_mean": round(r["mean"], 6),
            f"{label_a}_std":  round(r["std"], 6),
            f"{label_a}_median": round(r["median"], 6),
            "ci_lo": round(r["ci_lo"], 6), "ci_hi": round(r["ci_hi"], 6),
            f"{label_b}_ref": round(r["ref"], 6),
            "mean_diff": round(r["mean_diff"], 6),
            "median_diff": round(r["med_diff"], 6),
            "pct_diff": round(r["pct_diff"], 4),
            "n_better": r["n_better"], "n_worse": r["n_worse"],
            "W": r["W"], "p_value": round(r["p"], 6),
            f"sig_a{int(ALPHA*100):02d}": r["sig"],
        })
    path = HERE / f"tab_{tag}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

# ---------------------------------------------------------------------------
# Run all comparisons
# ---------------------------------------------------------------------------

COMPARISONS = [
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

    (HERE / f"report_{tag}.txt").write_text(report_str)
    csv_path  = save_summary_csv(tag, label_a, label_b, trace,
                                  seeds_a, results_by_metric, values_a_by_metric)
    plot_path = make_plot(tag, title, label_a, label_b, trace,
                           seeds_a, results_by_metric, values_a_by_metric)

    for (col, mlabel), r in zip(METRICS, results_by_metric):
        sig = "* sig" if r["sig"] else "  ns "
        print(f"  {mlabel:<28}  Δ={r['mean_diff']:+.4f}  W={r['W']}  p={r['p']:.4f}  {sig}")
    print(f"  → report: report_{tag}.txt")
    print(f"  → table:  {csv_path.name}")
    print(f"  → plot:   {plot_path.name}")

print(f"\nDone. All files saved to {HERE}/")
