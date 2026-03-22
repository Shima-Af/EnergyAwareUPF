#!/usr/bin/env python3
"""
Wilcoxon signed-rank test comparing PPO (mlp_hybrid) against rule-based baseline.
Uses existing B2 multi-seed evaluation results.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# 1. LOAD DATA
# ============================================================================

# Load B2 per-seed PPO results
b2_csv = Path(__file__).parent.parent / "B2_top2_seeds" / "b2_two_model_per_seed_comparison.csv"
b2_data = pd.read_csv(b2_csv)

# Filter for mlp_hybrid model (preferred for main results)
ppo_df = b2_data[b2_data['model'] == 'mlp_hybrid'].copy()
ppo_df = ppo_df.sort_values('seed')

# Load baseline reference from paper_results_assets
overall_csv = Path(__file__).parent / "tab_overall_results.csv"
overall_data = pd.read_csv(overall_csv)

# Extract rule baseline value
rule_baseline = overall_data[overall_data['Controller'] == 'Rule-based baseline']
if len(rule_baseline) != 1:
    raise ValueError(f"Expected 1 rule baseline row, found {len(rule_baseline)}")
rule_energy = rule_baseline['E_tot'].values[0]

print("=" * 80)
print("WILCOXON SIGNED-RANK TEST: PPO (mlp_hybrid) vs Rule-Based Baseline")
print("=" * 80)
print()

# ============================================================================
# Section A: RAW VALUES
# ============================================================================

print("SECTION A: RAW VALUES")
print("-" * 80)
print(f"\nRule-based baseline: {rule_energy:.4f} Wh")
print(f"  Source: experiments/results/paper_results_assets/tab_overall_results.csv")
print()
print("PPO (mlp_hybrid) by seed (total_energy_wh):")
print()

ppo_values_dict = {}
for idx, row in ppo_df.iterrows():
    seed = row['seed']
    energy = row['total_energy_wh']
    ppo_values_dict[seed] = energy
    print(f"  {seed:>4}: {energy:9.4f} Wh")

ppo_values = np.array(list(ppo_values_dict.values()))

print()
print(f"  N (PPO seeds): {len(ppo_values)}")
print()

# ============================================================================
# 2. COMPUTE WILCOXON TEST (ONE-SAMPLE)
# ============================================================================

# Compute differences: PPO - Rule
differences = ppo_values - rule_energy

print("SECTION B: DIFFERENCES (PPO - Rule)")
print("-" * 80)
print()

for seed, diff in zip(ppo_values_dict.keys(), differences):
    outcome = "BETTER" if diff < 0 else "WORSE"
    print(f"  {seed:>4}: {diff:+8.4f} Wh  ({outcome})")

print()
print(f"  Negative (PPO better):     {np.sum(differences < 0)}")
print(f"  Positive (PPO worse):      {np.sum(differences > 0)}")
print(f"  Zero (tie):                {np.sum(differences == 0)}")
print()

# Run Wilcoxon (one-sample, H0: median diff = 0)
w_stat, w_pval = stats.wilcoxon(differences, zero_method='wilcox')

# Also compute Bonferroni corrected threshold (since also testing other metrics)
alpha_bonf = 0.05 / 3  # 3 metrics: energy, SEC, QoS

print("SECTION C: WILCOXON SIGNED-RANK TEST")
print("-" * 80)
print()
print(f"  Test statistic (W):        {w_stat:.1f}")
print(f"  P-value (two-tailed):      {w_pval:.4f}")
print(f"  α = 0.05:                  Significant: {w_pval < 0.05}")
print(f"  α = 0.05 (Bonferroni 3x):  Significant: {w_pval < alpha_bonf}")
print()

# ============================================================================
# 3. COMPUTE SUPPORTING STATS
# ============================================================================

print("SECTION D: SUPPORTING STATISTICS")
print("-" * 80)
print()

# PPO statistics
ppo_mean = np.mean(ppo_values)
ppo_std = np.std(ppo_values, ddof=1)
ppo_median = np.median(ppo_values)
ppo_sem = ppo_std / np.sqrt(len(ppo_values))

print("PPO (mlp_hybrid) statistics:")
print(f"  Mean:                      {ppo_mean:9.4f} Wh")
print(f"  Std (sample, ddof=1):      {ppo_std:9.4f} Wh")
print(f"  Median:                    {ppo_median:9.4f} Wh")
print(f"  Min:                       {np.min(ppo_values):9.4f} Wh")
print(f"  Max:                       {np.max(ppo_values):9.4f} Wh")
print()

# 95% CI using t-distribution (small sample)
t_crit = stats.t.ppf(0.975, len(ppo_values) - 1)
ci_lower = ppo_mean - t_crit * ppo_sem
ci_upper = ppo_mean + t_crit * ppo_sem

print(f"  95% CI (t-dist, n={len(ppo_values)}):")
print(f"    [{ci_lower:9.4f}, {ci_upper:9.4f}] Wh")
print()

# Difference statistics
diff_mean = np.mean(differences)
diff_std = np.std(differences, ddof=1)
diff_median = np.median(differences)
diff_sem = diff_std / np.sqrt(len(differences))

print("Difference statistics (PPO - Rule):")
print(f"  Mean:                      {diff_mean:+9.4f} Wh")
print(f"  Std (sample, ddof=1):      {diff_std:9.4f} Wh")
print(f"  Median:                    {diff_median:+9.4f} Wh")
print()

# 95% CI for median difference (bootstrap-like interpretation)
t_crit_diff = stats.t.ppf(0.975, len(differences) - 1)
ci_diff_lower = diff_mean - t_crit_diff * diff_sem
ci_diff_upper = diff_mean + t_crit_diff * diff_sem

print(f"  95% CI of mean difference:")
print(f"    [{ci_diff_lower:+9.4f}, {ci_diff_upper:+9.4f}] Wh")
print()

# Comparison to baseline
print("Rule-based baseline value:")
print(f"  {rule_energy:9.4f} Wh")
print()

energy_saving_percent = (rule_energy - ppo_mean) / rule_energy * 100
print(f"PPO vs Rule (mean comparison):")
print(f"  PPO mean energy: {ppo_mean:9.4f} Wh")
print(f"  Rule energy:     {rule_energy:9.4f} Wh")
print(f"  Difference:      {diff_mean:+9.4f} Wh ({energy_saving_percent:+.2f}%)")
print()

# ============================================================================
# 4. ALSO TEST SECONDARY METRICS FOR COMPLETENESS
# ============================================================================

print("SECTION E: SECONDARY METRICS")
print("-" * 80)
print()

# Average SEC
sec_rule = overall_data[overall_data['Controller'] == 'Rule-based baseline']['avg_SEC'].values[0]
sec_ppo = ppo_df['average_sec'].values

sec_diff = sec_ppo - sec_rule
sec_w_stat, sec_w_pval = stats.wilcoxon(sec_diff, zero_method='wilcox')

print("Average SEC (Spectral Efficiency * Cooldown penalty):")
print(f"  Rule baseline:   {sec_rule:.6f}")
print(f"  PPO mean:        {np.mean(sec_ppo):.6f}")
print(f"  Wilcoxon p-val:  {sec_w_pval:.4f}")
print(f"  Significant:     {sec_w_pval < 0.05}")
print()

# QoS violation rate
qos_rule = overall_data[overall_data['Controller'] == 'Rule-based baseline']['v'].values[0]
qos_ppo = ppo_df['qos_violation_rate'].values

qos_diff = qos_ppo - qos_rule
qos_w_stat, qos_w_pval = stats.wilcoxon(qos_diff, zero_method='wilcox')

print("QoS violation rate:")
print(f"  Rule baseline:   {qos_rule:.6f}")
print(f"  PPO mean:        {np.mean(qos_ppo):.6f}")
print(f"  Wilcoxon p-val:  {qos_w_pval:.4f}")
print(f"  Significant:     {qos_w_pval < 0.05}")
print()

# ============================================================================
# 5. GENERATE PLOT
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Total Energy
ax = axes[0]
seeds_list = list(ppo_values_dict.keys())
seed_nums = [int(s.split('_')[1]) for s in seeds_list]
energies_list = list(ppo_values_dict.values())

ax.scatter(seed_nums, energies_list, s=100, alpha=0.7, color='steelblue', label='PPO (mlp_hybrid)')
ax.axhline(rule_energy, color='red', linestyle='--', linewidth=2, label=f'Rule baseline ({rule_energy:.2f} Wh)')
ax.fill_between([min(seed_nums)-10, max(seed_nums)+10], ci_lower, ci_upper, alpha=0.2, color='steelblue', label='PPO 95% CI')

ax.set_xlabel('Seed', fontsize=11)
ax.set_ylabel('Total Energy (Wh)', fontsize=11)
ax.set_title('Total Energy: PPO vs Rule Baseline', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(seed_nums)

# Plot 2: Average SEC
ax = axes[1]
ax.scatter(seed_nums, sec_ppo, s=100, alpha=0.7, color='steelblue', label='PPO (mlp_hybrid)')
ax.axhline(sec_rule, color='red', linestyle='--', linewidth=2, label=f'Rule baseline ({sec_rule:.5f})')

ax.set_xlabel('Seed', fontsize=11)
ax.set_ylabel('Average SEC', fontsize=11)
ax.set_title('Average SEC: PPO vs Rule Baseline', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(seed_nums)

# Plot 3: QoS Violation Rate
ax = axes[2]
ax.scatter(seed_nums, qos_ppo, s=100, alpha=0.7, color='steelblue', label='PPO (mlp_hybrid)')
ax.axhline(qos_rule, color='red', linestyle='--', linewidth=2, label=f'Rule baseline ({qos_rule:.5f})')

ax.set_xlabel('Seed', fontsize=11)
ax.set_ylabel('QoS Violation Rate', fontsize=11)
ax.set_title('QoS Violation Rate: PPO vs Rule Baseline', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(seed_nums)

plt.tight_layout()

plot_path = Path(__file__).parent / "wilcoxon_plot.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved: {plot_path}")
print()

# ============================================================================
# 6. PAPER-READY SUMMARY
# ============================================================================

print("=" * 80)
print("SECTION F: PAPER-READY STATEMENTS")
print("=" * 80)
print()

interpretation = "not statistically significant (p > 0.05)" if w_pval >= 0.05 else "statistically significant (p < 0.05)"

sentence1 = (
    f"We compared the PPO (feed-forward) policy against the rule-based baseline using a Wilcoxon "
    f"signed-rank test across 5 random seeds. The median difference in total energy was "
    f"{diff_median:+.2f} Wh (mean: {diff_mean:+.2f} ± {diff_std:.2f} Wh), with test statistic W = {w_stat:.0f} "
    f"and p-value = {w_pval:.4f}."
)

if w_pval < 0.05 and diff_mean < 0:
    interpretation_text = (
        f"PPO achieves significantly lower energy ({ppo_mean:.2f} Wh) compared to the rule-based baseline "
        f"({rule_energy:.2f} Wh), representing a {energy_saving_percent:.1f}% reduction (p = {w_pval:.4f})."
    )
elif w_pval < 0.05 and diff_mean > 0:
    interpretation_text = (
        f"The rule-based baseline achieves significantly lower energy ({rule_energy:.2f} Wh) than PPO "
        f"({ppo_mean:.2f} Wh), by {abs(energy_saving_percent):.1f}% (p = {w_pval:.4f})."
    )
else:
    interpretation_text = (
        f"No statistically significant difference was detected between PPO ({ppo_mean:.2f} Wh) and the "
        f"rule-based baseline ({rule_energy:.2f} Wh) at α = 0.05 "
        f"(p = {w_pval:.4f}, n = {len(ppo_values)} seeds). The sample size (n = {len(ppo_values)}) is small, "
        f"limiting statistical power for detecting small effect sizes."
    )

print("Sentence 1 (Statistical Test Description):")
print(sentence1)
print()

print("Sentence 2 (Interpretation - Conservative):")
print(interpretation_text)
print()

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("=" * 80)
print("QUICK REFERENCE TABLE")
print("=" * 80)
print()

summary_table = pd.DataFrame({
    'Metric': ['Total Energy (Wh)', 'Average SEC', 'QoS Violation Rate'],
    'PPO Mean': [f"{ppo_mean:.4f}", f"{np.mean(sec_ppo):.6f}", f"{np.mean(qos_ppo):.6f}"],
    'PPO Std': [f"{ppo_std:.4f}", f"{np.std(sec_ppo, ddof=1):.6f}", f"{np.std(qos_ppo, ddof=1):.6f}"],
    'Rule': [f"{rule_energy:.4f}", f"{sec_rule:.6f}", f"{qos_rule:.6f}"],
    'W-stat': [f"{w_stat:.1f}", f"{sec_w_stat:.1f}", f"{qos_w_stat:.1f}"],
    'p-value': [f"{w_pval:.4f}", f"{sec_w_pval:.4f}", f"{qos_w_pval:.4f}"],
    'Sig (α=0.05)': ["Yes" if w_pval < 0.05 else "No", "Yes" if sec_w_pval < 0.05 else "No", 
                      "Yes" if qos_w_pval < 0.05 else "No"]
})

print(summary_table.to_string(index=False))
print()

print("=" * 80)
print("END ANALYSIS")
print("=" * 80)
