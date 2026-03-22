# YouTube Trace — Matched Robustness Evaluation Report
**Date:** 2026-03-20
**Purpose:** Full matched additional-trace evaluation comparing PPO (all seeds), rule-based baseline, static baselines, and offline-optimal on the YouTube traffic trace.
**Status:** Complete — no retraining performed.
**Rule-based margin corrected:** `margin_mbps` updated from 23.0 → **44.55 Mbps** (= MAE × 1.5 = 29.7 × 1.5, per paper definition). Energy result unchanged (123.57 Wh) — see Section B2.

---

## Section A — Files Used

| Role | Path |
|------|------|
| YouTube trace (raw, forecasts empty) | `data/processed/processed_traffic_youtube_aggregated_scaled.csv` |
| YouTube trace with forecasts (generated prior session) | `data/processed/processed_traffic_youtube_with_forecast.csv` |
| Config template used for all PPO seeds | `experiments/results/youtube_eval_mainconfig_allseeds/config_template_youtube.yaml` |
| PPO seed_42 checkpoint | `experiments/results/B2_top2_seeds/mlp_hybrid/seed_42/best_model.zip` |
| PPO seed_123 checkpoint | `experiments/results/B2_top2_seeds/mlp_hybrid/seed_123/best_model.zip` |
| PPO seed_456 checkpoint | `experiments/results/B2_top2_seeds/mlp_hybrid/seed_456/best_model.zip` |
| PPO seed_789 checkpoint | `experiments/results/B2_top2_seeds/mlp_hybrid/seed_789/best_model.zip` |
| PPO seed_1024 checkpoint | `experiments/results/B2_top2_seeds/mlp_hybrid/seed_1024/best_model.zip` |
| Rule-based baseline script | `scripts/evaluation_baseline-rule.py` + `scripts/baseline_rule.py` |
| Rule-based outputs | `experiments/results/youtube_eval_mainconfig_allseeds/baselines/rule_youtube_results.csv` |
| Offline-optimal outputs | `experiments/results/youtube_eval_mainconfig_allseeds/baselines/baseline_offline_optimal_summary_youtube_mlp_hybrid_seed456.csv` |
| PPO per-seed summary | `experiments/results/youtube_eval_mainconfig_allseeds/analysis/ppo_per_seed_youtube.csv` |
| Aggregated PPO summary | `experiments/results/youtube_eval_mainconfig_allseeds/analysis/ppo_aggregated_youtube.csv` |
| Combined comparison table | `experiments/results/youtube_eval_mainconfig_allseeds/analysis/youtube_comparison_table.csv` |
| Plots | `experiments/results/youtube_eval_mainconfig_allseeds/analysis/plots/` |

---

## Section B — Commands Executed

```bash
# 1. Forecast generation (done in prior session, reused here)
python scripts/traffic/predict_traffic.py \
  --input  data/processed/processed_traffic_youtube_aggregated_scaled.csv \
  --output data/processed/processed_traffic_youtube_with_forecast.csv \
  --method residual

# 2. Create per-seed run directories (all 5 seeds)
for SEED in 42 123 456 789 1024; do
  mkdir -p experiments/results/youtube_eval_mainconfig_allseeds/mlp_hybrid/seed_${SEED}/eval
  cp experiments/results/B2_top2_seeds/mlp_hybrid/seed_${SEED}/best_model.zip \
     experiments/results/youtube_eval_mainconfig_allseeds/mlp_hybrid/seed_${SEED}/
  cp experiments/results/B2_top2_seeds/mlp_hybrid/seed_${SEED}/vec_normalize_stats.pkl \
     experiments/results/youtube_eval_mainconfig_allseeds/mlp_hybrid/seed_${SEED}/
  sed 's|processed_traffic_normalized.csv|processed_traffic_youtube_with_forecast.csv|g' \
      experiments/results/B2_top2_seeds/mlp_hybrid/seed_${SEED}/config.yaml \
    > experiments/results/youtube_eval_mainconfig_allseeds/mlp_hybrid/seed_${SEED}/config.yaml
done

# 3. PPO evaluation — all 5 seeds
for SEED in 42 123 456 789 1024; do
  python -m src.evaluate \
    --run_dir experiments/results/youtube_eval_mainconfig_allseeds/mlp_hybrid/seed_${SEED} \
    --stamp
done

# 4. Rule-based baseline (temporarily patched root config to YouTube trace)
#    One-line fix applied to evaluation_baseline-rule.py:
#    max_power = pre["test"]["max_power_for_normalization"]
#      → max_power = float(env_cfg.get("max_power", 3.58))
PYTHONPATH=/home/ubuntu/EnergyAwareUPF python scripts/evaluation_baseline-rule.py

# 5. Offline-optimal (seed_456 as reference run)
python -m scripts.run_offline_optimal_baseline \
  --run-dir experiments/results/youtube_eval_mainconfig_allseeds/mlp_hybrid/seed_456 \
  --output-dir experiments/results/youtube_eval_mainconfig_allseeds/baselines \
  --tag youtube_mlp_hybrid_seed456
```

---

## Section B2 — Rule-based Margin Correction

The `margin_mbps` is defined in the paper as **MAE × 1.5**. On the main trace the old predictor has MAE ≈ 15.3 Mbps → margin = 23.0 Mbps ✓ (already correct). On the YouTube trace the old predictor has MAE = **29.7 Mbps** → correct margin = **44.55 Mbps**.

After re-running with `margin_mbps = 44.55`:
- Total energy (actual): **123.570 Wh** — unchanged
- Switches: 16 — unchanged
- The YouTube traffic is approximately bimodal near the 90.5 Mbps threshold; the wider dead-band does not alter the switching sequence. All previously reported figures remain valid.

---

## Section C — What Was Run

| Component | Status | Notes |
|-----------|--------|-------|
| PPO seed_42 | ✓ complete | 4 QoS violations |
| PPO seed_123 | ✓ complete | 0 violations |
| PPO seed_456 | ✓ complete | 0 violations (paper representative seed) |
| PPO seed_789 | ✓ complete | 12 violations |
| PPO seed_1024 | ✓ complete | 0 violations |
| Rule-based baseline | ✓ complete | 1-line bug fix applied (see Section B) |
| Static DPDK | ✓ complete | Extracted from evaluate.py output |
| Static 1×OAI | ✓ complete | Extracted from evaluate.py output |
| Static 2×OAI | ✓ complete | Extracted from evaluate.py output |
| Offline-optimal | ✓ complete | DP exact planner, discounted, γ=0.995, seed_456 config |

**Rule-based script fix note:** `evaluation_baseline-rule.py` referenced `pre["test"]["max_power_for_normalization"]`, a key not returned by `utils.load_and_preprocess_data()`. Fixed to use `env_cfg.get("max_power", 3.58)`, consistent with the config value.

---

## Section D — Main Results Table

| Controller | Total Energy (Wh) | Avg SEC (W/Mbps) | QoS Viol. Rate | Notes |
|------------|:-----------------:|:----------------:|:--------------:|-------|
| **Offline Optimal** | **116.70** | 0.006981 | 1.66% | DP exact planner, γ=0.995 |
| **PPO mean (n=5)** | **124.47 ± 1.60** | 0.007821 | 0.44% | mlp_hybrid, all seeds |
| PPO median | 124.67 | 0.007809 | 0.00% | |
| PPO 95% CI (energy) | [122.48, 126.46] Wh | — | — | t-distribution, n=5 |
| **Rule-based** | **123.57** | 0.006801 | 0.00% | utility threshold, cooldown=4 |
| Static DPDK | 147.78 | 0.015205 | 0.00% | always-on DPDK |
| Static 1×OAI | 165.52 | 0.009168 | 0.00% | always-on 1×OAI |
| Static 2×OAI | 163.60 | 0.008793 | 1.66% | always-on 2×OAI |

### Per-seed PPO results

| Seed | Energy (Wh) | SEC (W/Mbps) | QoS Viol. Rate | Vs Rule |
|------|:-----------:|:------------:|:--------------:|---------|
| seed_42 | 125.42 | 0.007974 | 0.55% | WORSE (+1.85 Wh) |
| seed_123 | 126.31 | 0.008087 | 0.00% | WORSE (+2.74 Wh) |
| seed_456 | 123.83 | 0.007803 | 0.00% | WORSE (+0.26 Wh) |
| seed_789 | **122.12** | 0.007432 | 1.66% | **BETTER (−1.45 Wh)** |
| seed_1024 | 124.67 | 0.007809 | 0.00% | WORSE (+1.10 Wh) |

### Relative comparisons (PPO mean vs baselines)

| Comparison | Energy Δ | Direction |
|------------|----------|-----------|
| PPO vs Rule-based | +0.90 Wh (+0.73%) | PPO slightly worse on average |
| PPO vs Static DPDK | −23.31 Wh (−15.8%) | PPO clearly better |
| PPO vs Offline Optimal | +7.77 Wh (+6.7%) | PPO gap from optimum |

---

## Section E — Statistical Analysis

### Wilcoxon signed-rank test: PPO vs Rule-based (total_energy_wh)

| | |
|--|--|
| Test | One-sample Wilcoxon signed-rank (zero_method='wilcox', two-tailed) |
| n | 5 seeds |
| W statistic | 3.0 |
| p-value | **0.3125** |
| Significant at α=0.05 | **No** |
| n_negative (PPO < rule, i.e. better) | 1 |
| n_positive (PPO > rule, i.e. worse) | 4 |
| median(diff) | +1.10 Wh |

**Interpretation:** No statistically significant difference. In fact, 4/5 seeds are *worse* than the rule-based baseline on this trace, with a median penalty of +1.10 Wh. With n=5, the minimum achievable p-value is 0.0625.

### Wilcoxon signed-rank test: PPO vs Static DPDK (total_energy_wh)

| | |
|--|--|
| W statistic | 0.0 |
| p-value | **0.0625** |
| Significant at α=0.05 | **No** (just above threshold) |
| n_negative (PPO < DPDK) | **5 / 5** |
| median(diff) | −23.11 Wh |

**Interpretation:** All 5 seeds beat Static DPDK (consistent direction), but n=5 limits the test to p=0.0625 — just above α=0.05. The energy saving over DPDK is strong and consistent.

### Wilcoxon signed-rank test: PPO vs Rule-based (avg_sec)

| | |
|--|--|
| W statistic | 0.0 |
| p-value | **0.0625** |
| n_positive (PPO SEC > rule SEC) | **5 / 5** |
| median(diff) | +0.00101 |

**Interpretation:** All PPO seeds have *higher* SEC (worse efficiency) than the rule-based baseline on the YouTube trace. The rule-based achieves lower SEC because it more aggressively selects OAI at low traffic loads on this trace.

---

## Section F — Paper-ready Interpretation

> To assess the generalization behavior of the trained PPO controller, we evaluated all five random seeds of the main feed-forward PPO configuration (mlp\_hybrid) on a YouTube traffic trace using the identical trained checkpoints and environment parameters as in the primary evaluation, without retraining.

> The YouTube trace operates in a lower traffic regime (peak ≈ 170 Mbps) compared to the primary trace (peak ≈ 400 Mbps), and the traffic predictor — trained exclusively on the primary trace — yields substantially elevated prediction error on this out-of-distribution workload (MAPE ≈ 41%). The PPO policy therefore receives degraded forecast observations throughout this evaluation, which may systematically bias its decisions.

> On the YouTube trace, PPO achieved a mean total energy of 124.47 ± 1.60 Wh across five seeds, representing a 15.8% reduction relative to the always-on Static DPDK baseline (147.78 Wh). However, the rule-based baseline achieved 123.57 Wh — slightly lower than the PPO mean — with only one out of five PPO seeds outperforming it. A Wilcoxon signed-rank test found no statistically significant difference between PPO and the rule-based baseline (W=3.0, p=0.31, n=5), consistent with the known limitation that n=5 provides insufficient power for detecting small effects.

> The offline-optimal controller achieved 116.70 Wh on the same trace, indicating a 6.7% gap between the trained PPO policy and the theoretically achievable minimum, suggesting room for improvement that may partly reflect the degraded forecast quality available to the policy.

> These results indicate that the trained PPO policy maintains robust energy savings over static baselines across traces, but does not consistently outperform the rule-based controller on this particular additional trace — a result that is qualitatively different from the primary evaluation and merits disclosure. The YouTube trace evaluation should be treated as a supplementary out-of-distribution check rather than a generalization claim.

---

## Section G — Honesty Check

| Question | Assessment |
|----------|------------|
| **Is this a fair robustness test?** | Partially. PPO checkpoints, environment, and baselines are all methodologically matched on the same test split of the same trace. The main confound is the out-of-distribution traffic predictor (MAPE 41%), which degrades PPO's hybrid observations. All controllers use the same lookup tables, so static and rule-based results are fully fair. |
| **Key result to disclose honestly** | **PPO does not outperform the rule-based baseline on the YouTube trace** (4/5 seeds worse, mean gap +0.90 Wh). This is an important negative finding that differs from the primary evaluation result. |
| **Why PPO may underperform the rule** | (1) The rule uses actual traffic for its energy computation (matched to what the env returns); PPO uses degraded forecasts. (2) The YouTube load is consistently low (mostly 50–170 Mbps), where the rule's threshold logic is well-calibrated. (3) PPO was trained on a higher-traffic trace and may over-select DPDK at low loads. |
| **Limitations remaining** | n=5 seeds; traffic predictor not retrained; single 3-month YouTube trace; timestamps are aligned but distribution is shifted; no confidence on whether the test split is representative of YouTube traffic. |
| **Suitable for paper as...** | **Appendix or supplementary only, with the negative finding disclosed.** Recommended framing: "An additional evaluation on a YouTube trace shows PPO maintains substantial savings over static baselines but does not consistently outperform the rule-based controller on this out-of-distribution workload, highlighting the sensitivity of the hybrid observation to forecast quality." Do NOT present this as evidence of generalization — the result is mixed. |
| **Should it be used?** | Yes, with the above framing. The negative finding is scientifically honest and adds credibility to the paper by avoiding cherry-picking. |

---

## Output File Listing

```
experiments/results/youtube_eval_mainconfig_allseeds/
├── config_template_youtube.yaml
├── youtube_matched_eval_report.md             ← this file
├── mlp_hybrid/
│   ├── seed_42/eval/   (summary, results, bins, timeline, figures CSVs)
│   ├── seed_123/eval/
│   ├── seed_456/eval/
│   ├── seed_789/eval/
│   └── seed_1024/eval/
├── baselines/
│   ├── rule_youtube_results.csv
│   ├── rule_thresholds_youtube.json
│   ├── baseline_offline_optimal_summary_youtube_mlp_hybrid_seed456.csv
│   ├── offline_optimal_eval_youtube_mlp_hybrid_seed456.csv
│   ├── offline_optimal_vs_baselines_youtube_mlp_hybrid_seed456.csv
│   └── figures/
│       ├── baseline_rule_timeline_enhanced.png
│       └── utility_diagnostics.png
└── analysis/
    ├── youtube_comparison_table.csv
    ├── ppo_per_seed_youtube.csv
    ├── ppo_aggregated_youtube.csv
    └── plots/
        ├── youtube_energy_comparison.png
        ├── youtube_sec_comparison.png
        └── youtube_qos_comparison.png
```
