# B2 Multi-Seed PPO Results Aggregation Report

**Date**: March 18, 2026  
**Status**: ✅ Complete  

---

## Executive Summary

Successfully aggregated PPO evaluation results across 5 random seeds (42, 123, 456, 789, 1024) on the B2 test set. Generated manuscript-ready mean ± standard deviation metrics for both policies:

- **mlp_hybrid (feed-forward)**: 122.58 ± 1.50 Wh total energy
- **lstm_history (recurrent)**: 125.29 ± 5.11 Wh total energy

All metrics extracted from identical evaluation setting with no experiment configuration mixing.

---

## 1. Methodology

### 1.1 Data Sources

| Component | Source | Format |
|-----------|--------|--------|
| Evaluation metadata | `evaluation_analytics_summary.csv` | CSV (multi-row) |
| Per-seed energy | `*/seed_*/eval/evaluation_summary_*.csv` | CSV (Learned row) |
| Execution events | Parsed from CSV columns | Integer counts |

### 1.2 Experiment Configuration (B2_top2_seeds)

**Common baseline settings:**
- **Policy**: MlpLstmPolicy (baseline) adapted per config
- **Observation schema**: hybrid (traffic history + calendar features)
- **QoS penalty coefficient (λ)**: 30.0
- **Type-switch cost**: 0.03 per transition
- **Scaling costs**: 0.012 per scale-up, 0.003 per scale-down
- **Cooldown period (I_c)**: 4 timesteps
- **Max OAI instances**: 2
- **Performance threshold**: 0.95
- **Environment**: ManualCooldownEnv
- **Training timesteps**: 300,000 per seed

**Evaluated policies:**
1. **mlp_hybrid**: MlpPolicy (feed-forward) + hybrid observation
2. **lstm_history**: MlpLstmPolicy (recurrent) + history observation

### 1.3 Test Set Characteristics

- **Test split**: B2 holdout validation traffic
- **Episode length**: 721 timesteps (fixed)
- **Traffic patterns**: Different from training/validation
- **Evaluation runs**: Final model evaluation (after training converged)

### 1.4 Metrics Extracted

| Metric | Definition | Unit | Source |
|--------|-----------|------|--------|
| **E_tot** | Total energy consumption | Wh | `evaluation_summary_*.csv` → Total Energy (Wh) |
| **avg_SEC** | Specific energy consumption (power/throughput) | W/Mbps | `evaluation_analytics_summary.csv` → avg_sec |
| **v** | QoS violation rate | dimensionless | `evaluation_analytics_summary.csv` → violation_rate |
| **n_sw** | Type-switch events | count | switch_dpdk_to_oai + switch_oai_to_dpdk |
| **n_sc** | Scaling events | count | scale_up + scale_down |
| **n_bl** | Blocked action requests | count | cooldown_blocked |

### 1.5 Statistical Aggregation

- **Method**: Sample mean and standard deviation across seeds
- **Degrees of freedom**: ddof=1 (unbiased sample std)
- **Formula**: 
  - Mean: $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$
  - Std: $\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \mu)^2}$

---

## 2. Results Summary

### 2.1 Per-Seed Data (Raw)

**lstm_history (5 seeds)**

| Seed | E_tot (Wh) | avg_SEC (W/Mbps) | v | n_sw | n_sc | n_bl |
|------|-----------|-----------------|---|------|------|------|
| 42   | 122.11 | 0.007409 | 0.008322 | 38 | 10 | 54 |
| 123  | 134.00 | 0.010257 | 0.001387 | 18 | 6  | 27 |
| 456  | 125.74 | 0.007641 | 0.001387 | 16 | 0  | 7  |
| 789  | 122.71 | 0.007489 | 0.009709 | 14 | 7  | 21 |
| 1024 | 121.91 | 0.007398 | 0.009709 | 14 | 6  | 5  |

**mlp_hybrid (5 seeds)**

| Seed | E_tot (Wh) | avg_SEC (W/Mbps) | v | n_sw | n_sc | n_bl |
|------|-----------|-----------------|---|------|------|------|
| 42   | 122.93 | 0.007603 | 0.002774 | 18 | 4  | 9  |
| 123  | 122.32 | 0.007560 | 0.001387 | 14 | 11 | 4  |
| 456  | 120.79 | 0.007350 | 0.001387 | 17 | 7  | 3  |
| 789  | 121.98 | 0.007284 | 0.009709 | 21 | 8  | 20 |
| 1024 | 124.87 | 0.007603 | 0.001387 | 12 | 23 | 24 |

### 2.2 Aggregate Results (Mean ± Std)

**For Manuscript Table:**

| Policy | n | E_tot (Wh) | avg_SEC | v | n_sw | n_sc | n_bl |
|--------|---|-----------|---------|---|------|------|------|
| **lstm_history** | 5 | 125.29 ± 5.11 | 0.00804 ± 0.00124 | 0.00610 ± 0.00434 | 20.0 ± 10.2 | 5.8 ± 3.6 | 22.8 ± 19.8 |
| **mlp_hybrid** | 5 | 122.58 ± 1.50 | 0.00748 ± 0.00015 | 0.00333 ± 0.00362 | 16.4 ± 3.5 | 10.6 ± 7.4 | 12.0 ± 9.5 |

### 2.3 Key Findings

**Energy Efficiency:**
- mlp_hybrid shows 2.7% lower mean total energy (122.58 vs 125.29 Wh)
- mlp_hybrid has tighter energy variance (std 1.50 vs 5.11 Wh)
- Reflects more consistent learning across seeds

**SEC (Power Efficiency):**
- mlp_hybrid achieves 7.0% better avg_SEC (0.00748 vs 0.00804 W/Mbps)
- mlp_hybrid shows much lower variance (std 0.00015 vs 0.00124)
- Indicates more stable control policy

**QoS Compliance:**
- mlp_hybrid: v = 0.00333 ± 0.00362 (better compliance)
- lstm_history: v = 0.00610 ± 0.00434 (slightly higher violations)
- Both well within acceptable ranges

**Action Patterns:**
- mlp_hybrid: fewer type-switches (16.4 vs 20.0), more scaling (10.6 vs 5.8)
- mlp_hybrid: fewer blocked requests (12.0 vs 22.8), indicating better scheduling
- Suggests feed-forward policy makes more decisive OAI scaling decisions

---

## 3. Data Quality Assurance

### 3.1 Consistency Checks ✅

- [x] All 10 seed records (2 policies × 5 seeds) accounted for
- [x] All seeds from same experiment group: `B2_top2_seeds`
- [x] All evaluations on identical B2 test traffic split
- [x] Episode length: 721 steps (fixed, non-variable)
- [x] Same cooldown configuration (I_c=4)
- [x] Same observation schemas per policy (no cross-contamination)
- [x] All final-model evaluations (not intermediate checkpoints)

### 3.2 Seed Validation: Seed 456

Reconciliation with potential manuscript draft values:

**mlp_hybrid seed_456:**
- E_tot: 120.79 Wh ✅
- avg_SEC: 0.007350 W/Mbps ✅
- violation_rate: 0.001387 ✅
- All metrics aligned with source CSV

**lstm_history seed_456:**
- E_tot: 125.74 Wh ✅
- avg_SEC: 0.007641 W/Mbps ✅
- violation_rate: 0.001387 ✅
- All metrics aligned with source CSV

### 3.3 Outlier Analysis

**lstm_history seed 123:** E_tot = 134.00 Wh (outlier, +8.7Wh from mean)
- avg_SEC elevated (0.010257 vs mean 0.00804) → indicates less efficient episode
- Not a data quality issue; reflects valid training run variation
- **Action**: Retained (legitimate seed result, no errors)

**lstm_history seed 42:** n_sw = 38 (outlier, +18 from mean)
- Reflects actual high switching behavior that seed
- n_bl = 54 (also elevated, accumulation of blocks)
- Valid result from different random trajectory
- **Action**: Retained (legitimate behavior diversity)

### 3.4 Floating-Point Precision

- Energy values: 2 decimal places (hundredths of Wh)
- SEC/v: 6-7 decimal places (sufficient precision for energy metrics)
- No rounding artifacts detected
- Standard deviations computed at full precision

---

## 4. Generated Outputs

### 4.1 CSV Files

**`b2_seed_results_main_table.csv`**
- Format: One row per seed
- Fields: policy, seed, E_tot, avg_SEC, v, n_sw, n_sc, n_bl
- Rows: 10 (2 policies × 5 seeds)
- Use: Detailed per-seed values for sensitivity/robustness analysis

**`b2_seed_aggregate_main_table.csv`**
- Format: One row per policy
- Fields: policy, n_seeds, E_tot_mean, E_tot_std, avg_SEC_mean, avg_SEC_std, ... (all metrics mean±std)
- Rows: 2 (lstm_history, mlp_hybrid)
- Use: Direct inclusion in main results table of manuscript

### 4.2 Markdown Summary

**`b2_seed_aggregate_summary.md`**
- Formatted table with per-seed and aggregate results
- Includes experiment configuration details
- Quick-reference format for manuscript writing

---

## 5. Manuscript Integration

### 5.1 Recommended Table Format

```
Table X: B2 Test Set Performance Across Seeds (Mean ± Std)

Policy          | Seeds | E_tot (Wh)     | avg_SEC (W/Mbps) | v           | n_sw       | n_sc      | n_bl       |
----------------|-------|----------------|------------------|-------------|-----------|-----------|------------|
mlp_hybrid      | 5     | 122.58 ± 1.50  | 0.00748 ± 0.00015| 0.00333 ± x | 16.4 ± 3.5| 10.6 ± 7.4| 12.0 ± 9.5|
lstm_history    | 5     | 125.29 ± 5.11  | 0.00804 ± 0.00124| 0.00610 ± x | 20.0 ± 10.2| 5.8 ± 3.6| 22.8 ± 19.8|
```

### 5.2 Terminology Alignment

✅ Confirmed manuscript term usage:
- **DPDK** (not OAI singular) for single-instance high-performance mode
- **OAI** for scalable multi-instance mode
- **n_sw** for type-switch events ✅
- **n_sc** for scaling events ✅
- **n_bl** for blocked requests ✅
- **avg_SEC** for specific energy consumption ✅
- **v** for violation rate ✅

---

## 6. Validation Against Experiment Settings

### 6.1 No Configuration Mixing

All 10 records verified to belong to single experiment:
- **Group**: B2_top2_seeds (✅ confirmed)
- **Test set**: B2 holdout (✅ confirmed)
- **Environment**: ManualCooldownEnv (✅ confirmed)
- **Episode length**: 721 steps (✅ all identical)

### 6.2 Epoch/Training Completion

All seed records represent final converged models:
- Training timesteps: 300,000 per seed (✅)
- Evaluation: post-training final model (✅)
- No intermediate checkpoint pollution (✅)

### 6.3 Traffic and Dynamics

B2 test set characteristics consistent:
- One fixed traffic time series applied to all 10 seeds
- Same calendar features (weekday, hour encodings)
- Same performance predictor models
- Differences in results purely due to learned policy variation

---

## 7. Recommendations

### 7.1 For Manuscript

1. **Use mlp_hybrid aggregate row** as primary result for main table (lower variance, better performance)
2. **Supplement with lstm_history** for comparison/robustness
3. **Report seed 456 single-seed values** in caption if main paper only shows aggregate
4. **Add note**: "Metrics reported as mean ± standard deviation over 5 random seeds"

### 7.2 For Reproducibility

- Store seed summary: `b2_seed_aggregate_summary.md`
- Archive per-seed CSV: `b2_seed_results_main_table.csv`
- Link to experiment config: `experiments/sweep_config_b2_top2_seeds.yaml`
- Reference run directories: `experiments/results/B2_top2_seeds/{lstm_history,mlp_hybrid}/seed_*/`

### 7.3 For Supplementary Materials

- Include full per-seed table (Table X in Supplementary)
- Report confidence intervals (95% CI) alongside std if space permitted
- Consider seed variability discussion (especially lstm_history variance)

---

## 8. Audit Trail

### 8.1 Processing Steps

1. ✅ Located B2_top2_seeds experiment directory
2. ✅ Parsed evaluation_analytics_summary.csv (10 records)
3. ✅ Extracted energy data from per-seed evaluation_summary CSV files
4. ✅ Computed derived metrics: n_sw, n_sc, n_bl
5. ✅ Computed per-policy aggregates: mean, std (ddof=1)
6. ✅ Generated output CSVs and markdown summary
7. ✅ Verified data consistency and no outlier errors

### 8.2 Time Completed

- Aggregation script execution: < 1 second
- Files generated: 3
- Total records processed: 10 seeds × 8 metrics = 80 data points

---

## 9. Conclusion

✅ **Aggregation Status: COMPLETE**

All B2 multi-seed PPO results successfully extracted, validated, and aggregated. The generated CSV files are ready for direct integration into the manuscript's main results table. Both mlp_hybrid and lstm_history policies show consistent behavior across seeds, with mlp_hybrid demonstrating superior energy efficiency and lower variance.

---

**Generated by**: `scripts/aggregate_b2_seed_results.py`  
**Output directory**: `experiments/results/B2_top2_seeds/`  
**Files created**:
- `b2_seed_results_main_table.csv` (per-seed details)
- `b2_seed_aggregate_main_table.csv` (manuscript-ready aggregate)
- `b2_seed_aggregate_summary.md` (markdown summary)
- `AGGREGATION_REPORT.md` (this file)
