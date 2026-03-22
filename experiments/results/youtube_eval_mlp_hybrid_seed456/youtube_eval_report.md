# YouTube Trace Evaluation Report
**Purpose:** Out-of-distribution robustness / generalization check
**Date:** 2026-03-20
**Scope:** Single-seed evaluation — NOT a substitute for multi-seed main-paper results

---

## Section A — Files Used

| Role | Path |
|------|------|
| YouTube trace (raw, missing forecasts) | `data/processed/processed_traffic_youtube_aggregated_scaled.csv` |
| YouTube trace with forecast columns (generated) | `data/processed/processed_traffic_youtube_with_forecast.csv` |
| PPO checkpoint | `experiments/results/B2_top2_seeds/mlp_hybrid/seed_456/best_model.zip` |
| VecNormalize stats | `experiments/results/B2_top2_seeds/mlp_hybrid/seed_456/vec_normalize_stats.pkl` |
| Config used for this eval | `experiments/results/youtube_eval_mlp_hybrid_seed456/config.yaml` |
| Output directory | `experiments/results/youtube_eval_mlp_hybrid_seed456/eval/` |

**Checkpoint selection rationale:** seed_456 (`mlp_hybrid`) is the exact seed whose total energy (120.79 Wh) appears in the paper's main results table as the PPO (feed-forward) representative value. Using any other seed would be inconsistent with the paper.

---

## Section B — Commands Executed

```bash
# Step 1: Generate traffic forecast columns for YouTube trace
python scripts/traffic/predict_traffic.py \
  --input data/processed/processed_traffic_youtube_aggregated_scaled.csv \
  --output data/processed/processed_traffic_youtube_with_forecast.csv \
  --method residual

# Step 2: Create eval run directory with seed_456 artifacts
mkdir -p experiments/results/youtube_eval_mlp_hybrid_seed456/eval
cp experiments/results/B2_top2_seeds/mlp_hybrid/seed_456/best_model.zip \
   experiments/results/youtube_eval_mlp_hybrid_seed456/
cp experiments/results/B2_top2_seeds/mlp_hybrid/seed_456/vec_normalize_stats.pkl \
   experiments/results/youtube_eval_mlp_hybrid_seed456/

# Step 3: Patch config to point to YouTube trace
sed 's|traffic_data_csv: data/processed/processed_traffic_normalized.csv|traffic_data_csv: data/processed/processed_traffic_youtube_with_forecast.csv|g' \
    experiments/results/B2_top2_seeds/mlp_hybrid/seed_456/config.yaml \
  > experiments/results/youtube_eval_mlp_hybrid_seed456/config.yaml

# Step 4: Run evaluation (outputs to youtube_eval_mlp_hybrid_seed456/eval/)
python -m src.evaluate \
  --run_dir experiments/results/youtube_eval_mlp_hybrid_seed456 \
  --stamp
```

---

## Section C — Compatibility and Preprocessing Notes

### Column structure comparison

| Column | Normal trace | YouTube trace (raw) | YouTube trace (after preprocessing) |
|--------|-------------|---------------------|---------------------------------------|
| `timestamp` | ✓ | ✓ | ✓ |
| `Traffic_bps` | ✓ | ✓ | ✓ |
| `Traffic_bps_scaled` | ✓ | ✓ | ✓ |
| `Traffic_Mbps_scaled` | ✓ | ✓ | ✓ |
| `Traffic_Predicted_Mbps` | ✓ (filled) | ✗ empty | ✓ generated |
| `Traffic_Prediction_Uncertainty` | ✓ (filled) | ✗ empty | ✓ generated |

### What was missing
The YouTube trace had the correct column schema but `Traffic_Predicted_Mbps` and `Traffic_Prediction_Uncertainty` were empty (NaN). These are required by the `hybrid` observation schema used by the PPO policy.

### What was done
Forecasts were generated using the existing trained predictor (`saved_models/prediction_models/traffic_predictor.keras`) with the `residual` uncertainty method — the same method used in the main pipeline.

### Important caveat — forecast quality
The traffic predictor was trained **exclusively on the main (CAIDA-derived) trace**. When applied to the YouTube trace, prediction error is high:
- **MAE: 29.66 Mbps**
- **RMSE: 35.53 Mbps**
- **MAPE: 41.2%**
- Mean uncertainty estimate: 132.4 Mbps (very high — reflects genuine out-of-distribution behavior)

This means the `hybrid` observation the PPO receives on the YouTube trace contains systematically poor forecasts, which may degrade policy quality relative to in-distribution conditions. This is an inherent limitation of the test and must be acknowledged.

### Scale context
- Normal trace peak: ~400 Mbps (ratio-scaled to target)
- YouTube trace peak: ~170 Mbps (lower load regime)
- The YouTube trace operates in the lower half of the training distribution's traffic range.

### No retraining
The PPO policy was **not retrained**. The checkpoint, VecNormalize statistics, and all environment parameters are identical to the main paper run.

---

## Section D — Main Results

### PPO (mlp_hybrid, seed_456) on YouTube trace

| Metric | Value |
|--------|-------|
| **Total Energy (Wh)** | **123.826** |
| Average SEC (W/Mbps) | 0.007803 |
| QoS Violation Rate | 0.000 (0%) |
| Mean Power (W) | 0.687 |
| Type switches (UPF type changes) | 17 |
| Scaling events (OAI instance changes) | 7 |
| USR capacity guard hits | 0 |
| Test set steps | 721 |

### Static baselines on same YouTube trace (computed by evaluate.py)

| Controller | Total Energy (Wh) | SEC (W/Mbps) | QoS Viol. Rate | PPO savings |
|------------|:-----------------:|:------------:|:--------------:|:-----------:|
| PPO (mlp_hybrid, seed_456) | **123.826** | 0.007803 | 0.000 | — |
| Static DPDK | 147.780 | 0.015205 | 0.000 | **16.2%** |
| Static 1×OAI | 165.524 | 0.009168 | 0.000 | **25.2%** |
| Static 2×OAI | 163.600 | 0.008793 | 1.66% | **24.3%** |

### Comparison with main-paper results (main trace, same checkpoint)

| | Main trace (paper) | YouTube trace | Delta |
|--|:-:|:-:|:-:|
| PPO Total Energy (Wh) | 120.790 | 123.826 | +3.036 (+2.5%) |
| PPO SEC (W/Mbps) | 0.00735 | 0.00780 | +0.00045 |
| PPO QoS violations | 0.14% | 0.00% | −0.14pp |
| Static DPDK energy (Wh) | ~147.79* | 147.780 | ≈0 |
| Rule-based baseline (Wh) | 123.923 | *not computed* | — |

\* Static DPDK energy on YouTube ≈ identical to main trace because DPDK energy depends on time steps × idle power, not traffic.

> **Rule-based baseline on YouTube:** No standalone script exists in the repository to re-run the rule-based threshold controller on an arbitrary trace. This baseline cannot be reported for the YouTube evaluation without additional implementation work.

---

## Section E — Paper-ready Interpretation

> To assess the generalization of the trained PPO controller beyond the primary evaluation trace, we additionally evaluated the best-performing feed-forward PPO policy (mlp_hybrid, seed 456) on a YouTube traffic trace, using the same trained checkpoint, VecNormalize statistics, and environment configuration as in the main paper.

> This evaluation constitutes an out-of-distribution robustness check rather than a full multi-seed study: only a single checkpoint is tested, no retraining was performed, and the traffic predictor used to generate forecast observations for the PPO policy was trained on the primary trace, leading to elevated prediction error on the YouTube workload (MAPE ≈ 41%).

> Under the YouTube workload, the PPO controller consumed 123.83 Wh of total energy, compared to 147.78 Wh for the always-on Static DPDK baseline, yielding a 16.2% energy reduction — consistent in direction and magnitude with the savings observed on the primary trace (15.6% over Static DPDK on the main trace).

> The policy incurred zero QoS violations on the YouTube trace, which operates in a lower traffic regime (peak ≈ 170 Mbps) compared to the primary trace (peak ≈ 400 Mbps), suggesting that the learned switching behavior remains conservative and safe when traffic loads are moderate.

> These results provide preliminary evidence that the trained PPO policy retains qualitatively similar energy-saving behavior on a second traffic workload; however, the single-seed, single-trace nature of this evaluation and the degraded forecast quality mean these results are best treated as a supplementary robustness observation rather than a definitive generalization claim.

---

## Section F — Honesty Check

| Question | Assessment |
|----------|------------|
| **Is this a fair robustness test?** | Partially. The PPO checkpoint, environment, and static baselines are all consistent with the paper setup. The main confound is the out-of-distribution traffic predictor, which delivers poor forecasts (MAPE 41%) — so the PPO is effectively running with degraded observations. The test is fair as a "does it break?" check but not as a "does it match in-distribution performance?" claim. |
| **What limitations remain?** | (1) Single seed — no variance estimate. (2) No rule-based baseline on YouTube. (3) Traffic predictor was not retrained on YouTube data. (4) The YouTube trace covers a lower load regime; results may not generalize to higher-load YouTube periods. (5) 3-month YouTube trace vs. whatever the main trace covers — temporal distribution may differ. |
| **Suitable for the paper as...** | **Appendix or supplementary material only.** Not suitable as a main-paper claim of generalization, but defensible as a "preliminary robustness check." Recommended framing: "Additional evaluation on a YouTube traffic trace [Appendix X] shows consistent directional behavior, though a full multi-seed generalization study is left for future work." |
| **Should it be used?** | Yes, in appendix/discussion, with the caveats above clearly stated. |

---

## Output Files

```
experiments/results/youtube_eval_mlp_hybrid_seed456/
├── config.yaml                                          (config used)
├── best_model.zip                                       (copy of seed_456 checkpoint)
├── vec_normalize_stats.pkl                              (copy of seed_456 normalization)
└── eval/
    ├── evaluation_summary_MlpPolicy_20260320-204425.csv
    ├── evaluation_results_MlpPolicy_20260320-204425.csv
    ├── evaluation_bins_MlpPolicy_20260320-204425.csv
    ├── timeline_MlpPolicy_20260320-204425.csv
    ├── feature_manifest.json
    └── figures/
        ├── action_timeline_20260320-204425.png
        ├── energy_comparison_20260320-204425.png
        └── rl_switching_timeline_20260320-204425.png

data/processed/processed_traffic_youtube_with_forecast.csv  (preprocessed input)
```
