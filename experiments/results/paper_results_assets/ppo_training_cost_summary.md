# PPO Training Computational Cost (Paper-Ready)

## Paper-ready statement
Training the main feed-forward PPO controller used in the paper (A1 `mlp_hybrid`, target 3e5 timesteps) takes about 7.39 minutes per seed (443.25 s, exact run log), while the recurrent PPO comparator (`lstm_history`) takes about 11.55 minutes per seed (692.71 s, exact run log); robustness reruns over 5 seeds require about 37.06 minutes total (feed-forward) and 57.67 minutes total (recurrent).

## Main setting selected for manuscript timing
- Selected main setting: `A1_policy_observation/mlp_hybrid/seed_456`
- Why this is the main paper setting:
  - `PPO (feed-forward)` in the overall paper table maps to the A1 `mlp_hybrid` row.
  - The row values match exactly between the overall table and A1 model-tracking table.
  - A1 run metadata logs exact training wall-clock (`train_time_s`).

## Exact values and scope
- Feed-forward PPO main setting (`mlp_hybrid`, A1, seed 456):
  - `train_time_s = 443.25 s` (exact)
  - `7.39 minutes` (derived conversion)
  - Scope: one seed, one PPO variant, one experiment setting
- Recurrent PPO comparator (`lstm_history`, A1, seed 456):
  - `train_time_s = 692.71 s` (exact)
  - `11.55 minutes` (derived conversion)
  - Scope: one seed, one PPO variant, one experiment setting
- Feed-forward robustness total (B2, 5 seeds):
  - Sum of exact per-seed run times = `2223.37 s` (`37.06 minutes`)
  - Scope: total time for all seeds (5 runs)
- Recurrent robustness total (B2, 5 seeds):
  - Sum of exact per-seed run times = `3460.06 s` (`57.67 minutes`)
  - Scope: total time for all seeds (5 runs)

## Hardware used (from available logs)
- Directly available in runtime/resource artifacts:
  - CPU utilization and frequency sampled during training (example frequency around 4050 MHz)
  - RAM usage sampled during training (roughly 9.7-9.9 GB in D1 runtime logs)
  - GPU telemetry columns exist but are empty; aggregated tables report `gpu_logged = False`
- CPU/GPU model names and total system RAM are not explicitly logged in repository artifacts.
- Interpretation for manuscript:
  - Training appears CPU-only (or at least no observable GPU use) based on `gpu_logged=False`, empty GPU metrics, and zero cumulative GPU energy.
  - Confidence: medium-high for CPU-only classification; low for exact CPU/GPU model identification.

## Candidate timing reconciliation
Multiple valid timing artifacts exist:
- A1/B2 `run_meta.json` values (`train_time_s`) are end-to-end run wall-clock for each experiment run.
- D1 `training_resource_summary.json` / `tab_d1_resource_runtime_*.csv` provide training-loop elapsed values (`train_elapsed_s`).

For the manuscript main sentence, A1 `run_meta.json` is used because it corresponds exactly to the controller reported in the main paper table (`PPO (feed-forward)`), while D1 runtime assets are used as corroborative resource evidence.

## Evidence sources used
- `experiments/results/A1_policy_observation/mlp_hybrid/seed_456/run_meta.json`
- `experiments/results/A1_policy_observation/lstm_history/seed_456/run_meta.json`
- `experiments/results/B2_top2_seeds/mlp_hybrid/seed_42/run_meta.json`
- `experiments/results/B2_top2_seeds/mlp_hybrid/seed_123/run_meta.json`
- `experiments/results/B2_top2_seeds/mlp_hybrid/seed_456/run_meta.json`
- `experiments/results/B2_top2_seeds/mlp_hybrid/seed_789/run_meta.json`
- `experiments/results/B2_top2_seeds/mlp_hybrid/seed_1024/run_meta.json`
- `experiments/results/B2_top2_seeds/lstm_history/seed_42/run_meta.json`
- `experiments/results/B2_top2_seeds/lstm_history/seed_123/run_meta.json`
- `experiments/results/B2_top2_seeds/lstm_history/seed_456/run_meta.json`
- `experiments/results/B2_top2_seeds/lstm_history/seed_789/run_meta.json`
- `experiments/results/B2_top2_seeds/lstm_history/seed_1024/run_meta.json`
- `experiments/results/paper_results_assets/tab_overall_results.csv`
- `experiments/results/paper_results_assets/tab_mlp_hybrid_A_to_D.csv`
- `experiments/results/paper_results_assets/tab_lstm_history_A_to_D.csv`
- `experiments/sweep_config.yaml`
- `experiments/sweep_config_b2_top2_seeds.yaml`
- `experiments/results/paper_results_assets/tab_d1_resource_runtime_summary.csv`
- `experiments/results/paper_results_assets/tab_d1_resource_runtime_per_run.csv`
- `experiments/results/D1_mlp_lstm_cd4_k_threshold/mlp_hybrid_cd4_k2_t90/seed_456/resource_usage.csv`
- `experiments/results/D1_mlp_lstm_cd4_k_threshold/mlp_hybrid_cd4_k2_t90/seed_456/training_resource_summary.json`
- `scripts/run_experiments.py`
- `src/callbacks.py`
