# B2 Test Set PPO Results - Seed Aggregation

## Experiment Setting

- **Test Set**: B2 (validation holdout traffic)
- **Observation Schema**: hybrid (traffic + calendar features)
- **Cooldown Period (I_c)**: 4 timesteps
- **Max OAI Instances**: 2
- **Evaluation Environment**: ManualCooldownEnv

## Sources

- **Input data**: `experiments/results/B2_top2_seeds/evaluation_analytics_summary.csv`
- **Per-seed evaluations**: `experiments/results/B2_top2_seeds/{lstm_history|mlp_hybrid}/seed_*/eval/`
- **Energy data extracted from**: `evaluation_summary_*.csv` files

## Policies Evaluated

- **lstm_history**: 5 seeds
- **mlp_hybrid**: 5 seeds

## Per-Seed Results

| Policy | Seed | E_tot (Wh) | avg_SEC (W/Mbps) | v | n_sw | n_sc | n_bl |
|--------|------|-----------|-----------------|---|------|------|------|
| lstm_history | seed_1024 | 121.91 | 0.007398 | 0.009709 | 14 | 6 | 5 |
| lstm_history | seed_123 | 134.00 | 0.010257 | 0.001387 | 18 | 6 | 27 |
| lstm_history | seed_42 | 122.11 | 0.007409 | 0.008322 | 38 | 10 | 54 |
| lstm_history | seed_456 | 125.74 | 0.007641 | 0.001387 | 16 | 0 | 7 |
| lstm_history | seed_789 | 122.71 | 0.007489 | 0.009709 | 14 | 7 | 21 |
| mlp_hybrid | seed_1024 | 124.87 | 0.007603 | 0.001387 | 12 | 23 | 24 |
| mlp_hybrid | seed_123 | 122.32 | 0.007560 | 0.001387 | 14 | 11 | 4 |
| mlp_hybrid | seed_42 | 122.93 | 0.007603 | 0.002774 | 18 | 4 | 9 |
| mlp_hybrid | seed_456 | 120.79 | 0.007350 | 0.001387 | 17 | 7 | 3 |
| mlp_hybrid | seed_789 | 121.98 | 0.007284 | 0.009709 | 21 | 8 | 20 |

## Aggregate Results (Mean ± Std)

| Policy | n_seeds | E_tot (Wh) | avg_SEC (W/Mbps) | v | n_sw | n_sc | n_bl |
|--------|---------|-----------|-----------------|---|------|------|------|
| lstm_history | 5 | 125.29±5.11 | 0.008039±0.001244 | 0.006103±0.004342 | 20.0±10.2 | 5.8±3.6 | 22.8±19.8 |
| mlp_hybrid | 5 | 122.58±1.50 | 0.007480±0.000152 | 0.003329±0.003617 | 16.4±3.5 | 10.6±7.4 | 12.0±9.5 |

## Notes

- **E_tot**: Total energy consumption in Wh for B2 test episode (fixed 721 steps)
- **avg_SEC**: Mean specific energy consumption (power/throughput) in W/Mbps
- **v**: QoS violation rate during episode
- **n_sw**: Total type-switch count (DPDK↔OAI transitions)
- **n_sc**: Total scaling event count (OAI instance changes, scale_up + scale_down)
- **n_bl**: Accumulated cooldown blocking events
- **Standard deviation**: Computed across seeds using ddof=1 (sample std)

## Validation

✓ All metrics extracted from same evaluation setting (B2 test set)
✓ Seed 456 values match reported manuscript numbers where available
✓ No mixing of different observation schemas or cooldown configurations
✓ All seeds evaluated under identical environment semantics
