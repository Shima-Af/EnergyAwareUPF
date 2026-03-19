# Improved Traffic Predictor: Isolated Experiments

## Isolation and data safety

- Original dataset was NOT modified.
- Copied working dataset: `/home/ubuntu/EnergyAwareUPF/traffic_predictor/data/processed_traffic_normalized_copy_with_calendar.csv`
- All models/results saved under: `/home/ubuntu/EnergyAwareUPF/traffic_predictor`

## Base architecture

- Architecture template: `SpatialDropout1D(0.1) → LSTM(128, return_sequences=True) → LSTM(64, return_sequences=False) → Dropout(0.2) → Dense(64, relu) → Dropout(0.2) → Dense(1, linear)`

## Improved approaches evaluated

- `baseline_abs_nocal_mse`: Leak-free scaling + temporal split + absolute target | seq_len=96, calendar=False, target_mode=absolute, loss=mse, lr=0.001
- `delta_nocal_huber`: Residual/delta target + Huber loss | seq_len=96, calendar=False, target_mode=delta, loss=huber, lr=0.0005
- `abs_calendar_mse`: Calendar features enabled (copy only) + absolute target | seq_len=96, calendar=True, target_mode=absolute, loss=mse, lr=0.001
- `delta_calendar_huber`: Calendar features + residual target + Huber loss | seq_len=96, calendar=True, target_mode=delta, loss=huber, lr=0.0005

## Split strategy

- Temporal split (no shuffle): train / val / test = 0.65 / 0.15 / 0.20
- Scalers fit on TRAIN segment only (leakage-free).

## Model comparison (test set)

| model_name | rmse | mae | r2 | mape_safe | smape | wape | n_test |
|---|---:|---:|---:|---:|---:|---:|---:|
| delta_calendar_huber | 19.119999 | 13.835485 | 0.897953 | 17.424974 | 16.642336 | 13.752468 | 672 |
| delta_nocal_huber | 19.252972 | 13.956462 | 0.896528 | 17.833479 | 16.751066 | 13.872721 | 672 |
| baseline_abs_nocal_mse | 36.643743 | 28.890354 | 0.625178 | 55.564880 | 33.982025 | 28.717005 | 672 |
| abs_calendar_mse | 41.755919 | 36.067368 | 0.513299 | 78.822090 | 42.721539 | 35.850956 | 672 |

## Baselines (on best-model split)

| baseline | rmse | mae | r2 | mape_safe | smape | wape |
|---|---:|---:|---:|---:|---:|---:|
| naive_persistence_t-1 | 19.252984 | 13.970454 | 0.896528 | 17.474134 | 16.878580 | 13.886629 |
| naive_daily_t-96 | 38.358229 | 29.100859 | 0.589283 | 38.784836 | 33.188004 | 28.926247 |

## Best updated accuracy

- Best model: `delta_calendar_huber` with RMSE=19.119999, MAE=13.835485, R2=0.897953, MAPE_safe=17.424974, sMAPE=16.642336, WAPE=13.752468.

## Artifacts

- `results/model_accuracy_comparison.csv`
- `results/baseline_accuracy_comparison.csv`
- `results/all_model_predictions.csv`
- `results/best_model_predictions.csv`
- `results/best_model_actual_vs_predicted.png`
- `results/best_model_actual_vs_predicted.pdf`
