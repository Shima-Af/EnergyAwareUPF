# Forecast Model Accuracy Summary (Winner Traffic Model)

Date: 2026-03-19

## Scope (winner only)

This report is for the **winner traffic predictor model** from the isolated improvement workflow:

- Winner model: `delta_calendar_huber`
- Workflow: `traffic_predictor/run_improved_traffic_predictor.py`
- This is the model used to produce the final winner forecast accuracy table (not the original base `traffic_predictor.keras` report).

## 1) Forecasting pipeline used

### Scripts / artifacts used

- Winner training + evaluation pipeline: `traffic_predictor/run_improved_traffic_predictor.py`
- Winner comparison metrics: `traffic_predictor/results/model_accuracy_comparison.csv`
- Winner out-of-sample predictions: `traffic_predictor/results/best_model_predictions.csv`
- Winner summary: `traffic_predictor/README_RESULTS.md`

### Dataset used

- Source traffic dataset (unchanged): `data/processed/processed_traffic_normalized.csv`
- Isolated copy used for winner experiments: `traffic_predictor/data/processed_traffic_normalized_copy_with_calendar.csv`

### Split, window, horizon, and evaluation mode

From the winner pipeline:

- Input window length: `seq_len = 96`
- Forecast horizon: **t+1** (one-step ahead)
- Temporal split (no shuffle): `train/val/test = 0.65 / 0.15 / 0.20`
- Test evaluation mode: one-step-ahead predictions on held-out test suffix
- Rolling-horizon recursive multi-step evaluation: **No**

### Scaling/normalization

- Target/evaluation traffic signal: `Traffic_Mbps_scaled`
- Traffic normalization for model fitting: `MinMaxScaler`
- Inverse-transform is applied for metric reporting, returning to project `Traffic_Mbps_scaled` units
- Unit convention in this report: **project load-scaled traffic units** (`Traffic_Mbps_scaled`), not raw unscaled Mbps

## 2) Paper-setting linkage (forecast-augmented / hybrid schema)

- B2 sweep file: `experiments/sweep_config_b2_top2_seeds.yaml`
- B2 run configs use forecast-augmented/hybrid observations through:
  - `environment.forecast_column = Traffic_Predicted_Mbps`
  - `environment.forecast_horizon = 1`
- The winner report below is restricted to the winner traffic-model test split only and does not mix RL reward/energy tables.

## 3) Prediction extraction method

- Predictions were loaded from existing winner outputs: `traffic_predictor/results/best_model_predictions.csv`
- Accuracy CSVs in this report were regenerated from those stored winner predictions only
- Retraining was **not** run in this reporting step

## 4) Winner model architecture and setup

- Manuscript naming: **traffic predictor / CNN+LSTM**
- Winner implementation identifier: `delta_calendar_huber`
- Backbone used in winner run:
  - `SpatialDropout1D(0.1) -> LSTM(128, return_sequences=True) -> LSTM(64, return_sequences=False) -> Dropout(0.2) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, linear)`
- Winner-specific setup:
  - calendar features enabled (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`, `is_business_hours`)
  - residual/delta target (`target_mode=delta`)
  - Huber loss (`loss=huber`, `learning_rate=0.0005`)

## 5) Final winner accuracy table

From `forecast_model_accuracy.csv`:

| model_name | horizon | split_name | n_samples | rmse | mape | mae | r2 | unit |
|---|---|---|---:|---:|---:|---:|---:|---|
| delta_calendar_huber | t+1 (one-step ahead) | temporal_holdout_seq96_train0.65_val0.15_test0.20 | 672 | 19.120000 | 17.424974 | 13.835485 | 0.897953 | Traffic_Mbps_scaled (project load-scaled units; train-fitted MinMax inverse applied) |

## 6) Output files produced

- Accuracy CSV: `forecast_model_accuracy.csv`
- Summary markdown: `forecast_model_accuracy_summary.md`
- y_true/y_pred traceability CSV: `forecast_model_predictions_vs_target.csv`
