# Traffic Prediction Accuracy (Digital Twin)

## Files and pipeline used

- Configuration: `/home/ubuntu/EnergyAwareUPF/config.yaml`
- Training/prediction logic basis: provided `train_traffic_predictor.py` and `predict_traffic.py` scripts
- Saved traffic predictor artifact: `/home/ubuntu/EnergyAwareUPF/saved_models/prediction_models/traffic_predictor.keras`
- Traffic scaler artifact: `/home/ubuntu/EnergyAwareUPF/saved_models/prediction_models/traffic_predictor_scaler.pkl`
- Metadata artifact: `/home/ubuntu/EnergyAwareUPF/saved_models/prediction_models/traffic_predictor_metadata.pkl`
- Calendar scaler artifact: `/home/ubuntu/EnergyAwareUPF/saved_models/prediction_models/traffic_predictor_calendar_scaler.pkl`
- Traffic dataset: `/home/ubuntu/EnergyAwareUPF/data/processed/processed_traffic_normalized.csv`

## Training architecture (from scripts + config + saved artifact)

- Configured traffic predictor architecture: `SpatialDropout1D(0.1) → LSTM(128, return_sequences=True) → LSTM(64, return_sequences=False) → Dropout(0.2) → Dense(64, relu) → Dropout(0.2) → Dense(1, linear)`
- Saved model layer stack: `SpatialDropout1D(0.1) → LSTM(128) → LSTM(64) → Dropout(0.2) → Dense(64) → Dropout(0.2) → Dense(1)`
- Sequence length used: `96`
- Calendar features used: `False`

## Held-out split and preprocessing used

- Sequence construction follows training script: uses past `seq_len` timesteps to predict next timestep.
- Split rule: `train_test_split(test_size=0.2, random_state=42, shuffle=False)`
- Input preprocessing: `MinMaxScaler` transform on full traffic series before sequence construction.
- Target space during training/evaluation: `Traffic_Mbps_scaled`.
- Inverse-transform step undoes only `MinMaxScaler` normalization and returns values in `Traffic_Mbps_scaled` units.
- This does **not** undo upstream traffic-load scaling performed during data preprocessing.
- Evaluation split name: `holdout_test_seq96_testsize0.2_rs42_shuffleFalse`

## Training hyperparameters (config)

- learning_rate=0.001
- epochs=150
- batch_size=128
- test_size=0.2
- random_state=42
- validation_data uses the exact held-out split (per training script).

## MAPE handling

- Safe MAPE uses epsilon `0.001` in `Traffic_Mbps_scaled` units:
  - `MAPE_safe = mean(|y_true - y_pred| / max(|y_true|, 0.001)) * 100`

## Accuracy table

| model_name | target | split_name | n_samples | rmse | mape | r2 | mae |
|---|---|---|---:|---:|---:|---:|---:|
| traffic_predictor.keras | Traffic_Mbps_scaled (MinMax inverse only; load-scaling preserved) | holdout_test_seq96_testsize0.2_rs42_shuffleFalse | 653 | 36.145669 | 54.943817 | 0.637611 | 28.294836 |

## Traceability artifacts

- Summary metrics CSV: `traffic_prediction_accuracy.csv`
- Per-sample predictions CSV: `traffic_prediction_predictions_vs_target.csv`
