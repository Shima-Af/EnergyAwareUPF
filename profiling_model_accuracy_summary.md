# Profiling Model Accuracy (Digital Twin)

## Files and pipeline used

- Configuration: `/home/ubuntu/EnergyAwareUPF/config.yaml`
- Runtime profiling inference path: `src/utils.py` (`UPFApproximatorBundle.predict_batch`)
- Saved profiling artifacts (loaded, not refit):
  - `/home/ubuntu/EnergyAwareUPF/saved_models/prediction_models/dpdk_upf_power_model.keras`
  - `/home/ubuntu/EnergyAwareUPF/saved_models/prediction_models/dpdk_upf_performance_model.keras`
  - `/home/ubuntu/EnergyAwareUPF/saved_models/prediction_models/oai_upf_power_model.keras`
  - `/home/ubuntu/EnergyAwareUPF/saved_models/prediction_models/oai_upf_performance_model.keras`
  - scaler files in `/home/ubuntu/EnergyAwareUPF/saved_models/prediction_models`
- Profiling datasets:
  - DPDK: `/home/ubuntu/DDT_UPF_Selection_DRL/data/processed/dpdk_processed.csv`
  - USR: `/home/ubuntu/DDT_UPF_Selection_DRL/data/processed/oai_processed.csv`

## Training architectures (from scripts + config)

- Power model training behavior (as in `train_power_models.py`):
  - Uses `power_model_training.default` then applies `power_model_training.upf_specific[mode]` overrides.
  - DPDK attributed power architecture: `Dense(16, relu) → Dense(1, linear)`
  - USR attributed power architecture: `Dense(64, relu) → BatchNorm → Dropout(0.15) → Dense(32, relu) → BatchNorm → Dense(1, linear)`
  - Default power architecture (fallback): `Dense(64, relu) → BatchNorm → Dropout(0.15) → Dense(32, relu) → BatchNorm → Dense(1, linear)`

- QoS model training behavior (as in `train_performance_models.py`):
  - Uses one shared `performance_model_training.architecture` for both DPDK and USR.
  - QoS architecture: `Dense(64, relu) → BatchNorm → Dropout(0.2) → Dense(32, relu) → BatchNorm → Dense(16, relu) → Dense(1, linear)`

- Training hyperparameters from config/scripts:
  - Power: lr=0.0001, epochs=300, batch_size=32, early_stopping_patience=20, lr_patience=15
  - QoS: lr=0.001, epochs=300, batch_size=8, early_stopping_patience=25, lr_patience=10
  - Both scripts use the held-out split as `validation_data` during training.

## Held-out split and preprocessing used

- Split rule: `train_test_split(test_size=0.2, random_state=42, shuffle=True)`
- Input feature: `Objective Throughput (DL/UL)`
- Targets:
  - attributed power: `power_consumption_watts`
  - QoS score: `QoS_w`
- Input preprocessing: `log1p(offered_load)` then `StandardScaler` transform
- Target preprocessing: inverse `StandardScaler` transform after prediction
- Evaluation split name: `holdout_test_testsize0.2_rs42_shuffleTrue`

## MAPE handling

- Safe MAPE uses epsilon `0.001` in target units:
  - `MAPE_safe = mean(|y_true - y_pred| / max(|y_true|, 0.001)) * 100`

## Accuracy table

| mode | target | split_name | n_samples | rmse | mape | r2 | mae |
|---|---|---|---:|---:|---:|---:|---:|
| DPDK | QoS score | holdout_test_testsize0.2_rs42_shuffleTrue | 919 | 0.084656 | 5.569546 | 0.152856 | 0.042231 |
| DPDK | attributed power | holdout_test_testsize0.2_rs42_shuffleTrue | 919 | 0.003862 | 0.378906 | 0.239295 | 0.003110 |
| USR | QoS score | holdout_test_testsize0.2_rs42_shuffleTrue | 913 | 0.078475 | 186.409062 | 0.906817 | 0.043055 |
| USR | attributed power | holdout_test_testsize0.2_rs42_shuffleTrue | 913 | 0.303837 | 479.441664 | 0.960467 | 0.121736 |

## Traceability artifacts

- Summary metrics CSV: `profiling_model_accuracy.csv`
- Per-sample predictions CSV: `profiling_model_predictions_vs_target.csv`
