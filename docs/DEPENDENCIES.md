# Dependency Map

## Module-Level Dependencies

- `src/train.py`
  - depends on: `src/utils.py`, `src/agent.py`, callbacks
- `src/evaluate.py`
  - depends on: `src/utils.py`, `src/agent.py`, `src/environment.py`, matplotlib/pandas
- `src/utils.py`
  - depends on: pandas/numpy/sklearn, `src/environment.py`
- `src/environment.py`
  - depends on: gymnasium, numpy
- `src/agent.py`
  - depends on: stable-baselines3, sb3-contrib, torch

## Environment Dependencies (`ManualCooldownEnv`)

### Constructor inputs

- required:
  - `traffic_data`
  - `env_config`
  - `reward_config`
- predictor source:
  - precompute: `dpdk_lookup`, `oai_lookup`
  - runtime: `dpdk_bundle`, `oai_bundle`
- optional banks:
  - `extra_features`
  - `capacity_features`
  - `calendar_features`
  - `calendar_features_forecast`
  - `forecast_data` (required only when schema uses forecast)

### Config keys consumed

- `environment`:
  - `observation_schema`
  - `forecast_horizon`
  - `window_size`
  - `cooldown_period`
  - `performance_threshold`
  - `num_oai_instances`
  - `dpdk_idle_watts`
  - `oai_idle_watts_per_instance`
  - `use_dyn_features`
  - `use_capacity_features`
  - `use_powergap_features`
  - `use_calendar_features`
- `reward`:
  - `sec_scale`
  - `sec_eps_mbps`
  - `qos_lambda`
  - `type_switch_cost`
  - `scale_up_cost_per_inst`
  - `scale_down_cost_per_inst`

### Runtime method dependencies

- `_get_prediction`
  - uses lookup tables or model bundles
- `_get_observation`
  - uses `observation_schema` + feature flags
- `step`
  - uses cooldown logic, SEC reward terms, and current traffic

## Data Pipeline Dependencies (`utils.py`)

- `load_and_preprocess_data`
  - reads: traffic CSV + optional forecast column
  - computes: dynamics/capacity/powergap/calendar banks
  - outputs: train/test payloads aligned with schema
- `create_vectorized_envs`
  - wraps `ManualCooldownEnv` with `VecNormalize`
  - propagates `env_config` and `reward_config` directly

## Evaluation Dependencies (`evaluate.py`)

- consumes environment metadata:
  - `observation_schema`
  - `forecast_horizon`
  - `config_pos` / `cooldown_pos`
- uses `info` payload fields for timelines and summaries
- computes static baselines using lookup helpers

## Removed Deprecated Couplings

The current codepath no longer depends on:

- strategy-specific environment routing
- delayed action queue scheduling
- non-SEC reward branches
- internal `traffic_predictor_model` ownership in environment
