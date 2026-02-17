# API Reference

## `src/environment.py`

### `class ManualCooldownEnv(gym.Env)`

Gym environment for UPF action selection with cooldown and SEC reward.

#### `__init__(...)`

```python
def __init__(
    self,
    traffic_data,
    env_config,
    reward_config,
    dpdk_lookup=None,
    oai_lookup=None,
    dpdk_bundle=None,
    oai_bundle=None,
    extra_features=None,
    forecast_data=None,
    capacity_features=None,
    calendar_features=None,
    calendar_features_forecast=None,
)
```

Key behavior:
- Reads `environment.observation_schema` (`instant|history|forecast|hybrid`).
- Requires `forecast_data` only for `forecast`/`hybrid`.
- Supports precompute lookups or runtime bundles.

#### `reset(seed=None, options=None)`
Returns `(obs, info)` and initializes:
- DPDK start state
- cooldown counter
- start index by schema (history/hybrid use `window_size-1`, others start at `0`)

#### `step(action)`
Returns `(obs, reward, terminated, truncated, info)`.

Reward (SEC-only):

```text
reward = -sec_scale * sec - type_switch_penalty - scale_penalty - qos_penalty
```

#### `get_current_config_code()`
- `0` for DPDK
- `k` for `k×OAI`

#### `_get_prediction(traffic_val, upf_type_code, num_active_oai=0)`
Predicts `(performance, power)` using lookup or bundle path.

#### `_get_state()` / `_get_observation()`
Builds schema-dependent observation plus optional feature banks.

## `src/utils.py`

### `load_config(path="config.yaml")`
Loads YAML config, optionally via `XRL_CONFIG`.

### `load_and_preprocess_data(config)`
Returns `data_for_env` with `train` and `test` payloads:
- `traffic_data`
- predictor source (`dpdk_lookup/oai_lookup` or bundles)
- optional `forecast_data`
- optional feature banks

### `create_vectorized_envs(config, data_for_env)`
Creates train/eval vec envs with `ManualCooldownEnv` + `VecNormalize`.

### `compute_traffic_dynamics(traffic, window_size, ema_short=4, ema_long=16)`
Returns dynamic feature arrays.

### `compute_option_features(traffic, dpdk_lookup, oai_lookup, perf_threshold, max_k, perf_system_level=False)`
Returns capacity margins and power-gap features.

### `compute_calendar_features(df, timestamp_col, fmt=None)`
Returns cyclical calendar arrays: `sin_hour`, `cos_hour`, `sin_dow`, `cos_dow`.

## `src/agent.py`

### `create_agent(train_env, config, tensorboard_log_dir=None)`
Creates PPO/RecurrentPPO according to config policy.

### `load_agent(model_path, env=None)`
Loads saved policy for evaluation.

## `src/train.py`

### `train(config_file="config.yaml")`
Training pipeline entry point.

## `src/evaluate.py`

### `evaluate()`
Evaluation pipeline entry point.

Main artifacts:
- `evaluation_results_*.csv`
- `timeline_*.csv`
- `evaluation_summary_*.csv`
- `feature_manifest.json`
- `figures/*.png`

### `analyze_and_plot_results(df, config, stamp=None)`
Builds summary statistics and plots.

### `predict_dpdk_from_lookup(dpdk_lookup, T)`
Baseline helper.

### `predict_oai_from_lookup(oai_lookup, T, k)`
Baseline helper.

## Config Keys (Current)

### `environment`
- `traffic_column`
- `forecast_column`
- `window_size`
- `observation_schema`
- `forecast_horizon`
- `cooldown_period`
- `performance_threshold`
- `num_oai_instances`
- `use_dyn_features`
- `use_capacity_features`
- `use_powergap_features`
- `use_calendar_features`

### `reward`
- `sec_scale`
- `sec_eps_mbps`
- `qos_lambda`
- `type_switch_cost`
- `scale_up_cost_per_inst`
- `scale_down_cost_per_inst`
