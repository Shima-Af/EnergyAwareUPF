# Module Deep Dives

## `src/environment.py`

### Purpose
Gymnasium environment for UPF configuration control with cooldown constraints and SEC-focused reward.

### Key model choices

- Observation is schema-driven (`observation_schema`):
  - `instant`, `history`, `forecast`, `hybrid`
- Action semantics are immediate (no lead-time queue).
- Reward is SEC-only:
  - `-sec_scale * sec - switch_penalties - qos_penalty`
- Forecast is external data input, not an internal prediction model.

### Observation layout

- Base fields depend on schema, always include:
  - current config code
  - cooldown counter
- Optional appended banks (when enabled):
  - dynamics
  - capacity margins
  - power gaps
  - calendar

### Step behavior

1. Apply cooldown gating.
2. Execute resulting config.
3. Predict performance/power.
4. Compute SEC and penalties.
5. Return next observation and rich `info`.

## `src/utils.py`

### Purpose
Data loading, preprocessing, feature-bank generation, and vectorized env creation.

### Main responsibilities

- Load traffic and optional forecast arrays.
- Align splits for train/test.
- Build predictor assets (`precompute` lookups or `runtime` bundles).
- Compute feature banks controlled by `environment.use_*_features`.
- Create train/eval vectorized envs with `VecNormalize`.

### Notes

- `forecast_data` is required only when schema is `forecast` or `hybrid`.
- Uses `observation_schema` as the primary observation interface.

## `src/agent.py`

### Purpose
Create/load PPO or RecurrentPPO agents.

### Notes

- Supports both feed-forward and LSTM policies.
- Environment API is now policy-agnostic (A2C/DQN-compatible observation interface).

## `src/train.py`

### Purpose
Training orchestration.

### Flow

1. Load config.
2. Preprocess data.
3. Build vectorized envs.
4. Create agent.
5. Train with callbacks.
6. Save model and normalization stats.

## `src/evaluate.py`

### Purpose
Evaluate trained policy on test split and generate artifacts.

### Outputs

- `evaluation_results_*.csv`
- `timeline_*.csv`
- `evaluation_summary_*.csv`
- `feature_manifest.json`
- `figures/*.png`

### Metadata tracked

- `observation_schema`
- `forecast_horizon`
- feature-bank names and positions

## `config.yaml` (relevant sections)

- `environment.observation_schema`
- `environment.forecast_horizon`
- `environment.use_dyn_features`
- `environment.use_capacity_features`
- `environment.use_powergap_features`
- `environment.use_calendar_features`
- `reward.*` (SEC and penalties)

## Extension guidance

- To add an observation variant: extend `_get_observation` in `environment.py` and update manifest generation in `evaluate.py`.
- To add a feature bank: compute in `utils.py`, append in `environment.py`, expose in `info`.
- To change reward: update SEC terms in `step` and keep outputs consistent in evaluation summaries.
