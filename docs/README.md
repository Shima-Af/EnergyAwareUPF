# EnergyAwareUPF - Complete Code Documentation

## Overview

**EnergyAwareUPF** is a Reinforcement Learning system that trains an agent to dynamically select between different UPF (User Plane Function) implementations—DPDK and OAI—to optimize energy consumption while maintaining QoS (Quality of Service).

The system supports multiple observation schemas:
- **instant**: current traffic at $t$ (+ optional dynamics/calendar)
- **history**: traffic window over past $W$ steps
- **forecast**: forecasted traffic at $t+H$
- **hybrid**: history + forecast

## Project Structure

```
src/
├── __init__.py              # Empty module init
├── agent.py                 # PPO/RecurrentPPO agent creation & loading
├── environment.py           # ManualCooldownEnv (gymnasium RL environment)
├── utils.py                 # Data loading, preprocessing, env creation
├── train.py                 # Training pipeline entry point
├── evaluate.py              # Evaluation pipeline with plots & baselines
└── tf_config.py             # TensorFlow configuration (suppress warnings)

config.yaml                   # Main configuration file
requirements.txt             # Python dependencies
```

## Key Concepts

### UPF Types
- **DPDK (0)**: Single data plane instance (baseline)
- **OAI (k)**: k×OAI instances (k = 1 to `num_oai_instances`)

### Observation Schemas
- **instant**: Uses traffic at current slot
- **history**: Uses historical traffic window
- **forecast**: Uses forecasted traffic
- **hybrid**: Uses historical window + forecast

### Simulation Modes
- **precompute**: Pre-compute all lookups at startup, store in memory
- **runtime**: Compute metrics on-the-fly using model bundles

### Reward Type
- **SEC-only**: Uses Specific Energy Consumption (SEC = W/Mbps) as primary metric

## Key Files Overview

See individual documentation:
- [API Reference](API_REFERENCE.md) - Detailed function signatures & docstrings
- [Call Graph](CALL_GRAPH.md) - Which functions call which
- [Dependencies](DEPENDENCIES.md) - Per-function dependency analysis
- [Data Flow](DATA_FLOW.md) - How data flows through the system
- [Module Documentation](MODULES.md) - Per-file deep dives

## Quick Start

### Training
```bash
source venv/bin/activate
python -m src.train
```

### Evaluation
```bash
python -m src.evaluate --run_dir saved_models/best_recurrent_ppo/20250908-124959/
```

### Configuration
Edit `config.yaml` to control:
- Traffic data paths
- Agent hyperparameters
- Environment settings (window size, forecast horizon, thresholds)
- Feature banks (dynamics, calendar, capacity margins, power gaps)
- Reward function parameters

## Key Data Structures

### precomputed_data (from utils.load_and_preprocess_data)
```python
{
    'train': {
        'traffic_data': np.ndarray,           # Shape (N,)
        'dpdk_lookup': dict,                  # {traffic -> (perf, power)}
        'oai_lookup': dict,                   # {'keys': arr, 'perf': arr, 'power': arr}
        'extra_features': dict,               # {'dT': arr, 'mov_mean_W': arr, ...}
        'capacity_features': dict,            # {'dpdk_margin': arr, 'oai_margin_k': [...], ...}
        'calendar_features': dict,            # {'sin_hour': arr, 'cos_hour': arr, ...}
        'forecast_data': np.ndarray,          # For forecast/hybrid schemas
    },
    'test': { ... same structure ... }
}
```

### Environment obs (state vector)
```
[traffic[t-W], ..., traffic[t-1], forecast[t+H]?, config_code, cooldown_counter, 
 optional: dynamics_features..., capacity_margins..., power_gaps..., calendar_features...]
```

### Environment info (step metadata)
```python
{
    'traffic': float,                 # Current traffic (Mbps)
    'chosen_upf': str,               # 'DPDK' or 'kxOAI'
    'power': float,                  # Power consumption (W)
    'performance': float,            # Performance score
    'reward': float,                 # Step reward
    'sec': float,                    # Specific energy consumption
    'type_switch_penalty': float,    # Cost of switching UPF type
    'scale_penalty': float,          # Cost of scaling OAI instances
    'qos_penalty': float,            # Penalty for QoS violation
    'executed_action': int,          # Executed action (after cooldown)
    'requested_action': int,         # Raw policy decision
    'cooldown_blocked': bool,        # Whether cooldown prevented action
    # ... many more fields for analysis
}
```

## Training Pipeline

```
config.yaml
    ↓
utils.load_and_preprocess_data()
    ├─ Calculate traffic dynamics (Δt, moving averages, EMAs)
    ├─ Load surrogate models (DPDK, OAI)
    ├─ Pre-compute lookups OR prep bundles
    ├─ Compute capacity margins & power gaps
    ├─ Compute calendar features (sin/cos hour, day-of-week)
    ├─ Split train/test
    └─ Return data_for_env dict
    ↓
utils.create_vectorized_envs()
    ├─ Create ManualCooldownEnv instances in parallel
    ├─ Wrap with VecEnv (parallelization)
    ├─ Apply VecNormalize (observation normalization)
    └─ Return (train_env, eval_env)
    ↓
agent.create_agent(train_env, config)
    ├─ Determine PPO vs RecurrentPPO
    ├─ Select device (cuda/cpu)
    ├─ Build hyperparameter dict
    └─ Return SB3 model
    ↓
model.learn(total_timesteps, callbacks)
    ├─ EvalCallback: periodic evaluation
    ├─ CheckpointCallback: save checkpoints
    ├─ SecLoggingCallback: log SEC metrics
    ├─ ResourceUsageCallback: track GPU/CPU
    └─ Return trained model
    ↓
Save: final_model.zip, vec_normalize_stats.pkl, logs/
```

## Evaluation Pipeline

```
--run_dir (e.g., saved_models/best_recurrent_ppo/20250908-124959/)
    ├─ Load config from run_dir/config.yaml
    ├─ Load vec_normalize_stats.pkl
    └─ Load best_model.zip
    ↓
utils.load_and_preprocess_data() [same as training]
    ↓
ManualCooldownEnv (n_envs=1)
    ├─ Apply VecNormalize (using training stats)
    └─ Ready for rollout
    ↓
model.predict() in rollout loop
    ├─ Process observations
    ├─ Select actions
    ├─ Step environment
    └─ Collect info dicts
    ↓
results CSV
    ├─ Per-step traffic, power, performance, actions, penalties, etc.
    └─ Save to eval/evaluation_results_*.csv
    ↓
Baseline comparisons (Always-DPDK, Always-kxOAI)
    ├─ Compute SEC, QoS violations, energy
    ├─ Compare vs learned policy
    └─ Generate summary_*.csv
    ↓
Plots (figures/)
    ├─ action_timeline_*.png
    ├─ rl_switching_timeline_*.png
    ├─ energy_comparison_*.png
    └─ Traffic + power + actions over time
```

## Important Configuration Sections

### environment
- `traffic_column`: CSV column with actual traffic
- `forecast_column`: CSV column with predicted traffic (forecast/hybrid schemas)
- `window_size` (W): Historical traffic window
- `forecast_horizon` (H): Forecast horizon for forecast/hybrid schemas
- `performance_threshold`: QoS threshold
- `cooldown_period`: Min steps between consecutive switches
- `num_oai_instances` (K): Max OAI instances to consider

### agent
- `policy`: 'MlpPolicy' (PPO) or 'MlpLstmPolicy' (RecurrentPPO)
- `learning_rate`, `n_steps`, `batch_size`, etc.: Standard SB3 hyperparameters

### environment.observation_schema
- `observation_schema`: 'instant' | 'history' | 'forecast' | 'hybrid'

### simulation_mode
- `type`: 'precompute' or 'runtime'

### environment feature flags
- `use_dyn_features`: Enable traffic dynamics (Δt, EMA, moving avg)
- `use_capacity_features`: Enable QoS margins per option
- `use_powergap_features`: Enable power difference features
- `use_calendar_features`: Enable hour/day-of-week features

### approximator_models
- Specifies if DPDK/OAI models are 'keras' or 'polynomial'

## Next Steps

1. **Read [CALL_GRAPH.md](CALL_GRAPH.md)** to see which functions call which
2. **Read [DEPENDENCIES.md](DEPENDENCIES.md)** to understand per-function dependencies
3. **Read [DATA_FLOW.md](DATA_FLOW.md)** for detailed data transformations
4. **Read module-specific docs** in [MODULES.md](MODULES.md)
5. **Reference [API_REFERENCE.md](API_REFERENCE.md)** for function signatures
