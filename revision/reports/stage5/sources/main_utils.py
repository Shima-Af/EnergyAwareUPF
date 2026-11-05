# src/utils.py

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras # pylint: disable=no-name-in-module
from sklearn.model_selection import train_test_split
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Make sure the relative import is correct
from .environment import ManualCooldownEnv

def load_config(path="config.yaml"):
    """Loads the YAML configuration file."""
    cfg_path = os.getenv("XRL_CONFIG", path)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def _nearest_idx_sorted(keys: np.ndarray, x: float) -> int:
    idx = np.searchsorted(keys, x, side='left')
    if idx == 0: return 0
    if idx >= len(keys): return len(keys)-1
    return idx if (keys[idx] - x) < (x - keys[idx-1]) else idx-1

def compute_option_features(
    traffic: np.ndarray,
    dpdk_lookup: dict,
    oai_lookup: dict,
    perf_threshold: float,
    max_k: int,
    perf_system_level: bool = False
) -> dict:
    """
    Vectorized per-step features from precomputed lookups:
      - perf margins vs threshold for DPDK and k×OAI (k=1..K)
      - total power per option
      - power gaps between adjacent options (DPDK↔1x, 1x↔2x, ...)
    Returns float32 arrays aligned with traffic (len = N).
    """
    T = traffic.astype(np.float32)
    N = len(T)

    # DPDK: pack keys/vals once
    dpdk_keys = np.array(sorted(dpdk_lookup.keys()), dtype=np.float32)
    dpdk_vals = np.array([dpdk_lookup[float(k)] for k in dpdk_keys], dtype=np.float32)  # (perf, power)

    dpdk_perf = np.empty(N, dtype=np.float32)
    dpdk_pow  = np.empty(N, dtype=np.float32)

    for i, t in enumerate(T):
        j = _nearest_idx_sorted(dpdk_keys, t)
        dpdk_perf[i] = dpdk_vals[j, 0]
        dpdk_pow[i]  = dpdk_vals[j, 1]

    # OAI: arrays
    keys   = oai_lookup['keys']   # per-instance T
    perfs  = oai_lookup['perf']   # per-instance perf
    powers = oai_lookup['power']  # per-instance power

    oai_perf_k = []
    oai_pow_k  = []

    for k in range(1, max_k+1):
        per_inst = T / k
        idx = np.searchsorted(keys, per_inst, side='left')
        # clamp to nearest
        idx = np.clip(idx, 0, len(keys)-1)
        # adjust to nearer neighbor
        left_ok = idx > 0
        right_ok = idx < len(keys)
        choose_left = left_ok & ( (per_inst - keys[np.maximum(idx-1,0)]) <
                                  (keys[idx] - per_inst) )
        nearest = np.where(choose_left, idx-1, idx)

        perf_inst = perfs[nearest]
        pow_inst  = powers[nearest]
        if perf_system_level:
            perf_sys = k * perf_inst
        else:
            perf_sys = perf_inst  # your env treats perf per instance as the QoS score
        oai_perf_k.append(perf_sys.astype(np.float32))
        oai_pow_k.append( (k * pow_inst).astype(np.float32) )

    # Margins to threshold (positive = above threshold, safe)
    dpdk_margin = (dpdk_perf - perf_threshold).astype(np.float32)
    oai_margin_k = [ (arr - perf_threshold).astype(np.float32) for arr in oai_perf_k ]

    # Power gaps between options
    gaps = {}
    # gap_0_1 = dpdk - 1xOAI
    if max_k >= 1:
        gaps['gap_0_1'] = (dpdk_pow - oai_pow_k[0]).astype(np.float32)
    # gap_1_2 = 1x - 2x
    for k in range(1, max_k):
        gaps[f'gap_{k}_{k+1}'] = (oai_pow_k[k-1] - oai_pow_k[k]).astype(np.float32)

    return {
        'dpdk_perf': dpdk_perf, 'dpdk_power': dpdk_pow, 'dpdk_margin': dpdk_margin,
        'oai_perf_k': oai_perf_k, 'oai_power_k': oai_pow_k, 'oai_margin_k': oai_margin_k,
        'power_gaps': gaps,
        'feature_names': (
            ['dpdk_margin'] +
            [f'oai{k}_margin' for k in range(1, max_k+1)] +
            (['gap_0_1'] if max_k>=1 else []) +
            [f'gap_{k}_{k+1}' for k in range(1, max_k)]
        )
    }

def compute_calendar_features(df: pd.DataFrame, timestamp_col: str, fmt: str | None = None) -> dict:
    """
    Return sin/cos hour-of-day and day-of-week if a timestamp column exists.
    Aligned arrays (float32) with len == len(df).
    """
    if timestamp_col not in df.columns:
        return {'feature_names': []}
    ts = pd.to_datetime(df[timestamp_col], format=fmt, errors='coerce')
    # fallback if parse failed
    if ts.isna().all():
        return {'feature_names': []}

    hour = ts.dt.hour.to_numpy()
    dow  = ts.dt.dayofweek.to_numpy()

    # 24h cycle
    hour_rad = 2*np.pi*hour/24.0
    sin_h = np.sin(hour_rad).astype(np.float32)
    cos_h = np.cos(hour_rad).astype(np.float32)

    # weekly cycle (Mon=0..Sun=6)
    dow_rad = 2*np.pi*dow/7.0
    sin_d = np.sin(dow_rad).astype(np.float32)
    cos_d = np.cos(dow_rad).astype(np.float32)

    return {
        'sin_hour': sin_h,
        'cos_hour': cos_h,
        'sin_dow' : sin_d,
        'cos_dow' : cos_d,
        'feature_names': ['sin_hour','cos_hour','sin_dow','cos_dow']
    }

class UPFApproximatorBundle:
    """
    A generic bundle to load and manage different types of UPF approximator models
    (e.g., Keras Neural Networks or Polynomial Regressors).
    """
    _model_cache = {}

    def __init__(self, upf_name, model_dir, model_type='keras'):
        """
        Initializes and loads the models based on the specified type.

        Args:
            upf_name (str): The name of the UPF (e.g., 'dpdk', 'oai').
            model_dir (str): The directory containing the model files.
            model_type (str): The type of model to load ('keras' or 'polynomial').
        """
        cache_key = f"{upf_name}_{model_dir}_{model_type}"
        if cache_key in UPFApproximatorBundle._model_cache:
            self.__dict__.update(UPFApproximatorBundle._model_cache[cache_key])
            print(f"✓ Loaded {upf_name.upper()} UPF models ({model_type}) from cache.")
            return

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        print(f"-> Loading {model_type.capitalize()} models for {upf_name.upper()} UPF from {model_dir}...")
        self.name = upf_name
        self.model_type = model_type

        if self.model_type == 'keras':
            # Load Keras models and their corresponding scalers
            self.perf_model = keras.models.load_model(os.path.join(model_dir, f'{upf_name}_upf_performance_model.keras'))
            self.power_model = keras.models.load_model(os.path.join(model_dir, f'{upf_name}_upf_power_model.keras'))
            self.perf_scaler_X = joblib.load(os.path.join(model_dir, f'{upf_name}_perf_scaler_X.pkl'))
            self.perf_scaler_y = joblib.load(os.path.join(model_dir, f'{upf_name}_perf_scaler_y.pkl'))
            self.power_scaler_X = joblib.load(os.path.join(model_dir, f'{upf_name}_scaler_X.pkl'))
            self.power_scaler_y = joblib.load(os.path.join(model_dir, f'{upf_name}_scaler_y.pkl'))
        elif self.model_type == 'polynomial':
            # Load pickled polynomial models. Assumes they were trained on log-transformed data.
            self.perf_model = joblib.load(os.path.join(model_dir, f'{upf_name}_performance_model.pkl'))
            self.power_model = joblib.load(os.path.join(model_dir, f'{upf_name}_power_model.pkl'))
            # Polynomial models from scikit-learn pipelines often don't need separate scalers.
        else:
            raise ValueError(f"Unsupported model_type '{self.model_type}' in config. Use 'keras' or 'polynomial'.")
        
        UPFApproximatorBundle._model_cache[cache_key] = self.__dict__
        print(f"✓ Successfully loaded and cached models for {upf_name.upper()} UPF.")

    def predict_batch(self, throughput_array):
        """
        Predicts performance and power for a batch of throughput values.
        """
        if throughput_array.size == 0:
            return np.array([]), np.array([])
        
        performance, power = None, None

        # Reshape and apply log transform, which is common to both model types
        throughput_2d = throughput_array.reshape(-1, 1).astype(np.float32)
        throughput_log = np.log1p(throughput_2d)

        if self.model_type == 'keras':
            perf_input_scaled = self.perf_scaler_X.transform(throughput_log)
            performance = self.perf_scaler_y.inverse_transform(self.perf_model.predict(perf_input_scaled, verbose=0)).flatten()
            
            power_input_scaled = self.power_scaler_X.transform(throughput_log)
            power = self.power_scaler_y.inverse_transform(self.power_model.predict(power_input_scaled, verbose=0)).flatten()
        
        elif self.model_type == 'polynomial':
            # Assumes the pickled model is a trained pipeline that expects log-transformed input
            performance = self.perf_model.predict(throughput_log).flatten()
            power = self.power_model.predict(throughput_log).flatten()
            
        return performance, power


# Alias for convenience
ModelBundle = UPFApproximatorBundle


def load_precomputed_lookups(model_dir: str, config: dict):
    """
    Load precomputed lookup tables for DPDK and OAI.
    
    This is a simplified version of load_and_preprocess_data() that only
    generates the lookup tables without loading traffic data or splitting.
    Useful for digital twin and custom evaluation scenarios.
    
    Args:
        model_dir: Directory containing the surrogate models
        config: Configuration dictionary
        
    Returns:
        tuple: (dpdk_lookup, oai_lookup)
    """
    print("-> Loading surrogate models for lookup generation...")
    
    approximator_config = config.get('approximator_models', {})
    dpdk_model_type = approximator_config.get('dpdk', {}).get('type', 'keras')
    oai_model_type = approximator_config.get('oai', {}).get('type', 'keras')
    
    dpdk_bundle = UPFApproximatorBundle('dpdk', model_dir=model_dir, model_type=dpdk_model_type)
    oai_bundle = UPFApproximatorBundle('oai', model_dir=model_dir, model_type=oai_model_type)
    
    env_config = config.get('environment', {})
    max_oai_instances = env_config.get('num_oai_instances', 1)
    
    print("-> Generating lookup tables for common traffic range...")
    
    # Generate a comprehensive traffic range (0 to 1000 Mbps, sampled)
    traffic_values = np.linspace(0, 1000, 500)
    
    # DPDK lookup
    dpdk_perf, dpdk_power = dpdk_bundle.predict_batch(traffic_values)
    dpdk_lookup = dict(zip(traffic_values, zip(dpdk_perf, dpdk_power)))
    
    # OAI lookup - generate per-instance values
    def get_oai_precompute_keys(traffic_values):
        if max_oai_instances <= 1:
            return np.unique(traffic_values)
        
        all_per_instance_traffic = set()
        for n in range(1, max_oai_instances + 1):
            all_per_instance_traffic.update(traffic_values / n)
        
        return np.sort(list(all_per_instance_traffic))
    
    oai_keys = get_oai_precompute_keys(traffic_values)
    oai_perf, oai_power = oai_bundle.predict_batch(oai_keys)
    
    oai_lookup = {
        'keys': oai_keys.astype(np.float32),
        'perf': oai_perf.astype(np.float32),
        'power': oai_power.astype(np.float32)
    }
    
    print(f"✓ Generated DPDK lookup with {len(dpdk_lookup)} entries")
    print(f"✓ Generated OAI lookup with {len(oai_keys)} per-instance values")
    
    return dpdk_lookup, oai_lookup


def load_and_preprocess_data(config):
    """
    Loads data and prepares it for the environment based on the simulation_mode.
    """
    print("\033[94m\n--- 1. Loading Data & Preparing Environment Payload ---\033[0m")
    paths_config = config['paths']
    training_config = config['training']
    env_config = config['environment'] 
    traffic_col  = env_config.get('traffic_column',  'Traffic_Mbps_scaled')
    forecast_col = env_config.get('forecast_column', None)
    
    # Load common data
    traffic_df = pd.read_csv(paths_config['traffic_data_csv'])

    if traffic_col not in traffic_df.columns:
        raise KeyError(f"Traffic column '{traffic_col}' not found in CSV.")
    
    real_traffic_data = traffic_df[traffic_col].to_numpy().astype(np.float32)
    print(f"✓ Loaded {len(real_traffic_data)} traffic samples.")

    forecast_traffic_data = None
    if forecast_col:
        if forecast_col not in traffic_df.columns:
            raise KeyError(f"Forecast column '{forecast_col}' not found in CSV.")
        # keep NaNs if present; we’ll align below
        forecast_traffic_data = traffic_df[forecast_col].to_numpy().astype(np.float32)
        print(f"✓ Loaded {len(forecast_traffic_data)} forecast traffic samples.")

    # --- MODIFIED: Load bundles based on config ---
    approximator_config = config.get('approximator_models', {}) # Gracefully handle if section is missing
    dpdk_model_type = approximator_config.get('dpdk', {}).get('type', 'keras')
    oai_model_type = approximator_config.get('oai', {}).get('type', 'keras')

    dpdk_bundle = UPFApproximatorBundle('dpdk',
                                         model_dir=paths_config['prediction_model_dir'],
                                         model_type=dpdk_model_type)
    oai_bundle = UPFApproximatorBundle('oai',
                                        model_dir=paths_config['prediction_model_dir'],
                                        model_type=oai_model_type)
    
    W = int(env_config['window_size'])
    schema = env_config.get("observation_schema")
    if schema is None:
        feature_mode = env_config.get("feature_mode", "both")
        schema = {
            "historical_only": "history",
            "forecast_only": "forecast",
            "both": "hybrid",
        }.get(feature_mode, "hybrid")
    needs_history = schema in {"history", "hybrid"}
    needs_forecast = schema in {"forecast", "hybrid"}

    if needs_forecast and forecast_traffic_data is None:
        raise ValueError("forecast_column is required for forecast or hybrid schemas.")

    # ── Uniform trimming (schema-independent) ──────────────────────
    # Always apply the MOST restrictive start_idx so that ALL schemas
    # (instant, history, forecast, hybrid) train and evaluate on the
    # exact same data slice.  This is critical for fair cross-schema
    # comparisons in ablation studies.
    #
    # Requirement A: W-1 past points for history window
    i0_hist = W - 1

    # Requirement B: first valid (non-NaN) forecast value
    i0_fore = 0
    if forecast_traffic_data is not None:
        valid_fore_idx = np.flatnonzero(~np.isnan(forecast_traffic_data))
        if valid_fore_idx.size > 0:
            i0_fore = int(valid_fore_idx[0])
        else:
            i0_fore = len(real_traffic_data)

    start_idx = max(i0_hist, i0_fore)

    # Also, trim tails so traffic and forecast are same length if both exist
    end_len = len(real_traffic_data)
    if forecast_traffic_data is not None:
        end_len = min(end_len, len(forecast_traffic_data))

    # Apply head/tail alignment
    real_traffic_data  = real_traffic_data[start_idx:end_len]
    if forecast_traffic_data is not None:
        forecast_traffic_data = forecast_traffic_data[start_idx:end_len]

    # Optional: drop remaining NaNs in forecast by forward-fill or keep and let proactive raise
    if forecast_traffic_data is not None:
        # forward-fill remaining NaNs for stability (comment out if you prefer to raise)
        
        forecast_traffic_data = pd.Series(forecast_traffic_data).ffill().bfill().to_numpy(dtype=np.float32)

    print(f"\033[95m✓ Aligned arrays from index {start_idx}; N={len(real_traffic_data)} samples after trim.\033[0m")

    train_traffic, test_traffic = train_test_split(
    real_traffic_data, test_size=training_config['test_size'], shuffle=False, random_state=training_config['seed']
    )

    train_forecast = test_forecast = None
    if forecast_traffic_data is not None:
        train_forecast, test_forecast = train_test_split(
            forecast_traffic_data, test_size=training_config['test_size'], shuffle=False, random_state=training_config['seed']
        )

    # --- Feature families (optional, controlled by env_config flags) ---
    use_dynamics  = bool(env_config.get("use_dyn_features", True))
    use_capacity  = bool(env_config.get("use_capacity_features", False))
    use_powergap  = bool(env_config.get("use_powergap_features", False))
    use_calendar  = bool(env_config.get("use_calendar_features", True))
    train_dyn = test_dyn = None
    if use_dynamics:
        W = int(env_config['window_size'])
        ema_s = int(env_config.get('ema_short', 4))
        ema_l = int(env_config.get('ema_long', 16))
        train_dyn = compute_traffic_dynamics(train_traffic, W, ema_s, ema_l)
        test_dyn  = compute_traffic_dynamics(test_traffic,  W, ema_s, ema_l)

    simulation_type = config.get('simulation_mode', {}).get('type', 'precompute')

    if simulation_type == "precompute":
        print("-> Pre-computation mode selected. Creating lookup tables...")
        unique_train_traffic = np.unique(train_traffic)
        unique_test_traffic = np.unique(test_traffic)

        # --- DPDK Pre-computation (unchanged) ---
        train_dpdk_perf, train_dpdk_power = dpdk_bundle.predict_batch(unique_train_traffic)
        test_dpdk_perf, test_dpdk_power = dpdk_bundle.predict_batch(unique_test_traffic)

        # --- OAI Multi-Instance Pre-computation (## NEW LOGIC ##) ---
        max_oai_instances = env_config.get('num_oai_instances', 1)
        
        # Helper function to generate all possible per-instance traffic values
        def get_oai_precompute_keys(traffic_values):
            if max_oai_instances <= 1:
                return np.unique(traffic_values)
            
            # Create a set of all traffic values divided by 1, 2, ..., N instances
            all_per_instance_traffic = set()
            for n in range(1, max_oai_instances + 1):
                all_per_instance_traffic.update(traffic_values / n)
            
            # Return a sorted array of unique values
            return np.sort(list(all_per_instance_traffic))
        
        # Generate the keys and predict for train and test sets
        train_oai_keys = get_oai_precompute_keys(unique_train_traffic)
        train_oai_perf, train_oai_power = oai_bundle.predict_batch(train_oai_keys)
        
        test_oai_keys = get_oai_precompute_keys(unique_test_traffic)
        test_oai_perf, test_oai_power = oai_bundle.predict_batch(test_oai_keys)
        print(f"✓ Pre-computed {len(train_oai_keys)} values for OAI (up to {max_oai_instances} instances).")

        # --- Create Payloads with new structure for OAI ---
        train_payload = {
            'dpdk_lookup': dict(zip(unique_train_traffic, zip(train_dpdk_perf, train_dpdk_power))),
            # ## MODIFIED ##: Store OAI lookup as sorted arrays for efficient searching
            'oai_lookup': {
                'keys': train_oai_keys.astype(np.float32),
                'perf': train_oai_perf.astype(np.float32),
                'power': train_oai_power.astype(np.float32)
            },
            'extra_features': train_dyn if use_dynamics else None
        }
        test_payload = {
            'dpdk_lookup': dict(zip(unique_test_traffic, zip(test_dpdk_perf, test_dpdk_power))),
            # ## MODIFIED ##: Store OAI lookup as sorted arrays for efficient searching
            'oai_lookup': {
                'keys': test_oai_keys.astype(np.float32),
                'perf': test_oai_perf.astype(np.float32),
                'power': test_oai_power.astype(np.float32)
            },
            'extra_features': test_dyn if use_dynamics else None
        }
         

        max_k = int(env_config.get('num_oai_instances', 1))
        thr = float(env_config['performance_threshold'])
        perf_system_level = bool(env_config.get('perf_system_level', False))

        # calendar features (same indexing must align to the split)
        cal_train = {}
        cal_test  = {}
        if use_calendar:
            ts_col = env_config.get('timestamp_column', None)
            ts_fmt = env_config.get('timestamp_format', None)
            if ts_col and ts_col in traffic_df.columns:
                cal_all = compute_calendar_features(traffic_df, ts_col, ts_fmt)
                if cal_all.get('feature_names'):
                    # split calendar arrays the same way as traffic (no shuffle; same indices)
                    idx_split = len(train_traffic)
                    cal_train = {k: v[:idx_split] for k, v in cal_all.items() if k!='feature_names'}
                    cal_test  = {k: v[idx_split:] for k, v in cal_all.items() if k!='feature_names'}
                    cal_train['feature_names'] = cal_all['feature_names']
                    cal_test['feature_names']  = cal_all['feature_names']

        # capacity/powergap features (need lookups)
        train_opt = {}
        test_opt  = {}
        if use_capacity or use_powergap:
            train_opt = compute_option_features(
                train_traffic, 
                dict(train_payload['dpdk_lookup']),  # safe copy
                train_payload['oai_lookup'],
                thr, max_k, perf_system_level
            )
            test_opt = compute_option_features(
                test_traffic,
                dict(test_payload['dpdk_lookup']),
                test_payload['oai_lookup'],
                thr, max_k, perf_system_level
            )
        # attach into payloads
        train_payload['capacity_features'] = train_opt if (use_capacity or use_powergap) else {}
        train_payload['calendar_features'] = cal_train if use_calendar else {}
        test_payload['capacity_features']  = test_opt if (use_capacity or use_powergap) else {}
        test_payload['calendar_features']  = cal_test if use_calendar else {}
        train_payload.update({
            'forecast_data': train_forecast,                          # raw forecast, aligned by slot
    
            'calendar_features_forecast': cal_train or {},            # calendar is safe; env will index at j
        })

        test_payload.update({
            'forecast_data': test_forecast,
            'calendar_features_forecast': cal_test or {},
        })


    else:  # 'runtime' mode
        print("-> Runtime mode selected. Passing model bundles directly to environment...")
        
        train_payload = {
            'dpdk_bundle': dpdk_bundle,
            'oai_bundle': oai_bundle,
            'extra_features': train_dyn
        }
        test_payload = {
            'dpdk_bundle': dpdk_bundle,
            'oai_bundle': oai_bundle,
            'extra_features': test_dyn
        }

    # Assemble the final dictionary to be returned
    data_for_env = {
        'train': {
            'traffic_data': train_traffic,
            **train_payload
        },
        'test': {
            'traffic_data': test_traffic,
            **test_payload
        }
    }
    return data_for_env

def create_vectorized_envs(config, data_for_env):
    """Creates correctly wrapped and vectorized training and evaluation environments."""
    print("\033[94m\n--- 2. Creating Vectorized Environments ---\033[0m")
    
    train_env_kwargs = {
        **data_for_env['train'],
        'env_config': config['environment'],
        'reward_config': config['reward']
    }
    
    num_cpu = min(config['training']['num_cpu'], os.cpu_count() or 1)
    print(f"-> Using {num_cpu} parallel environments for training.")
    train_env = make_vec_env(ManualCooldownEnv, n_envs=num_cpu, seed=config['training']['seed'], env_kwargs=train_env_kwargs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    eval_env_kwargs = {
        **data_for_env['test'],
        'env_config': config['environment'],
        'reward_config': config['reward']
    }
    eval_env = make_vec_env(ManualCooldownEnv, n_envs=1, env_kwargs=eval_env_kwargs)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.)
    eval_env.obs_rms = train_env.obs_rms
    
    # Determine model types for logging
    approximator_config = config.get('approximator_models', {})
    dpdk_type = approximator_config.get('dpdk', {}).get('type', 'keras')
    oai_type = approximator_config.get('oai', {}).get('type', 'keras')
    
    obs_schema = config.get('environment', {}).get('observation_schema', 'hybrid')
    print(f"✓ Vectorized environments created for '{obs_schema}' observation schema in '{config.get('simulation_mode', {}).get('type', 'precompute')}' mode.")
    print(f"✓ Approximator models: DPDK ({dpdk_type}), OAI ({oai_type})")
    return train_env, eval_env

def compute_traffic_dynamics(traffic: np.ndarray,
                             window_size: int,
                             ema_short: int = 4,
                             ema_long: int = 16) -> dict:
    """
    Compute per-step scalars aligned with traffic: ΔT, moving mean/std over W,
    and short/long EMAs. All outputs are float32, length == len(traffic).
    """
    t = traffic.astype(np.float32).copy()
    n = t.shape[0]
    W = int(window_size)

    # ΔT
    dT = np.zeros_like(t, dtype=np.float32)
    dT[1:] = t[1:] - t[:-1]

    # moving mean/std over W (zero-padded at start)
    # Use cumulative sums for O(n)
    csum = np.cumsum(np.concatenate([np.zeros(1, dtype=np.float32), t]))
    # mean over last W: for index i, sum = csum[i+1] - csum[i+1-W] (clip)
    means = np.zeros_like(t)
    for i in range(n):
        start = max(0, i - W + 1)
        total = csum[i + 1] - csum[start]
        denom = (i - start + 1)
        means[i] = total / max(denom, 1)

    # std over W (one-pass via Welford or two-pass with means)
    # Simple two-pass with padding-aware subset:
    stds = np.zeros_like(t)
    for i in range(n):
        start = max(0, i - W + 1)
        segment = t[start:i + 1]
        stds[i] = segment.std(dtype=np.float64).astype(np.float32)

    # EMAs
    def ema(arr, N):
        alpha = 2.0 / (N + 1.0)
        out = np.empty_like(arr, dtype=np.float32)
        out[0] = arr[0]
        for k in range(1, arr.shape[0]):
            out[k] = alpha * arr[k] + (1.0 - alpha) * out[k - 1]
        return out

    ema_s = ema(t, int(ema_short))
    ema_l = ema(t, int(ema_long))

    return {
        'dT': dT.astype(np.float32),
        'mov_mean_W': means.astype(np.float32),
        'mov_std_W': stds.astype(np.float32),
        'ema_short': ema_s.astype(np.float32),
        'ema_long': ema_l.astype(np.float32),
        'feature_names': ['dT', 'mov_mean_W', 'mov_std_W', 'ema_short', 'ema_long']
    }