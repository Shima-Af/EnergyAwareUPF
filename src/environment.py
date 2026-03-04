# src/environment.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ManualCooldownEnv(gym.Env):
    def __init__(self, traffic_data, env_config, reward_config,
                 dpdk_lookup=None, oai_lookup=None,  # For precompute mode
                 dpdk_bundle=None, oai_bundle=None, # For runtime mode
                 extra_features=None,
                 forecast_data=None,
                 capacity_features=None,
                 calendar_features=None, calendar_features_forecast=None): 
        super().__init__()

        # Precompute mode if lookups are provided
        
        self.is_precompute_mode = (dpdk_lookup is not None)

        # --- Store models or lookups ---
        self.dpdk_lookup = dpdk_lookup
        self.oai_lookup = oai_lookup
        self.dpdk_bundle = dpdk_bundle
        self.oai_bundle = oai_bundle

        if not self.is_precompute_mode and (dpdk_bundle is None or oai_bundle is None):
            raise ValueError("Model bundles must be provided for 'runtime' simulation mode.")

        
        # --- Data & Config Unpacking ---
        self.traffic_data = traffic_data.astype(np.float32)

        # Observation schema: instant, history, forecast, hybrid
        schema = env_config.get("observation_schema")
        if schema is None:
            feature_mode = env_config.get("feature_mode", "both")
            schema = {
                "historical_only": "history",
                "forecast_only": "forecast",
                "both": "hybrid",
            }.get(feature_mode, "hybrid")
        self.observation_schema = schema
        if self.observation_schema not in {"instant", "history", "forecast", "hybrid"}:
            raise ValueError("observation_schema must be one of: instant, history, forecast, hybrid")

        self.forecast_horizon = int(env_config.get("forecast_horizon", 1))

        # --- Forecast wiring ---
        self.forecast_data = None
        if self.observation_schema in {"forecast", "hybrid"}:
            if forecast_data is None:
                raise ValueError("forecast_data is required for forecast or hybrid schemas.")
            self.forecast_data = np.asarray(forecast_data, dtype=np.float32)
            if len(self.forecast_data) != len(self.traffic_data):
                raise ValueError("forecast_data length must match traffic_data.")

        self.window_size = env_config['window_size']
        self.performance_threshold = env_config['performance_threshold']
        self.cooldown_period = env_config['cooldown_period']


        self.dpdk_idle = float(env_config.get('dpdk_idle_watts', 0.0))
        self.oai_idle_per_inst = float(env_config.get('oai_idle_watts_per_instance', 0.0))

        self.sec_eps = float(reward_config.get('sec_eps_mbps', 1e-6))
        self.qos_lambda = float(reward_config.get('qos_lambda', 0.0))

        self.type_switch_cost = float(reward_config.get('type_switch_cost', 0.0))
        self.scale_up_cost = float(reward_config.get('scale_up_cost_per_inst', 0.0))
        self.scale_down_cost = float(reward_config.get('scale_down_cost_per_inst', 0.0))

        self.use_dyn_features      = bool(env_config.get('use_dyn_features', True))
        if self.observation_schema == "forecast":
            self.use_dyn_features = False
        self.use_capacity_features = bool(env_config.get('use_capacity_features', False))
        self.use_powergap_features = bool(env_config.get('use_powergap_features', False))
        self.use_calendar_features = bool(env_config.get('use_calendar_features', True))
        self.perf_system_level     = bool(env_config.get('perf_system_level', False))
        

        self.usr_capacity_mbps = env_config.get("usr_capacity_mbps", None)
        self.usr_capacity_safety_margin = float(env_config.get("usr_capacity_safety_margin", 0.0))  # e.g., 0–5 Mbps


        # feature banks
        self.extra_features     = extra_features or {}      # dynamics (your ΔT, EMA, etc.)
        self.capacity_features  = capacity_features or {}   # dict from compute_option_features
        self.calendar_features  = calendar_features or {}   # sin/cos cycles
        self.calendar_features_forecast = calendar_features_forecast or {}
        
        # Optional forecast-slot feature banks (used only in proactive but I am not using them at all in this stage)
        # self.capacity_features_forecast = capacity_features_forecast or {}
        # self.calendar_features_forecast = calendar_features_forecast or {}

        # derive names and counts (safe existence checks)
        self.dyn_names  = self.extra_features.get('feature_names', [])
        max_k = int(env_config.get('num_oai_instances', 1))
        # self.num_dyn_features = len([n for n in self.dyn_feature_names if n in ('dT','mov_mean_W','mov_std_W','ema_short','ema_long')])

        # capacity (margins) feature names: dpdk_margin + oai{k}_margin
        cap_names = []
        if 'dpdk_margin' in self.capacity_features:
            cap_names.append('dpdk_margin')
        for k in range(1, max_k+1):
            name = f'oai{k}_margin'
            if 'oai_margin_k' in self.capacity_features and len(self.capacity_features['oai_margin_k']) >= k:
                cap_names.append(name)
        self.cap_names = cap_names

        # power gap feature names
        gap_names = []
        for key in sorted(self.capacity_features.get('power_gaps', {}).keys()):
            gap_names.append(key)
        self.gap_names = gap_names

        # calendar names
        cal_names = [n for n in ['sin_hour','cos_hour','sin_dow','cos_dow']
                     if n in self.calendar_features]
        self.cal_names = cal_names

        self.sec_scale = float(reward_config.get('sec_scale', 1.0))

        # No warmup in this build; keep neutral defaults
        self.activation_delay_steps = 0
        self.pending_activations = []
        self.warm_pool = 0
        
        # --- EXTENSIBLE ACTION SPACE ---
        self.num_oai_instances = env_config.get('num_oai_instances', 1)
        self.action_space = spaces.Discrete(1 + self.num_oai_instances)

        # --- Observation space length ---
        # Schema affects what we include:
        # - instant: current traffic (t)
        # - history: window of past traffic (t-W+1..t)
        # - forecast: forecast at t+H
        # - hybrid: history + forecast
        if self.observation_schema == "instant":
            base_len = 1 + 2  # traffic_t + config + cooldown
        elif self.observation_schema == "history":
            base_len = self.window_size + 2  # W history + config + cooldown
        elif self.observation_schema == "forecast":
            base_len = 1 + 2  # forecast + config + cooldown
        else:  # hybrid
            base_len = self.window_size + 1 + 2  # W history + 1 forecast + config + cooldown
        
        add_len = 0
        if self.use_dyn_features:      add_len += len(self.dyn_names)
        if self.use_capacity_features: add_len += len(self.cap_names)
        if self.use_powergap_features: add_len += len(self.gap_names)
        if self.use_calendar_features: add_len += len(self.cal_names)
        obs_len = base_len + add_len

        low  = np.zeros((obs_len,), dtype=np.float32)
        high = np.full((obs_len,), np.finfo(np.float32).max, dtype=np.float32)
        
        # Config code and cooldown positions differ between schemas
        if self.observation_schema in {"instant", "forecast"}:
            config_pos = 1
            cooldown_pos = 2
        elif self.observation_schema == "history":
            config_pos = self.window_size
            cooldown_pos = self.window_size + 1
        else:  # hybrid
            config_pos = self.window_size + 1
            cooldown_pos = self.window_size + 2
        
        high[config_pos] = self.num_oai_instances  # config code
        high[cooldown_pos] = self.cooldown_period   # cooldown
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.base_len = base_len
        self.config_pos = config_pos
        self.cooldown_pos = cooldown_pos

        # optional: track time-since-last-violation (hysteresis memory)
        self.use_tslv = True
        self.time_since_last_violation = 0

        # obs_shape = (self.window_size + 2,)
        # low = np.zeros(obs_shape, dtype=np.float32)
        # high = np.full(obs_shape, np.finfo(np.float32).max, dtype=np.float32)
        # high[-2] = self.num_oai_instances # Max action index
        # high[-1] = self.cooldown_period
        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Internal State Tracking ---
        self.current_step = 0
        self.current_upf_type = 0
        self.num_active_oai = 0
        self.steps_since_last_switch = self.cooldown_period

    def _get_prediction(self, traffic_val, upf_type_code, num_active_oai=0):

        if self.is_precompute_mode:
            if upf_type_code == 0: # DPDK (Simple dictionary lookup)
                return self.dpdk_lookup.get(traffic_val, (0, 0))
            else: # OAI (Multi-instance lookup using binary search)
                # if num_active_oai == 0:
                #     return 0, 0
                
                # num_active_oai is what the agent requested, but some may be pending activation
                effective_oai = num_active_oai

                # If activation delays exist, reduce the effective capacity by pending instances
                if self.activation_delay_steps > 0 and len(self.pending_activations) > 0:
                    activated_now = []
                    remaining_pending = []
                    for steps_left, cnt in self.pending_activations:
                        if steps_left <= 0:
                            activated_now.append(cnt)          # will be counted in num_active_oai already next step
                        else:
                            remaining_pending.append((steps_left, cnt))
                    self.pending_activations = remaining_pending
                    # effective_oai is the requested minus those still pending
                    still_pending = sum(cnt for _, cnt in self.pending_activations)
                    effective_oai = max(num_active_oai - still_pending, 0)

                if effective_oai == 0:
                    return 0, 0

                traffic_per_instance = traffic_val / effective_oai
                
                # Access the pre-computed sorted arrays
                keys = self.oai_lookup['keys']
                perfs = self.oai_lookup['perf']
                powers = self.oai_lookup['power']
                
                # Find the index of the closest pre-computed key using binary search
                # np.searchsorted finds where the element would be inserted to maintain order
                idx = np.searchsorted(keys, traffic_per_instance, side='left')

                # Handle edge cases: requested value is smaller/larger than all keys
                if idx == 0:
                    closest_idx = 0
                elif idx == len(keys):
                    closest_idx = len(keys) - 1
                else:
                    # Compare with the neighbor to find the truly closest key
                    left_neighbor_dist = traffic_per_instance - keys[idx - 1]
                    right_neighbor_dist = keys[idx] - traffic_per_instance
                    if left_neighbor_dist < right_neighbor_dist:
                        closest_idx = idx - 1
                    else:
                        closest_idx = idx
                        
                perf_per_instance = perfs[closest_idx]
                power_per_instance = powers[closest_idx]
                
                # Final performance is per-instance, total power is aggregated
                total_power = power_per_instance * effective_oai
                return perf_per_instance, total_power

        # Runtime Calculation Mode (Unchanged)
        if upf_type_code == 0: # DPDK
            traffic_per_instance = traffic_val
            bundle = self.dpdk_bundle
            perf, power = bundle.predict_batch(np.array([traffic_per_instance]))
            total_power = power[0]
            final_perf = perf[0]
        else: # OAI
            effective_oai = num_active_oai
            if effective_oai == 0:
                return 0, 0
            traffic_per_instance = traffic_val / effective_oai
            bundle = self.oai_bundle
            perf, power_per_instance = bundle.predict_batch(np.array([traffic_per_instance]))
            total_power = power_per_instance[0] * effective_oai
            final_perf = perf[0]
            
        return final_perf, total_power

    # --- State Getters ---
    def _get_state(self):
        return self._get_observation()

    def _forecast_index(self) -> int:
        return min(self.current_step + self.forecast_horizon, len(self.traffic_data) - 1)

    def _get_observation(self):
        i = self.current_step
        j = self._forecast_index()
        base_len = self.base_len

        add_len = 0
        if self.use_dyn_features:      add_len += len(self.dyn_names)
        if self.use_capacity_features: add_len += len(self.cap_names)
        if self.use_powergap_features: add_len += len(self.gap_names)
        if self.use_calendar_features: add_len += len(self.cal_names)
        obs_len = base_len + add_len

        state = np.zeros(obs_len, dtype=np.float32)

        if self.observation_schema == "instant":
            state[0] = float(self.traffic_data[i])
        elif self.observation_schema == "history":
            start_index = max(0, i - self.window_size + 1)
            hist = self.traffic_data[start_index:i + 1]
            state[self.window_size - len(hist):self.window_size] = hist
        elif self.observation_schema == "forecast":
            state[0] = float(self.forecast_data[j])
        else:  # hybrid
            start_index = max(0, i - self.window_size + 1)
            hist = self.traffic_data[start_index:i + 1]
            state[self.window_size - len(hist):self.window_size] = hist
            state[self.window_size] = float(self.forecast_data[j])

        state[self.config_pos] = self.get_current_config_code()
        state[self.cooldown_pos] = self.steps_since_last_switch

        pos = base_len
        # dynamics (scalars aligned at current_step)
        if self.use_dyn_features:
            for name in self.dyn_names:
                arr = self.extra_features.get(name, None)
                if arr is not None and i < len(arr):
                    state[pos] = float(arr[i]); pos += 1

        # capacity margins (dpdk, oai k=1..K)
        if self.use_capacity_features:
            if 'dpdk_margin' in self.cap_names:
                state[pos] = float(self.capacity_features['dpdk_margin'][i]); pos += 1
            if 'oai_margin_k' in self.capacity_features:
                for k in range(1, self.num_oai_instances + 1):
                    if f'oai{k}_margin' in self.cap_names:
                        arr = self.capacity_features['oai_margin_k'][k - 1]
                        state[pos] = float(arr[i]); pos += 1

        # power gaps
        if self.use_powergap_features:
            for key in self.gap_names:
                state[pos] = float(self.capacity_features['power_gaps'][key][i]); pos += 1

        # calendar aligned to traffic signal (t for instant/history, t+H for forecast/hybrid)
        if self.use_calendar_features:
            cal_idx = j if self.observation_schema in {"forecast", "hybrid"} else i
            src = self.calendar_features_forecast or self.calendar_features
            for key in self.cal_names:
                arr = src.get(key, None)
                if arr is not None and cal_idx < len(arr):
                    state[pos] = float(arr[cal_idx]); pos += 1

        return state
    
    def get_current_config_code(self):
        """Helper to get the single integer representing the current state."""
        return 0 if self.current_upf_type == 0 else self.num_active_oai

    # --- Gym Methods ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ALWAYS start at W-1 for fair cross-schema comparison
        # All schemas evaluate on the same episode length (data[W-1:end])
        self.current_step = self.window_size - 1
        self.current_upf_type = 0 # Start with DPDK
        self.num_active_oai = 0
        self.steps_since_last_switch = self.cooldown_period
        return self._get_state(), {}

    def step(self, action):
        self.steps_since_last_switch += 1
        
        requested = int(action)

        # # --- USR capacity safety guard (optional) ---
        safety_applied = False
        # if self.usr_capacity_mbps is not None and requested != 0:
        #     current_load = float(self.traffic_data[self.current_step])  # load at slot t
        #     if current_load > (self.usr_capacity_mbps - self.usr_capacity_safety_margin):
        #         requested = 0  # force DPDK
        #         safety_applied = True
        
        effective = requested

        # Map action -> (upf_type, oai_count)
        if effective == 0:
            chosen_upf_type, chosen_oai_count = 0, 0
        else:
            chosen_upf_type, chosen_oai_count = 1, effective  # OAI with k instances

    
        # ----- compute penalties against the PREVIOUS state -----

        old_type, old_k = self.current_upf_type, self.num_active_oai

        # Cooldown enforcement at execution time
        if self.steps_since_last_switch < self.cooldown_period:
            chosen_upf_type, chosen_oai_count = old_type, old_k

        # Type/scale penalties are charged when the executed config actually changes at t
        type_switch_penalty = 0.0
        scale_penalty = 0.0

        if chosen_upf_type != old_type:
            # if chosen_upf_type == 1:
            #     type_switch_penalty = 0.5*self.type_switch_cost
            # elif chosen_upf_type == 0:
            type_switch_penalty = self.type_switch_cost
        else:
            delta_k = chosen_oai_count - old_k
            if   delta_k > 0:  scale_penalty = self.scale_up_cost   * delta_k
            elif delta_k < 0:  scale_penalty = self.scale_down_cost * (-delta_k)

        # If anything changed, reset cooldown
        if (chosen_upf_type != old_type) or (chosen_oai_count != old_k):
            self.steps_since_last_switch = 0

        # ---- now commit the new state ----
        # Commit executed state for slot t
        self.current_upf_type = chosen_upf_type
        self.num_active_oai   = chosen_oai_count
        
        current_traffic = float(self.traffic_data[self.current_step])
        performance, power = self._get_prediction(current_traffic, self.current_upf_type, self.num_active_oai)

        # --- Idle totals for SEC ---
        if self.current_upf_type == 0:  # DPDK
            idle_total = self.dpdk_idle
        else:
            idle_total = self.oai_idle_per_inst * self.num_active_oai

        # --- SEC: incremental watts per Mbps --- 
        traffic_mbps = max(current_traffic, 0.0)
        incremental_power = max(power - idle_total, 0.0) # it's not incremental in this context so it's reported but not used 
        sec = power / max(traffic_mbps, self.sec_eps)  # J/bit up to constants

        # QoS penalty
        if performance >= self.performance_threshold:
            qos_penalty = 0.0
        else:
            qos_penalty = self.qos_lambda * (self.performance_threshold - performance)

        # Switching / scaling penalties
        # Determine requested config for this step (chosen_upf_type, chosen_oai_count)

        reward = - self.sec_scale * sec - type_switch_penalty - scale_penalty - qos_penalty

         
        self.current_step += 1
        terminated = self.current_step >= len(self.traffic_data) - 1
        truncated = False

        # chosen_upf_str = 'DPDK' if self.current_upf_type == 0 else f'{self.num_active_oai}xOAI'
            
        # info = {
        #     'traffic': traffic_mbps,
        #     'chosen_upf': 'DPDK' if self.current_upf_type == 0 else f'{self.num_active_oai}xOAI',
        #     'power': float(power),
        #     'performance': float(performance),
        #     'reward': float(reward),
        #     'sec': float(sec) if self.reward_type != 'legacy' else None,
        #     'incremental_power': float(incremental_power),
        #     'idle_total': float(idle_total),
        #     'type_switch_penalty': float(type_switch_penalty),
        #     'scale_penalty': float(scale_penalty),
        #     'qos_penalty': float(qos_penalty) if self.reward_type != 'legacy' else None,
        #     'warm_pool': 0,  # int(self.warm_pool),
        #     'pending_activations': 0,  # int(sum(cnt for _, cnt in self.pending_activations)),
        #     'cooldown': self.steps_since_last_switch
        # }
                # ----- build info dict with rich logging -----

        # previous config code (before applying cooldown)
        prev_cfg_code = 0 if old_type == 0 else old_k

        # requested vs executed action (after cooldown gating)
        requested_code = 0 if int(action) == 0 else int(action)  # 0=DPDK, k=OAI
        executed_code  = 0 if self.current_upf_type == 0 else self.num_active_oai
        did_switch = int(executed_code != prev_cfg_code)

        # expose traffic window (last W samples, left-padded with 0s)
        start_index = max(0, self.current_step - self.window_size)
        tw = self.traffic_data[start_index:self.current_step].astype(float)
        if len(tw) < self.window_size:
            pad = np.zeros(self.window_size - len(tw), dtype=np.float32)
            traffic_window = np.concatenate([pad, tw])
        else:
            traffic_window = tw[-self.window_size:]

        # optional feature snapshots aligned at current_step
        dyn_snapshot = {}
        if self.use_dyn_features:
            for name in self.dyn_names:
                arr = self.extra_features.get(name, None)
                if arr is not None and self.current_step < len(arr):
                    dyn_snapshot[name] = float(arr[self.current_step])

        cap_snapshot = {}
        if self.use_capacity_features:
            if 'dpdk_margin' in self.cap_names:
                cap_snapshot['dpdk_margin'] = float(self.capacity_features['dpdk_margin'][self.current_step])
            if 'oai_margin_k' in self.capacity_features:
                for k in range(1, self.num_oai_instances + 1):
                    key = f'oai{k}_margin'
                    if key in self.cap_names:
                        cap_snapshot[key] = float(self.capacity_features['oai_margin_k'][k - 1][self.current_step])

        gap_snapshot = {}
        if self.use_powergap_features and 'power_gaps' in self.capacity_features:
            for key in self.gap_names:
                gap_snapshot[key] = float(self.capacity_features['power_gaps'][key][self.current_step])

        cal_snapshot = {}
        if self.use_calendar_features:
            for key in self.cal_names:
                arr = self.calendar_features.get(key, None)
                if arr is not None and self.current_step < len(arr):
                    cal_snapshot[key] = float(arr[self.current_step])

        info = {
            # core signals
            'traffic': float(traffic_mbps),
            'chosen_upf': 'DPDK' if self.current_upf_type == 0 else f'{self.num_active_oai}xOAI',
            'power': float(power),
            'performance': float(performance),
            'reward': float(reward),

            # energy metrics
            'sec': float(sec),
            'incremental_power': float(incremental_power),
            'idle_total': float(idle_total),

            # penalties
            'type_switch_penalty': float(type_switch_penalty),
            'scale_penalty': float(scale_penalty),
            'qos_penalty': float(qos_penalty),

            # gating + state memory
            'cooldown': int(self.steps_since_last_switch),
            'cooldown_period': int(self.cooldown_period),
            'prev_cfg_code': int(prev_cfg_code),
            'requested_code': int(requested_code),
            'executed_code': int(executed_code),
            'switch': int(did_switch),

            # observation structure (per-step snapshot)
            'obs_len': int(self.observation_space.shape[0]),
            'window_size': int(self.window_size),

            # snapshots for analysis
            'traffic_window': traffic_window.tolist(),

            # optional banks
            **({f'dyn_{k}': v for k, v in dyn_snapshot.items()} if dyn_snapshot else {}),
            **({f'cap_{k}': v for k, v in cap_snapshot.items()} if cap_snapshot else {}),
            **({f'gap_{k}': v for k, v in gap_snapshot.items()} if gap_snapshot else {}),
            **({f'cal_{k}': v for k, v in cal_snapshot.items()} if cal_snapshot else {}),

            # activation/warmup state
            "requested_action": int(action),      # policy's raw request at t
            "executed_action": int(executed_code),# actually ran at t, after cooldown gating
            "cooldown_blocked": bool(requested_code != executed_code),
            "executed_upf": "DPDK" if self.current_upf_type == 0 else f"{self.num_active_oai}xOAI",
            "usr_capacity_guard": bool(safety_applied),
        }


        return self._get_state(), float(reward), terminated, truncated, info