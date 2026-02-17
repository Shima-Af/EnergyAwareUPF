# src/evaluate.py
# Import TensorFlow configuration first to suppress warnings
from . import tf_config  # noqa: F401  (side-effect import)

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Import our custom modules
from . import utils
from . import agent
from .environment import ManualCooldownEnv


# ==============================
# Helpers
# ==============================

def _nearest_from_sorted(keys: np.ndarray, x: float) -> int:
    """Return index of nearest value to x in a sorted 1D array."""
    idx = np.searchsorted(keys, x, side="left")
    if idx == 0:
        return 0
    if idx >= len(keys):
        return len(keys) - 1
    return idx if (keys[idx] - x) < (x - keys[idx - 1]) else idx - 1


def predict_dpdk_from_lookup(dpdk_lookup: dict, T: float):
    """
    dpdk_lookup is a dict {throughput: (perf, power)} with float keys.
    Use nearest-neighbor to avoid exact-float key issues.
    """
    ks = np.array(sorted(dpdk_lookup.keys()), dtype=np.float32)
    vals = np.array([dpdk_lookup[float(k)] for k in ks], dtype=np.float32)
    j = _nearest_from_sorted(ks, T)
    perf, power = vals[j]
    return float(perf), float(power)


def predict_oai_from_lookup(oai_lookup: dict, T: float, k: int):
    """
    oai_lookup has 'keys' (sorted per-instance throughput), 'perf' (per-instance), 'power' (per-instance).
    Returns (perf_per_instance, total_power = k * power_per_instance) to mirror the env’s logic.
    """
    if k <= 0:
        return 0.0, 0.0
    per_inst = T / k
    keys = oai_lookup["keys"]
    perfs = oai_lookup["perf"]
    powers = oai_lookup["power"]
    j = _nearest_from_sorted(keys, per_inst)
    perf_inst = float(perfs[j])
    power_tot = float(powers[j]) * k
    return perf_inst, power_tot


def to_int_series(s: pd.Series) -> pd.Series:
    """Coerce a series to pandas nullable integer (Int64), tolerating NaNs."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# ---- Y-axis (OAI-first) mapping ----
def make_label_order(K: int):
    """Returns (labels_in_order, code->ordinal function, label->ordinal mapper)."""
    # Desired y-axis order: 1xOAI, 2xOAI, ..., KxOAI, DPDK (last)
    labels = [f"{k}xOAI" for k in range(1, K + 1)] + ["DPDK"]

    def code_to_ord(code: int) -> int:
        # Env codes: 0 = DPDK, 1..K = kxOAI
        c = int(code)
        if c == 0:
            return K  # DPDK goes last
        # 1..K map to 0..K-1
        return max(0, min(K - 1, c - 1))

    label2ord = {lab: i for i, lab in enumerate(labels)}
    return labels, code_to_ord, label2ord


def chosen_upf_label_to_ord(s: pd.Series, label2ord: dict) -> pd.Series:
    """Map 'DPDK' or 'kxOAI' strings to OAI-first ordinal indices, robust to bad values."""
    return s.astype(str).map(label2ord).fillna(len(label2ord) - 1).astype(int)  # unknown → DPDK row


# ==============================
# Analysis + Plots
# ==============================

def analyze_and_plot_results(df: pd.DataFrame, config: dict, stamp: str | None = None):
    """Print a summary of results and generate performance plots."""
    env_cfg = config["environment"]
    thr = env_cfg["performance_threshold"]
    K = int(env_cfg.get("num_oai_instances", 1))
    # Ensure core numeric columns are numeric
    for col in ("traffic", "power", "performance", "sec"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    labels_order, code_to_ord, label2ord = make_label_order(K)

    print("\n" + "=" * 50)
    print("      EVALUATION RESULTS ON TEST DATA")
    print("=" * 50)
    print(f"Total Steps in Test Set: {len(df)}")
    if "power" in df:
        print(f"Average Power Consumption: {df['power'].mean():.3f} W")
    if "performance" in df:
        print(f"Average Performance Score: {df['performance'].mean():.3f}")
    if "sec" in df and df["sec"].notna().any():
        print(f"Average SEC (W/Mbps): {df['sec'].mean():.6f}")

    # Switching & QoS counts (robust to missing columns)
    total_type_switches = int((df.get("type_switch_penalty", 0) > 0).sum())
    total_scale_events = int((df.get("scale_penalty", 0) != 0).sum())
    qos_violations = (
        int((df["performance"] < thr).sum())
        if "performance" in df
        else int((df.get("qos_penalty", 0) > 0).sum())
    )
    guard_overrides = int(df.get("usr_capacity_guard", False).astype(bool).sum())

    print(f"Type Switches:            {total_type_switches}")
    print(f"Scaling Events:           {total_scale_events}")
    print(f"Performance Violations:   {qos_violations} (Threshold: {thr})")
    if "usr_capacity_guard" in df.columns:
        print(f"USR Capacity Guard Hits:  {guard_overrides}")

    print("=" * 50)

    # ----- Build numeric series (ORDINAL indices with OAI-first axis) -----
    # Executed (actual, after cooldown/safety)
    if "executed_action" in df.columns:
        exec_ord = to_int_series(df["executed_action"]).fillna(0).apply(code_to_ord).astype(int)
    elif "executed_code" in df.columns:
        exec_ord = to_int_series(df["executed_code"]).fillna(0).apply(code_to_ord).astype(int)
    elif "chosen_upf" in df.columns:
        exec_ord = chosen_upf_label_to_ord(df["chosen_upf"], label2ord)
    else:
        exec_ord = pd.Series(np.zeros(len(df), dtype=int))

    # Requested (same-step request)
    plot_req_ord = None
    if "requested_action" in df.columns:
        req_code = to_int_series(df["requested_action"]).fillna(0)
        plot_req_ord = req_code.apply(code_to_ord).astype(int)

    # ----- Plotting -----
    results_dir = config["paths"]["results_dir"]
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ========== PLOT 1: Original Two-Panel Plot (Traffic/Power + Actions) ==========
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    t = np.arange(len(df))

    # Traffic & Power
    if "traffic" in df:
        ax1.plot(t, df["traffic"], label="Traffic Load (Mbps)")
        ax1.set_ylabel("Traffic (Mbps)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1b = ax1.twinx()
    if "power" in df:
        ax1b.plot(t, df["power"], label="Power (W)", linestyle="--")
        ax1b.set_ylabel("Power (W)")
    # combine legends for twin axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    if h1 or h2:
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    # Actions on OAI-first axis
    ax2.step(t, exec_ord, where="post", label="Executed (post-cooldown)", linewidth=2)
    if plot_req_ord is not None:
        ax2.step(t, plot_req_ord, where="post", label="Requested", linestyle="--", alpha=0.75)

    # Capacity guard markers
    if "usr_capacity_guard" in df.columns:
        idx_guard = np.where(df["usr_capacity_guard"].astype(bool).to_numpy())[0]
        if idx_guard.size > 0:
            ax2.scatter(idx_guard, np.asarray(exec_ord)[idx_guard],
                        marker="x", s=46, label="USR guard override")

    # Cooldown blocked markers
    blocked_idx = None
    if "cooldown_blocked" in df.columns:
        blocked_idx = np.where(df["cooldown_blocked"].fillna(False).to_numpy())[0]
    elif {"scheduled_action", "executed_action"}.issubset(df.columns):
        sched = to_int_series(df["scheduled_action"])
        exe = to_int_series(df["executed_action"])
        blocked_idx = np.where(sched.ne(exe).fillna(False).to_numpy())[0]
    if blocked_idx is not None and len(blocked_idx) > 0:
        ax2.scatter(blocked_idx, np.asarray(exec_ord)[blocked_idx],
                    marker="o", s=30, facecolors="none", edgecolors="tab:red",
                    label="Blocked by cooldown")

    # OAI-first ticks & labels
    ax2.set_yticks(list(range(0, K + 1)))
    ax2.set_yticklabels(labels_order)
    ax2.set_xlabel("Time Step (15 min intervals)")
    ax2.set_ylabel("UPF Selection (OAI first)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    fig1.suptitle("RL Agent — Traffic, Power, and Actions", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fname_suffix = f"_{stamp}" if stamp else ""
    fig1_path = os.path.join(fig_dir, f"action_timeline{fname_suffix}.png")
    plt.savefig(fig1_path, dpi=150)
    print(f"✓ Action timeline (original) saved to {fig1_path}")
    plt.show()

    # ========== PLOT 2: New Traffic-Focused Plot with UPF Background Zones ==========
    fig2, ax = plt.subplots(figsize=(16, 6))
    t = np.arange(len(df))

    # Define colors for each specific UPF configuration
    upf_colors = {
        'DPDK': '#E8F4F8',      # Light blue for DPDK
        '1xOAI': '#FFF4E6',     # Light orange for 1xOAI
        '2xOAI': '#F0E6FF',     # Light purple for 2xOAI
    }

    # ----- Background zones for UPF configurations -----
    # Map executed ordinal codes back to configuration labels
    upf_configs = []
    for code in exec_ord:
        if code == K:  # DPDK is mapped to K in OAI-first ordering
            upf_configs.append('DPDK')
        else:
            # code 0 = 1xOAI, code 1 = 2xOAI, etc.
            upf_configs.append(f'{code+1}xOAI')
    
    current_config = upf_configs[0]
    start_idx = 0

    for i in range(1, len(upf_configs) + 1):
        # Check if we've reached end or if UPF config changed
        if i == len(upf_configs) or upf_configs[i] != current_config:
            # Fill the region with appropriate color
            color = upf_colors.get(current_config, '#F5F5F5')
            ax.axvspan(start_idx, i-1, alpha=0.3, color=color, zorder=1)
            
            if i < len(upf_configs):
                current_config = upf_configs[i]
                start_idx = i

    # ----- Traffic line -----
    if "traffic" in df:
        ax.plot(t, df["traffic"], label="Traffic Load", color="gray", linewidth=2, zorder=3)
        ax.set_ylabel("Traffic (Mbps)", fontsize=11)

    # ----- Switch event markers -----
    switch_indices = []
    for i in range(1, len(upf_configs)):
        if upf_configs[i] != upf_configs[i-1]:
            switch_indices.append(i)
            ax.axvline(x=i, color="green", linestyle="--", linewidth=1, alpha=0.5, zorder=2)

    # ----- Legend -----
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Traffic Load'),
        Patch(facecolor=upf_colors['DPDK'], alpha=0.3, label='DPDK Active'),
    ]
    
    # Add OAI configurations dynamically based on K
    for k in range(1, K + 1):
        key = f'{k}xOAI'
        if key in upf_colors:
            legend_elements.append(Patch(facecolor=upf_colors[key], alpha=0.3, label=f'{key} Active'))
    
    legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, 
                                     label=f'Switch Events (n={len(switch_indices)})'))

    # ----- Formatting -----
    ax.set_xlabel("Time Step (15-min intervals)", fontsize=11)
    ax.set_title("RL Agent: UPF Selection Timeline with Traffic Pattern", fontsize=12, fontweight='bold')
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim(left=0, right=len(t)-1)

    plt.tight_layout()

    fig2_path = os.path.join(fig_dir, f"rl_switching_timeline{fname_suffix}.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"✓ RL switching timeline saved to {fig2_path}")
    plt.show()


# ==============================
# Main eval
# ==============================

def evaluate():
    # --- CLI: which config and which RUN to evaluate ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional config override; by default uses run_dir/config.yaml")
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to a run folder under saved_models/best_recurrent_ppo (e.g. saved_models/best_recurrent_ppo/20250908-124959)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional explicit model path; overrides --run_dir/best_model.zip if given",
    )
    parser.add_argument(
        "--stamp",
        action="store_true",
        help="Append a timestamp to saved CSV/fig filenames to avoid overwriting when re-evaluating the same run.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # --- 1. Load Config and Data ---
    # IMPORTANT: Use the config from the run directory to ensure observation space matches
    run_config_path = os.path.join(run_dir, "config.yaml")
    if args.config is not None:
        print(f"\033[93m⚠ Using override config: {args.config}\033[0m")
        config = utils.load_config(args.config)
    elif os.path.exists(run_config_path):
        print(f"\033[92m✓ Using run-specific config: {run_config_path}\033[0m")
        config = utils.load_config(run_config_path)
    else:
        print(f"\033[91m✗ No config.yaml found in {run_dir}, falling back to default config.yaml\033[0m")
        config = utils.load_config("config.yaml")
    
    precomputed_data = utils.load_and_preprocess_data(config)

    per_run_eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(per_run_eval_dir, exist_ok=True)

    # Override results_dir for this evaluation only
    config["paths"]["results_dir"] = per_run_eval_dir
    paths_config = config["paths"]
    print(f"✓ Per-run eval output dir: {paths_config['results_dir']}")

    # --- 2. Create the Evaluation Environment and Load Stats ---
    print("\n--- 2. Creating Evaluation Environment ---")
    eval_env_kwargs = {
        **precomputed_data["test"],
        "env_config": config["environment"],
        "reward_config": config["reward"],
    }

    eval_env = make_vec_env(ManualCooldownEnv, n_envs=1, env_kwargs=eval_env_kwargs)

    # Prefer per-run VecNormalize stats
    stats_path = os.path.join(run_dir, "vec_normalize_stats.pkl")
    print(f"\033[95mUsing VecNormalize stats from run directory: {stats_path}\033[0m")
    if not os.path.exists(stats_path):
        print(f"ERROR: VecNormalize stats not found at {stats_path}")
        return

    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    print(f"✓ Evaluation environment created and stats loaded from {stats_path}")

    # --- Save a feature manifest (obs size + names) for traceability ---
    try:
        env0 = eval_env.envs[0].unwrapped  # VecEnv -> Gym env
        schema = getattr(env0, "observation_schema", "hybrid")
        horizon = int(getattr(env0, "forecast_horizon", 1))
        W = int(env0.window_size)
        obs_len = int(env0.observation_space.shape[0])
        # Traffic-related feature names
        if schema == "instant":
            traffic_names = ["traffic[t]"]
        elif schema == "history":
            traffic_names = [f"traffic[t-{i}]" for i in range(W - 1, -1, -1)]
        elif schema == "forecast":
            traffic_names = [f"forecast_traffic[t+{horizon}]"]
        else:  # hybrid
            traffic_names = (
                [f"traffic[t-{i}]" for i in range(W - 1, -1, -1)]
                + [f"forecast_traffic[t+{horizon}]"]
            )
        manifest = {
            "observation_len": obs_len,
            "window_size": W,
            "observation_schema": schema,
            "forecast_horizon": horizon,
            "banks_enabled": {
                "use_dyn_features": bool(getattr(env0, "use_dyn_features", False)),
                "use_capacity_features": bool(getattr(env0, "use_capacity_features", False)),
                "use_powergap_features": bool(getattr(env0, "use_powergap_features", False)),
                "use_calendar_features": bool(getattr(env0, "use_calendar_features", False)),
            },
            "feature_names": {
                "traffic_window": traffic_names,
                "config_code": f"index {getattr(env0, 'config_pos', 'unknown')}",
                "cooldown": f"index {getattr(env0, 'cooldown_pos', 'unknown')}",
                "dyn_names": list(getattr(env0, "dyn_names", [])),
                "cap_names": list(getattr(env0, "cap_names", [])),
                "gap_names": list(getattr(env0, "gap_names", [])),
                "cal_names": list(getattr(env0, "cal_names", [])),
            },
        }
        manifest_path = os.path.join(paths_config["results_dir"], "feature_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"✓ Feature manifest saved to {manifest_path}")
    except Exception as e:
        print(f"Feature manifest save skipped: {e}")

    # --- 3. Load the Trained Agent ---
    print("\n--- 3. Loading Best Trained Agent ---")
    model_path = args.model if args.model is not None else os.path.join(args.run_dir, "best_model.zip")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
    model = agent.load_agent(model_path, env=eval_env)


    is_recurrent = getattr(model.policy, "is_recurrent", False)


    # --- 4. Run Evaluation Loop ---
    print("\n--- 4. Running Evaluation on Test Data ---")
    results = []

    # Reset can return obs or (obs, info) depending on versions; normalize to obs only
    obs = eval_env.reset()
    if isinstance(obs, tuple):  # gymnasium-style reset
        obs = obs[0]

    # Determine if model uses forecast (for extracting forecast from obs)
    env0 = eval_env.envs[0].unwrapped
    schema = getattr(env0, "observation_schema", "hybrid")
    uses_forecast = schema in {"forecast", "hybrid"}
    window_size = int(getattr(env0, "window_size", 96))
    print(f"Observation schema: {schema}, Window size: {window_size}")

    state = None
    # SB3 expects a (num_envs,) bool array for recurrent policies
    episode_start = np.ones((eval_env.num_envs,), dtype=bool)

    while True:
        # Branch to avoid passing episode_start/state to non-recurrent PPO on older SB3
        if is_recurrent:
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=True
            )
        else:
            action, _ = model.predict(obs, deterministic=True)

        # VecEnv step returns (obs, rewards, dones, infos) in SB3; normalize to obs only
        step_out = eval_env.step(action)
        if len(step_out) == 4:
            obs, _, dones, infos = step_out
        else:
            # Defensive fallback in case of gymnasium 5-tuple, though VecEnv should be 4-tuple
            obs, _, terminated, truncated, infos = step_out
            dones = np.logical_or(terminated, truncated)

        # Ensure obs is not a (obs, info) tuple after step (rare, but just in case)
        if isinstance(obs, tuple):
            obs = obs[0]

        # Update episode starts for recurrent nets; shape must be (num_envs,)
        episode_start = np.asarray(dones, dtype=bool).reshape(-1)

        # Collect first env's info (you run n_envs=1)
        step_info = infos[0].copy()
        
        # Extract predicted traffic if forecast is used
        if uses_forecast:
            try:
                if hasattr(env0, "_forecast_index"):
                    f_idx = env0._forecast_index()
                else:
                    f_h = int(getattr(env0, "forecast_horizon", 1))
                    f_idx = min(env0.current_step + f_h, len(env0.forecast_data) - 1)
                raw_forecast = env0.forecast_data[f_idx] if hasattr(env0, 'forecast_data') and hasattr(env0, 'current_step') else np.nan
                step_info['predicted_traffic'] = float(raw_forecast)
            except (IndexError, AttributeError):
                step_info['predicted_traffic'] = np.nan
        else:
            step_info['predicted_traffic'] = np.nan
        
        results.append(step_info)

        # Single-env stop condition
        if episode_start[0]:
            break

    # --- 5. Process and Display Results ---
    if not results:
        print("No results collected.")
        return

    eval_df = pd.DataFrame(results)

    # Derive executed_upf_label from codes (0=DPDK, 1..K=OAI)
    K = int(config["environment"].get("num_oai_instances", 1))
    if "executed_action" in eval_df.columns:
        exec_codes = to_int_series(eval_df["executed_action"]).fillna(0).astype(int).clip(0, K)
    elif "executed_code" in eval_df.columns:
        exec_codes = to_int_series(eval_df["executed_code"]).fillna(0).astype(int).clip(0, K)
    else:
        exec_codes = pd.Series(np.zeros(len(eval_df), dtype=int))
    exec_labels = exec_codes.apply(lambda c: "DPDK" if c == 0 else f"{c}xOAI")
    eval_df["executed_upf_label"] = exec_labels

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") if args.stamp else None

    # Save results CSV (per-run dir, optional timestamp)
    results_fname = f"evaluation_results_{config['agent']['policy']}{('_' + stamp) if stamp else ''}.csv"
    results_path = os.path.join(paths_config["results_dir"], results_fname)
    eval_df.to_csv(results_path, index=False)
    print(f"\n✓ Evaluation results saved to {results_path}")

    # Save a compact timeline CSV for plotting elsewhere
    timeline_cols = {
        "t": np.arange(len(eval_df)),
        "traffic": pd.to_numeric(eval_df.get("traffic"), errors="coerce"),
        "power": pd.to_numeric(eval_df.get("power"), errors="coerce"),
        "requested_action": eval_df.get("requested_action"),
        "executed_action": eval_df.get("executed_action"),
        "usr_capacity_guard": eval_df.get("usr_capacity_guard", False),
        "cooldown_blocked": eval_df.get("cooldown_blocked", False),
    }
    timeline = pd.DataFrame(timeline_cols)
    timeline_fname = f"timeline_{config['agent']['policy']}{('_' + stamp) if stamp else ''}.csv"
    timeline_path = os.path.join(paths_config["results_dir"], timeline_fname)
    timeline.to_csv(timeline_path, index=False)
    print(f"✓ Timeline CSV saved to {timeline_path}")

    # Plots and summary
    analyze_and_plot_results(eval_df, config, stamp=stamp)

    # ---------------- Baselines on the SAME traffic ----------------
    env_cfg = config["environment"]
    thr = env_cfg["performance_threshold"]
    N = int(env_cfg.get("num_oai_instances", 1))

    dpdk_lookup = precomputed_data["test"]["dpdk_lookup"]
    oai_lookup = precomputed_data["test"]["oai_lookup"]

    Ts = pd.to_numeric(eval_df["traffic"], errors="coerce").to_numpy()

    # Always-DPDK
    dpdk_perf, dpdk_pow = [], []
    for T in Ts:
        p, w = predict_dpdk_from_lookup(dpdk_lookup, float(T))
        dpdk_perf.append(p)
        dpdk_pow.append(w)

    # Always-k×OAI for k=1..N
    oai_perf_k, oai_pow_k = {}, {}
    for k in range(1, N + 1):
        pfk, pwk = [], []
        for T in Ts:
            p, w = predict_oai_from_lookup(oai_lookup, float(T), k)
            pfk.append(p)
            pwk.append(w)
        oai_perf_k[k] = np.array(pfk, dtype=np.float32)
        oai_pow_k[k] = np.array(pwk, dtype=np.float32)

    dpdk_perf = np.array(dpdk_perf, dtype=np.float32)
    dpdk_pow = np.array(dpdk_pow, dtype=np.float32)

    # SEC = power / throughput (guard small T)
    eps = float(config["reward"].get("sec_eps_mbps", 1e-6))
    sec_policy = pd.to_numeric(eval_df["power"], errors="coerce").to_numpy() / np.maximum(Ts, eps)
    sec_dpdk = dpdk_pow / np.maximum(Ts, eps)
    sec_oai_k = {k: oai_pow_k[k] / np.maximum(Ts, eps) for k in oai_pow_k.keys()}

    # QoS violation masks
    perf_policy = pd.to_numeric(eval_df.get("performance"), errors="coerce").to_numpy() if "performance" in eval_df else np.zeros_like(Ts)
    viol_policy = (perf_policy < thr)
    viol_dpdk = (dpdk_perf < thr)
    viol_oai_k = {k: (oai_perf_k[k] < thr) for k in oai_perf_k.keys()}

    # Energy (Wh) over the horizon—assuming 15 min steps
    STEP_HOURS = 0.25
    E_policy = pd.to_numeric(eval_df["power"], errors="coerce").to_numpy().sum() * STEP_HOURS
    E_dpdk = dpdk_pow.sum() * STEP_HOURS
    E_oai_k = {k: oai_pow_k[k].sum() * STEP_HOURS for k in oai_pow_k.keys()}

    # Summaries
    summary_rows = [
        {
            "Policy": "Learned",
            "Mean Power (W)": float(np.nanmean(eval_df["power"])),
            "Mean SEC (W/Mbps)": float(np.nanmean(sec_policy)),
            "QoS Viol. Rate": float(np.nanmean(viol_policy)),
            "Total Energy (Wh)": float(E_policy),
        },
        {
            "Policy": "Static DPDK",
            "Mean Power (W)": float(dpdk_pow.mean()),
            "Mean SEC (W/Mbps)": float(sec_dpdk.mean()),
            "QoS Viol. Rate": float(viol_dpdk.mean()),
            "Total Energy (Wh)": float(E_dpdk),
        },
    ]
    for k in range(1, N + 1):
        summary_rows.append({
            "Policy": f"Static {k}xOAI",
            "Mean Power (W)": float(oai_pow_k[k].mean()),
            "Mean SEC (W/Mbps)": float(sec_oai_k[k].mean()),
            "QoS Viol. Rate": float(viol_oai_k[k].mean()),
            "Total Energy (Wh)": float(E_oai_k[k]),
        })

    summary_df = pd.DataFrame(summary_rows)

    # % energy saved by the learned policy vs each baseline
    summary_df["% Saved by Learned"] = np.nan
    for i in range(1, len(summary_df)):
        baseline_E = summary_df.loc[i, "Total Energy (Wh)"]
        summary_df.loc[i, "% Saved by Learned"] = 100.0 * (baseline_E - E_policy) / baseline_E

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("\n=== Policy vs Static Baselines (same traffic) ===")
    print(summary_df.to_string(index=False))
    summary_fname = f"evaluation_summary_{config['agent']['policy']}{('_' + stamp) if stamp else ''}.csv"
    summary_csv = os.path.join(paths_config["results_dir"], summary_fname)
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary saved to {summary_csv}")

    # ---- Optional: bar chart of total energy ----
    try:
        labels_bar = summary_df["Policy"].tolist()
        energies = summary_df["Total Energy (Wh)"].tolist()
        plt.figure(figsize=(8, 4))
        plt.bar(labels_bar, energies)
        plt.ylabel("Total Energy (Wh)")
        plt.title("Energy over Evaluation Horizon")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=20)
        plt.tight_layout()
        fig_dir = os.path.join(paths_config["results_dir"], "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"energy_comparison{('_' + stamp) if stamp else ''}.png")
        plt.savefig(fig_path, dpi=150)
        plt.show()
        print(f"✓ Energy bar chart saved to {fig_path}")
    except Exception as e:
        print("Plot skipped:", e)

    # ---- Per-traffic-bin SEC & QoS ----
    bins = [0, 50, 100, 150, 200, 250, 300, 400, 1e9]
    bin_labels = [f"{bins[i]}–{bins[i+1]} Mbps" for i in range(len(bins) - 1)]
    b = np.digitize(Ts, bins) - 1

    rows = []
    for i, lab in enumerate(bin_labels):
        mask = (b == i)
        if not mask.any():
            continue
        rows.append({
            "Bin": lab,
            "Learned SEC": float(np.nanmean(sec_policy[mask])),
            "DPDK SEC": float(np.nanmean(sec_dpdk[mask])),
            **{f"{k}xOAI SEC": float(np.nanmean(sec_oai_k[k][mask])) for k in range(1, N + 1)},
            "Learned QoS viol.": float(np.nanmean(perf_policy[mask] < thr)),
            "DPDK QoS viol.": float(np.nanmean(viol_dpdk[mask])),
            **{f"{k}xOAI QoS viol.": float(np.nanmean(viol_oai_k[k][mask])) for k in range(1, N + 1)},
        })

    bin_df = pd.DataFrame(rows)
    print("\n=== Per-traffic-bin SEC & QoS ===")
    if not bin_df.empty:
        print(bin_df.to_string(index=False))
        bin_fname = f"evaluation_bins_{config['agent']['policy']}{('_' + stamp) if stamp else ''}.csv"
        bin_csv = os.path.join(paths_config["results_dir"], bin_fname)
        bin_df.to_csv(bin_csv, index=False)
        print(f"✓ Bin summary saved to {bin_csv}")
    else:
        print("No bins populated (check traffic value ranges).")


if __name__ == "__main__":
    evaluate()
