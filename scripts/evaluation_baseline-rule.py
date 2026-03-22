# scripts/Baseline/evaluation_baseline_rule.py
import os, sys, math, time, numpy as np, pandas as pd,json

# Add project root to path for src imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src import utils
from baseline_rule import RuleBasedSelector, BaselineConfig, BaselineWeights, dpdk_predict, oai_predict
from tqdm import tqdm

# ================== 1. Setup ==================
print("\n--- Baseline Evaluation (Rule-based) ---")
print("⚠ IMPORTANT: This baseline uses the SAME preprocessed data as RL evaluation")
print("             to ensure apple-to-apple comparison.")

config = utils.load_config("config.yaml")
env_cfg = config["environment"]

# Load aligned preprocessed data (SAME as RL - includes trimming for window_size + lead_time)
pre = utils.load_and_preprocess_data(config)

# Extract test data - already aligned by load_and_preprocess_data
Ts_actual_full   = np.asarray(pre["test"]["traffic_data"], dtype=np.float32)
Ts_expected_full = np.asarray(pre["test"]["forecast_data"], dtype=np.float32)
dpdk_lookup = pre["test"]["dpdk_lookup"]
oai_lookup  = pre["test"]["oai_lookup"]
max_power   = float(env_cfg.get("max_power", 3.58))  # fixed: key not in pre["test"]

# RL agent needs window_size steps before making first decision
# Skip first window_size steps to match RL evaluation period exactly
env_cfg = config["environment"]
window_size = int(env_cfg.get("window_size", 96))
Ts_actual   = Ts_actual_full[window_size:]
Ts_expected = Ts_expected_full[window_size:]

N = len(Ts_actual)
print(f"✓ Loaded {N} test samples (skipped first {window_size} steps to match RL evaluation)")
print(f"  - First actual traffic: {Ts_actual[0]:.2f} Mbps")
print(f"  - First predicted traffic: {Ts_expected[0]:.2f} Mbps")

# ================== 2. Config ==================
# Read baseline config from the nested location in config.yaml
baseline_cfg = env_cfg.get("baseline", {})

cfg = BaselineConfig(
    performance_threshold=float(env_cfg.get("performance_threshold", 90)),
    num_oai_instances=int(env_cfg.get("num_oai_instances", 1)),
    allow_multi_oai=bool(baseline_cfg.get("allow_multi_oai", False)),
    cooldown_steps=int(env_cfg.get("cooldown_period", 4)),
    margin_mbps=float(baseline_cfg.get("margin_mbps", 23.0)),
    T_min_mbps=0.0,
    T_max_mbps=float(baseline_cfg.get("T_max_mbps", 1000.0)),
    T_step_mbps=float(baseline_cfg.get("T_step_mbps", 1.0))
)

w = BaselineWeights(
    alpha_perf=float(baseline_cfg.get("alpha_perf", 0.30)),
    beta_eff=float(baseline_cfg.get("beta_eff", 0.70))
)
selector = RuleBasedSelector(dpdk_lookup, oai_lookup, cfg=cfg, w=w)

# ================== 3. Simulation Parameters ==================
interval_h = env_cfg.get("interval_duration_hours", 0.25)
switching_time = {"dpdk_up": 50, "dpdk_down": 0, "oai_up": 7, "oai_down": 3}
use_multi_oai = cfg.allow_multi_oai
OAI_CAP_MBPS = env_cfg.get("OAI_capacity_mbps", 600)

# ================== 4. Initialize Logs ==================
current_cfg = ("DPDK", 1)
cooldown = 0
switches, downtime_s = 0, 0

# energy_dyn, qos_dyn = [], []
energy_dpdk, qos_dpdk = [], []
energy_oai, qos_oai = [], []
config_log = []

energy_exp, qos_exp = [], []
energy_act, qos_act = [], []
# Track instantaneous power directly from predictors (W)
power_exp_list, power_act_list = [], []

# ================== 5. Simulation Loop ==================
print("\nRunning rule-based simulation...\n")
for i in tqdm(range(N)):
    predicted = Ts_expected[i]
    actual    = Ts_actual[i]

    # === Decision step (based on forecasted load) ===
    utilities = {}

    # DPDK option
    perf_dpdk, power_dpdk = dpdk_predict(dpdk_lookup, predicted)
    eff_dpdk = 100.0 * (max_power - power_dpdk) / max_power
    U_dpdk = 0.3 * perf_dpdk + 0.7 * eff_dpdk
    utilities[("DPDK", 1)] = U_dpdk

    # OAI (1 instance)
    perf_oai, power_oai = oai_predict(oai_lookup, predicted, 1)
    eff_oai = 100.0 * (max_power - power_oai) / max_power
    U_oai1 = 0.3 * perf_oai + 0.7 * eff_oai
    utilities[("OAI", 1)] = U_oai1

    # OAI (multi-instance)
    if use_multi_oai:
        n = int(math.ceil(predicted / OAI_CAP_MBPS))
        n = max(1, n)
        perf_n, power_n = oai_predict(oai_lookup, predicted / n, 1)
        eff_n = 100.0 * (max_power - n * power_n) / max_power
        U_multi = 0.3 * perf_n + 0.7 * eff_n
        utilities[("OAI", n)] = U_multi

    best_cfg = max(utilities, key=utilities.get)

    # === Cooldown & Switching ===
    if best_cfg != current_cfg:
        if cooldown <= 0:
            switches += 1
            downtime_s += (
                switching_time[f"{current_cfg[0].lower()}_down"]
                + switching_time[f"{best_cfg[0].lower()}_up"]
            )
            current_cfg = best_cfg
            cooldown = cfg.cooldown_steps
    cooldown = max(0, cooldown - 1)
    config_log.append(current_cfg)

    # === Evaluation step (two perspectives: expected vs actual) ===
    # Expected → what the baseline *thought* would happen
    if current_cfg[0] == "DPDK":
        perf_exp, power_exp = dpdk_predict(dpdk_lookup, predicted)
    else:
        n = current_cfg[1]
        perf_exp, power_exp = oai_predict(oai_lookup, predicted / n, 1)
        power_exp *= n
    energy_exp.append(power_exp * interval_h)
    qos_exp.append(perf_exp)
    power_exp_list.append(float(power_exp))

    # Actual → what truly happens under real traffic
    if current_cfg[0] == "DPDK":
        perf_act, power_act = dpdk_predict(dpdk_lookup, actual)
    else:
        n = current_cfg[1]
        perf_act, power_act = oai_predict(oai_lookup, actual / n, 1)
        power_act *= n
    energy_act.append(power_act * interval_h)
    qos_act.append(perf_act)
    power_act_list.append(float(power_act))


    # Static baselines
    perf_d, power_d = dpdk_predict(dpdk_lookup, actual)
    perf_o, power_o = oai_predict(oai_lookup, actual, 1)
    energy_dpdk.append(power_d * interval_h)
    qos_dpdk.append(perf_d)
    energy_oai.append(power_o * interval_h)
    qos_oai.append(perf_o)

# ================== 6. Results ==================
E_exp, E_act = np.sum(energy_exp), np.sum(energy_act)
Q_exp, Q_act = np.mean(qos_exp), np.mean(qos_act)
E_dpdk, E_oai = np.sum(energy_dpdk), np.sum(energy_oai)
Q_dpdk, Q_oai = np.mean(qos_dpdk), np.mean(qos_oai)


best_base_energy = min(E_dpdk, E_oai)
best_base_qos = Q_oai if E_oai < E_dpdk else Q_dpdk
energy_saving = 100 * (best_base_energy - E_act) / best_base_energy
perf_change = 100 * (Q_act - best_base_qos) / best_base_qos

total_time_h = N * interval_h
downtime_pct = 100 * downtime_s / (total_time_h * 3600)

print("\n--- Simulation Summary ---")
print(f"Samples simulated: {N}")
print(f"Switches: {switches}")
print(f"Total downtime: {downtime_s:.1f}s ({downtime_pct:.3f}%)")
print(f"Dynamic total energy (expected): {E_exp:.3f} Wh")
print(f"Dynamic total energy (actual):   {E_act:.3f} Wh")
print(f"Always-DPDK energy:  {E_dpdk:.3f} Wh")
print(f"Always-OAI  energy:  {E_oai:.3f} Wh")

print(f"Avg QoS (expected): {Q_exp:.4f}")
print(f"Avg QoS (actual):   {Q_act:.4f}")

print(f"Energy saving vs best static: {energy_saving:.2f}%")
print(f"Average QoS change vs best static: {perf_change:.2f}%")

# Save results
out_df = pd.DataFrame({
    "Actual_Traffic": Ts_actual,
    "Predicted_Traffic": Ts_expected,
    "UPF_Config": config_log,
    # Energy written per 15‑min step (Wh)
    "Energy_Wh_Expected": energy_exp,
    "Energy_Wh_Actual": energy_act,
    # Instantaneous power per step in Watts (direct from predictors)
    "Power_W_Expected": power_exp_list,
    "Power_W_Actual": power_act_list,
    # QoS
    "QoS_Score_Expected": qos_exp,
    "QoS_Score_Actual": qos_act
})
os.makedirs("results/baseline", exist_ok=True)
out_df.to_csv("results/baseline/baseline_rule_results.csv", index=False)
print("✓ Saved results to results/baseline/baseline_rule_results.csv (includes Energy_Wh_* and Power_W_* columns)")
# ================== 7. Visualization (Traffic-focused with UPF zones) ==================
import matplotlib.pyplot as plt

print("✓ Generating traffic-focused timeline plot with UPF background zones...")

# Build time index
t = np.arange(len(out_df))

# Define colors for each UPF type
upf_colors = {
    'OAI': '#E8F4F8',      # Light blue for OAI
    'DPDK': '#FFF4E6',     # Light orange for DPDK
}

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(16, 6))

# ====================== Background zones for UPF states ======================
current_upf = config_log[0][0]  # 'OAI' or 'DPDK'
start_idx = 0

for i in range(1, len(config_log) + 1):
    # Check if we've reached end or if UPF type changed
    if i == len(config_log) or config_log[i][0] != current_upf:
        # Fill the region with appropriate color
        color = upf_colors.get(current_upf, '#F5F5F5')
        ax.axvspan(start_idx, i-1, alpha=0.3, color=color, zorder=1)
        
        if i < len(config_log):
            current_upf = config_log[i][0]
            start_idx = i

# ====================== Traffic lines ======================
ax.plot(t, out_df["Actual_Traffic"], label="Actual Traffic", color="green", linewidth=1.5, zorder=3)
ax.plot(t, out_df["Predicted_Traffic"], label="Predicted Traffic", color="orange", 
        linewidth=1.5, linestyle="--", alpha=0.7, zorder=3)

# ====================== Threshold lines ======================
thresholds = []
if hasattr(selector, "thresholds"):
    thresholds = list(selector.thresholds.values())
elif hasattr(selector, "Th_dict"):
    thresholds = list(selector.Th_dict.values())

if thresholds:
    for j, Th in enumerate(thresholds, start=1):
        ax.axhline(y=Th, color="red", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
        
        # Margin zone around threshold
        ax.axhline(y=Th + cfg.margin_mbps, color="blue", linestyle=":", linewidth=1, alpha=0.6, zorder=2)
        ax.axhline(y=Th - cfg.margin_mbps, color="blue", linestyle=":", linewidth=1, alpha=0.6, zorder=2)

# ====================== Crossover/Switch events ======================
# Mark switch points with vertical dashed lines
switch_idx = []
for i in range(1, len(config_log)):
    if config_log[i] != config_log[i-1]:
        switch_idx.append(i)
        ax.axvline(x=i, color="green", linestyle="--", linewidth=1, alpha=0.5, zorder=2)

# ====================== Legend and labels ======================
from matplotlib.patches import Patch
legend_elements = [
    plt.Line2D([0], [0], color='green', linewidth=1.5, label='Actual Traffic'),
    plt.Line2D([0], [0], color='orange', linewidth=1.5, linestyle='--', label='Predicted Traffic'),
    Patch(facecolor=upf_colors['OAI'], alpha=0.3, label='OAI Active'),
    Patch(facecolor=upf_colors['DPDK'], alpha=0.3, label='DPDK Active'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Threshold'),
    plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, label=f'Crossover ({len(switch_idx)})'),
    plt.Line2D([0], [0], color='blue', linestyle=':', linewidth=1, label=f'(+/-) Margin ({cfg.margin_mbps} Mbps)')
]

# Format title with parameters
title = f"Predictive UPF Selection (Utility Weights P:{w.alpha_perf:.1f}/E:{w.beta_eff:.1f}, Margin:{cfg.margin_mbps:.0f}, Hold:{cfg.cooldown_steps})"
ax.set_title(title, fontsize=12, fontweight='bold')

ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Throughput (Mbps)', fontsize=11)
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.3, zorder=0)
ax.set_xlim(left=0, right=len(t)-1)

# --- Layout & Save ---
plt.tight_layout()
fig_dir = "results/figures/baseline"
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "baseline_rule_timeline_enhanced.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Traffic-focused timeline plot saved to {fig_path}")
