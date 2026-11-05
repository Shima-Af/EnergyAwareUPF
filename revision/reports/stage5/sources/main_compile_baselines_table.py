#!/usr/bin/env python3
"""
Compile baseline evaluation table + per-step CSVs for paper results.
Controllers : Rule-based, Static DPDK, Static OAI-1×, Offline Optimal
Traces      : Main (Netflix), YouTube
Metrics     : E_tot (Wh), SEC (W/Mbps), v (violation rate), n_sw, n_sc, n_bl
Output dir  : polishe_results/
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import utils
from baseline_rule import dpdk_predict, oai_predict

OUT_DIR = os.path.join(ROOT, "polishe_results")
os.makedirs(OUT_DIR, exist_ok=True)

INTERVAL_H = 0.25   # 15-min steps

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trace(config_path, data_csv_override=None):
    config = utils.load_config(config_path)
    if data_csv_override:
        config["paths"]["traffic_data_csv"] = data_csv_override
    env_cfg = config["environment"]
    pre = utils.load_and_preprocess_data(config)
    window_size = int(env_cfg.get("window_size", 95))
    Ts_act = np.asarray(pre["test"]["traffic_data"],  dtype=np.float64)[window_size:]
    Ts_exp = np.asarray(pre["test"]["forecast_data"], dtype=np.float64)[window_size:]
    return {
        "Ts_act":        Ts_act,
        "Ts_exp":        Ts_exp,
        "dpdk_lookup":   pre["test"]["dpdk_lookup"],
        "oai_lookup":    pre["test"]["oai_lookup"],
        "interval_h":    float(env_cfg.get("interval_duration_hours", INTERVAL_H)),
        "qos_threshold": float(env_cfg.get("performance_threshold", 0.9)),
        "max_power":     float(env_cfg.get("max_power", 3.58)),
        "cooldown_steps":int(env_cfg.get("cooldown_period", 4)),
    }

# ---------------------------------------------------------------------------
# Per-step helpers
# ---------------------------------------------------------------------------

def _step_row(step, T_act, T_pred, upf_label, pwr, qos, interval_h, qos_threshold):
    return {
        "step":                   step,
        "actual_traffic_mbps":    round(float(T_act), 4),
        "predicted_traffic_mbps": round(float(T_pred), 4) if T_pred is not None else None,
        "upf_config":             upf_label,
        "power_w":                round(float(pwr), 6),
        "energy_wh":              round(float(pwr * interval_h), 6),
        "qos":                    round(float(qos), 6),
        "sec":                    round(float(pwr / max(T_act, 1e-6)), 8),
        "qos_violation":          int(qos < qos_threshold),
    }

# ---------------------------------------------------------------------------
# Controller functions  (return summary dict + per-step DataFrame)
# ---------------------------------------------------------------------------

def static_metrics(d, upf="dpdk", n=1):
    Ts_act        = d["Ts_act"]
    dpdk_lookup   = d["dpdk_lookup"]
    oai_lookup    = d["oai_lookup"]
    interval_h    = d["interval_h"]
    qos_threshold = d["qos_threshold"]
    label         = "DPDK" if upf == "dpdk" else f"OAI-{n}x"

    rows = []
    for i, T in enumerate(Ts_act):
        if upf == "dpdk":
            perf, pwr = dpdk_predict(dpdk_lookup, float(T))
        else:
            perf, pwr = oai_predict(oai_lookup, float(T), n)
        rows.append(_step_row(i, T, None, label, pwr, perf, interval_h, qos_threshold))

    df = pd.DataFrame(rows)
    E_tot = df["energy_wh"].sum()
    SEC   = df["sec"].mean()
    v     = df["qos_violation"].mean()
    summary = {"E_tot": E_tot, "SEC": SEC, "v": v, "n_sw": 0, "n_sc": 0, "n_bl": 0}
    return summary, df


def rule_based_metrics(d):
    Ts_act        = d["Ts_act"]
    Ts_exp        = d["Ts_exp"]
    dpdk_lookup   = d["dpdk_lookup"]
    oai_lookup    = d["oai_lookup"]
    interval_h    = d["interval_h"]
    qos_threshold = d["qos_threshold"]
    max_power     = d["max_power"]
    cooldown_steps= d["cooldown_steps"]

    current_cfg = ("DPDK", 1)
    cooldown    = 0
    n_sw = 0
    n_bl = 0
    rows = []

    for i, (T_pred, T_act) in enumerate(zip(Ts_exp, Ts_act)):
        T_pred, T_act = float(T_pred), float(T_act)

        perf_d, pwr_d = dpdk_predict(dpdk_lookup, T_pred)
        U_dpdk = 0.3 * perf_d + 0.7 * 100.0 * (max_power - pwr_d) / max_power

        perf_o, pwr_o = oai_predict(oai_lookup, T_pred, 1)
        U_oai  = 0.3 * perf_o + 0.7 * 100.0 * (max_power - pwr_o) / max_power

        best_cfg = ("DPDK", 1) if U_dpdk >= U_oai else ("OAI", 1)

        if best_cfg != current_cfg:
            if cooldown <= 0:
                n_sw       += 1
                current_cfg = best_cfg
                cooldown    = cooldown_steps
            else:
                n_bl += 1
        cooldown = max(0, cooldown - 1)

        if current_cfg[0] == "DPDK":
            perf_a, pwr_a = dpdk_predict(dpdk_lookup, T_act)
            label = "DPDK"
        else:
            perf_a, pwr_a = oai_predict(oai_lookup, T_act, 1)
            label = "OAI-1x"

        rows.append(_step_row(i, T_act, T_pred, label, pwr_a, perf_a, interval_h, qos_threshold))

    df = pd.DataFrame(rows)
    E_tot = df["energy_wh"].sum()
    SEC   = df["sec"].mean()
    v     = df["qos_violation"].mean()
    summary = {"E_tot": E_tot, "SEC": SEC, "v": v, "n_sw": n_sw, "n_sc": 0, "n_bl": n_bl}
    return summary, df


def offline_optimal_metrics(summary_csv, eval_csv):
    s   = pd.read_csv(summary_csv).iloc[0]
    raw = pd.read_csv(eval_csv)

    # Keep only the columns we need, rename to match other CSVs
    df = pd.DataFrame({
        "step":                   range(len(raw)),
        "actual_traffic_mbps":    raw["traffic"].round(4),
        "predicted_traffic_mbps": None,
        "upf_config":             raw["executed_upf_label"],
        "power_w":                raw["power"].round(6),
        "energy_wh":              (raw["power"] * INTERVAL_H).round(6),
        "qos":                    raw["performance"].round(6),
        "sec":                    raw["sec"].round(8),
        "qos_violation":          (raw["performance"] < float(s["qos_threshold"])).astype(int),
    })

    summary = {
        "E_tot": float(s["total_energy_wh"]),
        "SEC":   float(s["avg_sec"]),
        "v":     float(s["violation_rate"]),
        "n_sw":  int(s["switch_dpdk_to_oai"]) + int(s["switch_oai_to_dpdk"]),
        "n_sc":  int(s["scale_up"])            + int(s["scale_down"]),
        "n_bl":  int(s["cooldown_blocked"]),
    }
    return summary, df

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG  = os.path.join(ROOT, "config.yaml")
YT_CSV  = os.path.join(ROOT, "data/processed/processed_traffic_youtube_with_forecast.csv")

TRACES = [
    ("Main",    None,   "main"),
    ("YouTube", YT_CSV, "youtube"),
]

OFFLINE_SUMMARY = {
    "Main":    os.path.join(ROOT, "experiments/results/Baselines/"
                            "baseline_offline_optimal_summary_d1_mlp_hybrid_cd4_k2_t90.csv"),
    "YouTube": os.path.join(ROOT, "experiments/results/youtube_eval_mainconfig_allseeds/baselines/"
                            "baseline_offline_optimal_summary_youtube_mlp_hybrid_seed456.csv"),
}
OFFLINE_EVAL = {
    "Main":    os.path.join(ROOT, "experiments/results/Baselines/"
                            "offline_optimal_eval_d1_mlp_hybrid_cd4_k2_t90.csv"),
    "YouTube": os.path.join(ROOT, "experiments/results/youtube_eval_mainconfig_allseeds/baselines/"
                            "offline_optimal_eval_youtube_mlp_hybrid_seed456.csv"),
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

summary_rows = []

for trace_name, csv_override, tag in TRACES:
    print(f"\n{'='*60}")
    print(f"Trace: {trace_name}")
    print(f"{'='*60}")
    d = load_trace(CONFIG, data_csv_override=csv_override)
    print(f"  {len(d['Ts_act'])} test steps")

    CONTROLLERS = [
        ("Rule-based",    lambda d=d: rule_based_metrics(d)),
        ("Static-DPDK",   lambda d=d: static_metrics(d, upf="dpdk")),
        ("Static-OAI-1x", lambda d=d: static_metrics(d, upf="oai", n=1)),
        ("Offline-Optimal", lambda: offline_optimal_metrics(
            OFFLINE_SUMMARY[trace_name], OFFLINE_EVAL[trace_name])),
    ]

    for ctrl_name, fn in CONTROLLERS:
        summary, df_steps = fn()

        # Save per-step CSV
        fname = f"steps_{tag}_{ctrl_name.lower().replace(' ', '_').replace('-', '_')}.csv"
        df_steps.to_csv(os.path.join(OUT_DIR, fname), index=False)

        print(f"  {ctrl_name:20s}  E={summary['E_tot']:.3f} Wh  SEC={summary['SEC']:.5f}"
              f"  v={summary['v']:.4f}  n_sw={summary['n_sw']}"
              f"  n_sc={summary['n_sc']}  n_bl={summary['n_bl']}"
              f"  → {fname}")

        summary_rows.append({"Trace": trace_name, "Controller": ctrl_name, **summary})

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

df_summary = pd.DataFrame(
    summary_rows,
    columns=["Trace", "Controller", "E_tot", "SEC", "v", "n_sw", "n_sc", "n_bl"]
)
df_summary["E_tot"] = df_summary["E_tot"].round(3)
df_summary["SEC"]   = df_summary["SEC"].round(6)
df_summary["v"]     = df_summary["v"].round(4)

print(f"\n{'='*80}")
print("SUMMARY TABLE — old traffic predictor, margin=23 Mbps")
print(f"{'='*80}")
print(df_summary.to_string(index=False))

summary_path = os.path.join(OUT_DIR, "tab_baselines_both_traces.csv")
df_summary.to_csv(summary_path, index=False)
print(f"\nSaved summary → {summary_path}")
print(f"Saved per-step CSVs → {OUT_DIR}/steps_*.csv")
