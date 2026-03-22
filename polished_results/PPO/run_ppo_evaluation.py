#!/usr/bin/env python3
"""
Evaluate all PPO experiments (A1, B1, C1, D1) on both traffic traces.
Saves per-step CSVs and summary tables to polishe_results/PPO/.

Experiments
-----------
A1  : 8 policy × observation variants, seed_456
B1  : mlp_hybrid + lstm_history, 5 seeds  (paper calls this B2)
C1  : 5 cooldown/instance configs, seed_456
D1  : 8 MLP+LSTM × k × threshold configs, seed_456

Metrics (per row in summary table)
-----------------------------------
E_tot  total energy (Wh)
SEC    mean specific energy consumption (W/Mbps)
v      QoS violation rate
n_sw   type-switches  (DPDK ↔ OAI)
n_sc   scale events   (1xOAI ↔ 2xOAI)
n_bl   cooldown-blocked switch attempts
"""

import os, sys, glob, shutil, subprocess, time
import numpy as np
import pandas as pd
import yaml

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PPO_DIR = os.path.dirname(__file__)           # polishe_results/PPO/
STEPS_DIR = os.path.join(PPO_DIR, "steps")
STAGING   = os.path.join(PPO_DIR, "_yt_staging")
YT_CSV    = os.path.join(ROOT, "data/processed/processed_traffic_youtube_with_forecast.csv")

os.makedirs(STEPS_DIR, exist_ok=True)
os.makedirs(STAGING,   exist_ok=True)

INTERVAL_H = 0.25
QOS_THR    = 0.9

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "A1": {
        "base": os.path.join(ROOT, "experiments/results/A1_policy_observation"),
        "runs": [
            ("mlp_forecast",  "seed_456"),
            ("mlp_history",   "seed_456"),
            ("mlp_hybrid",    "seed_456"),
            ("mlp_instant",   "seed_456"),
            ("lstm_forecast", "seed_456"),
            ("lstm_history",  "seed_456"),
            ("lstm_hybrid",   "seed_456"),
            ("lstm_instant",  "seed_456"),
        ],
    },
    "B1": {
        "base": os.path.join(ROOT, "experiments/results/B2_top2_seeds"),
        "runs": [
            ("mlp_hybrid",    "seed_42"),
            ("mlp_hybrid",    "seed_123"),
            ("mlp_hybrid",    "seed_456"),
            ("mlp_hybrid",    "seed_789"),
            ("mlp_hybrid",    "seed_1024"),
            ("lstm_history",  "seed_42"),
            ("lstm_history",  "seed_123"),
            ("lstm_history",  "seed_456"),
            ("lstm_history",  "seed_789"),
            ("lstm_history",  "seed_1024"),
        ],
    },
    "C1": {
        "base": os.path.join(ROOT, "experiments/results/C1_cooldown_instances"),
        "runs": [
            ("cd0_k2", "seed_456"),
            ("cd2_k2", "seed_456"),
            ("cd4_k1", "seed_456"),
            ("cd4_k2", "seed_456"),
            ("cd8_k2", "seed_456"),
        ],
    },
    "D1": {
        "base": os.path.join(ROOT, "experiments/results/D1_mlp_lstm_cd4_k_threshold"),
        "runs": [
            ("mlp_hybrid_cd4_k1_t90",    "seed_456"),
            ("mlp_hybrid_cd4_k1_t95",    "seed_456"),
            ("mlp_hybrid_cd4_k2_t90",    "seed_456"),
            ("mlp_hybrid_cd4_k2_t95",    "seed_456"),
            ("lstm_history_cd4_k1_t90",  "seed_456"),
            ("lstm_history_cd4_k1_t95",  "seed_456"),
            ("lstm_history_cd4_k2_t90",  "seed_456"),
            ("lstm_history_cd4_k2_t95",  "seed_456"),
        ],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_results_csv(eval_dir):
    """Return the most recently modified evaluation_results_*.csv in eval_dir."""
    candidates = (
        glob.glob(os.path.join(eval_dir, "evaluation_results_*.csv")) +
        glob.glob(os.path.join(eval_dir, "evaluation_results.csv"))
    )
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def extract_metrics(results_csv, qos_thr=QOS_THR, interval_h=INTERVAL_H):
    """Parse an evaluation_results CSV → summary dict + clean per-step DataFrame."""
    df = pd.read_csv(results_csv)

    # Derive per-step metrics
    pwr      = df["power"].values
    traffic  = df["traffic"].values
    perf     = df["performance"].values
    sec_col  = df["sec"].values

    E_tot = float(np.sum(pwr * interval_h))
    SEC   = float(np.mean(sec_col))
    v     = float(np.mean(perf < qos_thr))
    n_sw  = int((df["type_switch_penalty"] > 0).sum())
    n_sc  = int((df["scale_penalty"]       > 0).sum())
    n_bl  = int(df["cooldown_blocked"].astype(int).sum())

    pred_col = df["predicted_traffic"] if "predicted_traffic" in df.columns else None

    clean = pd.DataFrame({
        "step":                   range(len(df)),
        "actual_traffic_mbps":    traffic.round(4),
        "predicted_traffic_mbps": pred_col.round(4).values if pred_col is not None else None,
        "upf_config":             df["executed_upf_label"].values,
        "power_w":                pwr.round(6),
        "energy_wh":              (pwr * interval_h).round(6),
        "qos":                    perf.round(6),
        "sec":                    sec_col.round(8),
        "qos_violation":          (perf < qos_thr).astype(int),
        "type_switch":            (df["type_switch_penalty"] > 0).astype(int).values,
        "scale_event":            (df["scale_penalty"]       > 0).astype(int).values,
        "cooldown_blocked":       df["cooldown_blocked"].astype(int).values,
    })

    summary = {"E_tot": E_tot, "SEC": SEC, "v": v,
               "n_sw": n_sw, "n_sc": n_sc, "n_bl": n_bl}
    return summary, clean


def run_youtube_eval(run_dir, staging_tag):
    """
    Evaluate a trained model on the YouTube trace.
    Creates a staging dir with model files + YouTube config, runs evaluate.py,
    returns (summary_dict, per_step_df).
    """
    stage = os.path.join(STAGING, staging_tag)
    eval_subdir = os.path.join(stage, "eval")
    os.makedirs(eval_subdir, exist_ok=True)

    # Modified config: point traffic CSV to YouTube
    src_config = os.path.join(run_dir, "config.yaml")
    with open(src_config) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["traffic_data_csv"] = YT_CSV
    with open(os.path.join(stage, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # Symlink model files
    for fname in ["best_model.zip", "vec_normalize_stats.pkl"]:
        src = os.path.join(run_dir, fname)
        dst = os.path.join(stage, fname)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)

    # Run evaluate.py
    cmd = [
        sys.executable, "-m", "src.evaluate",
        "--run_dir", stage,
        "--stamp",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=ROOT, env=env,
        capture_output=True, text=True
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    ✗ evaluate.py failed ({elapsed:.1f}s)")
        print(result.stderr[-500:])
        return None, None

    csv_path = latest_results_csv(eval_subdir)
    if csv_path is None:
        print(f"    ✗ No results CSV found in {eval_subdir}")
        return None, None

    summary, clean = extract_metrics(csv_path)
    return summary, clean


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

all_rows = []

for exp_name, exp_cfg in EXPERIMENTS.items():
    base_dir = exp_cfg["base"]
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*70}")

    for config_name, seed in exp_cfg["runs"]:
        run_dir  = os.path.join(base_dir, config_name, seed)
        eval_dir = os.path.join(run_dir, "eval")
        tag      = f"{exp_name}_{config_name}_{seed}"

        print(f"\n  {tag}")

        # ---- Main trace (parse existing eval CSV) ----
        csv_main = latest_results_csv(eval_dir)
        if csv_main is None:
            print(f"    ✗ No existing eval CSV for Main trace, skipping")
            continue

        sum_main, df_main = extract_metrics(csv_main)
        steps_fname_main = os.path.join(STEPS_DIR, f"steps_{tag}_main.csv")
        df_main.to_csv(steps_fname_main, index=False)
        print(f"    Main    E={sum_main['E_tot']:.3f} Wh  SEC={sum_main['SEC']:.5f}"
              f"  v={sum_main['v']:.4f}  n_sw={sum_main['n_sw']}"
              f"  n_sc={sum_main['n_sc']}  n_bl={sum_main['n_bl']}")

        all_rows.append({
            "Exp": exp_name, "Config": config_name, "Seed": seed, "Trace": "Main",
            **sum_main
        })

        # ---- YouTube trace (run evaluate.py on staging dir) ----
        sum_yt, df_yt = run_youtube_eval(run_dir, staging_tag=tag)
        if sum_yt is None:
            print(f"    YouTube ✗ evaluation failed")
            continue

        steps_fname_yt = os.path.join(STEPS_DIR, f"steps_{tag}_youtube.csv")
        df_yt.to_csv(steps_fname_yt, index=False)
        print(f"    YouTube E={sum_yt['E_tot']:.3f} Wh  SEC={sum_yt['SEC']:.5f}"
              f"  v={sum_yt['v']:.4f}  n_sw={sum_yt['n_sw']}"
              f"  n_sc={sum_yt['n_sc']}  n_bl={sum_yt['n_bl']}")

        all_rows.append({
            "Exp": exp_name, "Config": config_name, "Seed": seed, "Trace": "YouTube",
            **sum_yt
        })

# ---------------------------------------------------------------------------
# Build and save summary table
# ---------------------------------------------------------------------------
df_all = pd.DataFrame(all_rows, columns=[
    "Exp", "Config", "Seed", "Trace",
    "E_tot", "SEC", "v", "n_sw", "n_sc", "n_bl"
])
df_all["E_tot"] = df_all["E_tot"].round(3)
df_all["SEC"]   = df_all["SEC"].round(6)
df_all["v"]     = df_all["v"].round(4)

print(f"\n{'='*80}")
print("PPO EVALUATION SUMMARY — all experiments, both traces")
print(f"{'='*80}")
print(df_all.to_string(index=False))

# Full combined table
out_combined = os.path.join(PPO_DIR, "tab_ppo_all.csv")
df_all.to_csv(out_combined, index=False)
print(f"\nSaved combined table → {out_combined}")

# Per-trace tables
for trace in ["Main", "YouTube"]:
    sub = df_all[df_all["Trace"] == trace].drop(columns="Trace")
    out = os.path.join(PPO_DIR, f"tab_ppo_{trace.lower()}.csv")
    sub.to_csv(out, index=False)
    print(f"Saved {trace} table     → {out}")

# B1 aggregate (mean ± std across seeds)
b1 = df_all[df_all["Exp"] == "B1"].copy()
if not b1.empty:
    agg_rows = []
    for (config, trace), grp in b1.groupby(["Config", "Trace"]):
        for col in ["E_tot", "SEC", "v", "n_sw", "n_sc", "n_bl"]:
            pass  # just build the row
        row = {"Exp": "B1", "Config": config, "Seed": "mean±std", "Trace": trace}
        for col in ["E_tot", "SEC", "v", "n_sw", "n_sc", "n_bl"]:
            row[col] = f"{grp[col].mean():.4f}±{grp[col].std():.4f}"
        agg_rows.append(row)
    df_b1_agg = pd.DataFrame(agg_rows)
    out_b1 = os.path.join(PPO_DIR, "tab_ppo_B1_aggregate.csv")
    df_b1_agg.to_csv(out_b1, index=False)
    print(f"Saved B1 aggregate      → {out_b1}")

# Cleanup staging
shutil.rmtree(STAGING, ignore_errors=True)
print(f"\nDone. Per-step CSVs → {STEPS_DIR}/")
