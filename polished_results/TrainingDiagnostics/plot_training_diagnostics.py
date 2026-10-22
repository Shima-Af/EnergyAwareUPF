#!/usr/bin/env python3
"""Generate training convergence diagnostics figure for the paper.

HOW TO CUSTOMIZE:
  Edit the STYLE section below, then re-run:
      python3 plot_training_diagnostics.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# ██ STYLE CONFIG — edit this section to change appearance ██
# =============================================================================

# ── Font stack (matches Elsevier CAS template / notebook) ────────────────
ELSEVIER_SERIF_STACK = [
    'Times New Roman', 'Times', 'Nimbus Roman No9 L',
    'Nimbus Roman', 'Liberation Serif', 'STIXGeneral',
    'DejaVu Serif',
]

PAPER_STYLE = {
    "figure.dpi":        130,
    "savefig.dpi":       300,
    "font.family":       "serif",
    "font.serif":        ELSEVIER_SERIF_STACK,
    "font.size":         10,           # base font size
    "axes.titlesize":    14,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "mathtext.fontset":  "stix",
    "axes.unicode_minus": False,
}

# ── Figure size (width, height) in inches ─────────────────────────────────
FIGSIZE = (15, 4.2)           # combined (kept for reference)
FIGSIZE_SINGLE = (5.5, 3.8)  # individual panel size

# ── Colors (from notebook PAPER_COLORS) ───────────────────────────────────
C_FF_PPO  = "#1f77b4"       # Feed-forward PPO  (blue)
C_REC_PPO = "#2ca02c"       # Recurrent PPO     (green)
C_RULE    = "#ff7f0e"       # Rule-based        (orange)

# For the ablation panels (a)/(b) where we show 4 observation schemas,
# we use the FF/Rec colors as base, with line style distinguishing schemas.
# Solid = FF-PPO, Dashed = Rec-PPO.  Shape per observation schema.
C_INSTANT  = "#1f77b4"      # same blue family as FF-PPO
C_HISTORY  = "#ff7f0e"      # orange
C_FORECAST = "#2ca02c"      # green
C_HYBRID   = "#d62728"      # red

# ── Axis labels ───────────────────────────────────────────────────────────
XLABEL   = "Training steps [×10³]"
YLABEL_A = "Evaluation return"
YLABEL_B = "Policy entropy"
YLABEL_C = "Evaluation return"

# ── Legend ─────────────────────────────────────────────────────────────────
LEGEND_FONTSIZE_AB = 8          # panels (a) and (b)
LEGEND_FONTSIZE_C  = 9          # panel (c)
LEGEND_LOC         = "upper center"   # placed on top of each panel
LEGEND_NCOL_AB     = 4          # columns for 8 series → 2 rows
LEGEND_NCOL_C      = 2
LEGEND_FRAMEALPHA  = 0.85

# ── Lines ─────────────────────────────────────────────────────────────────
LINE_WIDTH_AB = 1.4
LINE_WIDTH_C  = 1.8
LINE_ALPHA    = 0.85
FILL_ALPHA    = 0.12      # shaded std band in panel (c)

# ── Smoothing window (moving average) ────────────────────────────────────
SMOOTH_EVAL    = 7
SMOOTH_ENTROPY = 3

# ── Annotations ──────────────────────────────────────────────────────────
SHOW_COLLAPSE_ANNOTATION = True
COLLAPSE_TEXT = "policy collapse\n(best-model checkpointing\nprevents deployment)"
COLLAPSE_FONTSIZE = 7

SHOW_ENTROPY_ZONE = True
ENTROPY_ZONE_TEXT = "entropy\ncollapse\nzone"
ENTROPY_ZONE_FONTSIZE = 7

SHOW_BEST_MODEL_LINES = True
BEST_MODEL_FONTSIZE = 7

# ── Panel labels (a), (b), (c) ──────────────────────────────────────────
PANEL_LABELS = ["(a)", "(b)", "(c)"]
PANEL_LABEL_FONTSIZE = 12
PANEL_LABEL_FONTWEIGHT = "bold"
PANEL_LABEL_X = 0.02          # axes-fraction from left
PANEL_LABEL_Y = 0.95          # axes-fraction from bottom

# ── Grid ──────────────────────────────────────────────────────────────────
GRID_ALPHA = 0.25

# =============================================================================
# ██ END STYLE CONFIG — you should not need to edit below this line ██
# =============================================================================

plt.rcParams.update(PAPER_STYLE)

# ── Paths ────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent.parent  # polished_results -> EnergyAwareUPF
A1_DIR  = ROOT / "experiments" / "results" / "A1_policy_observation"
B2_DIR  = ROOT / "experiments" / "results" / "B2_top2_seeds"
OUT_DIR = Path(__file__).resolve().parent

# ── Series configs ───────────────────────────────────────────────────────
A1_CONFIGS = {
    "mlp_instant":   {"label": "FF: instant",   "color": C_INSTANT,  "ls": "-",  "algo": "PPO_1"},
    "mlp_history":   {"label": "FF: history",   "color": C_HISTORY,  "ls": "-",  "algo": "PPO_1"},
    "mlp_forecast":  {"label": "FF: forecast",  "color": C_FORECAST, "ls": "-",  "algo": "PPO_1"},
    "mlp_hybrid":    {"label": "FF: hybrid",    "color": C_HYBRID,   "ls": "-",  "algo": "PPO_1"},
    "lstm_instant":  {"label": "Rec: instant",  "color": C_INSTANT,  "ls": "--", "algo": "RecurrentPPO_1"},
    "lstm_history":  {"label": "Rec: history",  "color": C_HISTORY,  "ls": "--", "algo": "RecurrentPPO_1"},
    "lstm_forecast": {"label": "Rec: forecast", "color": C_FORECAST, "ls": "--", "algo": "RecurrentPPO_1"},
    "lstm_hybrid":   {"label": "Rec: hybrid",   "color": C_HYBRID,   "ls": "--", "algo": "RecurrentPPO_1"},
}
B2_CONFIGS = {
    "mlp_hybrid":   {"label": "FF-PPO (hybrid)",  "color": C_FF_PPO,  "ls": "-",  "algo": "PPO_1"},
    "lstm_history": {"label": "Rec-PPO (history)", "color": C_REC_PPO, "ls": "--", "algo": "RecurrentPPO_1"},
}
SEEDS = ["seed_42", "seed_123", "seed_456", "seed_789", "seed_1024"]


# ── Helpers ──────────────────────────────────────────────────────────────
def load_eval_curves(base_dir, config, seed="seed_456"):
    data = np.load(base_dir / config / seed / "evaluations.npz")
    return data["timesteps"], data["results"].mean(axis=1)

def load_tb_scalar(base_dir, config, seed, algo, tag):
    tb_base = base_dir / config / seed / "tb_logs"
    tb_path = sorted(tb_base.iterdir())[0] / algo
    ea = EventAccumulator(str(tb_path)); ea.Reload()
    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])

def smooth(y, window=5):
    if len(y) < window: return y
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window // 2, y[0]), y, np.full(window // 2, y[-1])])
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


# ── Helper: save a single-panel figure ───────────────────────────────────
def save_panel(fig, name):
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"Saved {name}.pdf / .png")
    plt.close(fig)


# ── Panel (a): Learning curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
for config, meta in A1_CONFIGS.items():
    ts, rew = load_eval_curves(A1_DIR, config)
    ax.plot(ts / 1000, smooth(rew, SMOOTH_EVAL), label=meta["label"],
            color=meta["color"], ls=meta["ls"], lw=LINE_WIDTH_AB, alpha=LINE_ALPHA)
ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL_A)
ax.legend(fontsize=LEGEND_FONTSIZE_AB, ncol=LEGEND_NCOL_AB, loc="lower center",
          framealpha=LEGEND_FRAMEALPHA, columnspacing=1.0, handlelength=2.2,
          bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
ax.grid(True, alpha=GRID_ALPHA); ax.set_xlim(0, 305)
if SHOW_COLLAPSE_ANNOTATION:
    ax.annotate(COLLAPSE_TEXT, xy=(200, -975), xytext=(210, -800),
                fontsize=COLLAPSE_FONTSIZE, color="gray", ha="left",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
fig.tight_layout(rect=[0, 0, 1, 0.88])
save_panel(fig, "fig_training_learning_curves")


# ── Panel (b): Entropy ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
for config, meta in A1_CONFIGS.items():
    steps, entropy = load_tb_scalar(A1_DIR, config, "seed_456",
                                     meta["algo"], "train/entropy_loss")
    ax.plot(steps / 1000, smooth(-entropy, SMOOTH_ENTROPY), label=meta["label"],
            color=meta["color"], ls=meta["ls"], lw=LINE_WIDTH_AB, alpha=LINE_ALPHA)
ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL_B)
ax.legend(fontsize=LEGEND_FONTSIZE_AB, ncol=LEGEND_NCOL_AB, loc="lower center",
          framealpha=LEGEND_FRAMEALPHA, columnspacing=1.0, handlelength=2.2,
          bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
ax.grid(True, alpha=GRID_ALPHA); ax.set_xlim(0, 305)
if SHOW_ENTROPY_ZONE:
    ax.axhspan(0.0, 0.5, color="red", alpha=0.04)
    ax.text(280, 0.25, ENTROPY_ZONE_TEXT, fontsize=ENTROPY_ZONE_FONTSIZE,
            color="red", ha="center", alpha=0.6, style="italic")
fig.tight_layout(rect=[0, 0, 1, 0.88])
save_panel(fig, "fig_training_entropy")


# ── Panel (c): Multi-seed robustness ────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
for config, meta in B2_CONFIGS.items():
    all_rew = np.array([load_eval_curves(B2_DIR, config, s)[1] for s in SEEDS])
    ts_k = load_eval_curves(B2_DIR, config, SEEDS[0])[0] / 1000
    sm_mean = smooth(all_rew.mean(axis=0), SMOOTH_EVAL)
    sm_std  = smooth(all_rew.std(axis=0),  SMOOTH_EVAL)
    ax.plot(ts_k, sm_mean, label=meta["label"],
            color=meta["color"], ls=meta["ls"], lw=LINE_WIDTH_C)
    ax.fill_between(ts_k, sm_mean - sm_std, sm_mean + sm_std,
                     color=meta["color"], alpha=FILL_ALPHA)
    if SHOW_BEST_MODEL_LINES:
        mean_best = all_rew.max(axis=1).mean()
        ax.axhline(mean_best, color=meta["color"], ls=":", lw=1.0, alpha=0.6)
        ax.text(305, mean_best + 5, f"mean best\n({mean_best:.0f})",
                fontsize=BEST_MODEL_FONTSIZE, color=meta["color"], va="bottom", ha="right")
ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL_C)
ax.legend(fontsize=LEGEND_FONTSIZE_C, ncol=LEGEND_NCOL_C, loc="lower center",
          framealpha=LEGEND_FRAMEALPHA, bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
ax.grid(True, alpha=GRID_ALPHA); ax.set_xlim(0, 310)
fig.tight_layout(rect=[0, 0, 1, 0.88])
save_panel(fig, "fig_training_multiseed")
