# scripts/Baseline/baseline_rule.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

# -------- lookups --------

def _nearest_from_sorted(keys: np.ndarray, x: float) -> int:
    idx = np.searchsorted(keys, x, side="left")
    if idx == 0:
        return 0
    if idx >= len(keys):
        return len(keys) - 1
    return idx if (keys[idx] - x) < (x - keys[idx - 1]) else idx - 1

def dpdk_predict(dpdk_lookup: Dict[float, Tuple[float, float]], T: float) -> Tuple[float, float]:
    """Return (P_u(T), W_u(T)) for DPDK, where P_u is a positive QoS score (0..100)."""
    ks = np.array(sorted(dpdk_lookup.keys()), dtype=np.float32)
    vals = np.array([dpdk_lookup[float(k)] for k in ks], dtype=np.float32)
    j = _nearest_from_sorted(ks, T)
    perf, power = vals[j]
    return float(perf), float(power)

def oai_predict(oai_lookup: Dict[str, np.ndarray], T: float, k: int) -> Tuple[float, float]:
    """
    OAI with k instances. Lookup arrays are per-instance; power sums over k.
    Return (P_u(T), W_u(T)) where P_u is *per-instance* QoS score (0..100).
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

# -------- config --------

@dataclass
class BaselineWeights:
    # Utility weights (sum≈1): U = alpha * PerfScore + beta * EffScore (higher is better)
    alpha_perf: float = 0.30  # QoS/Performance weight
    beta_eff: float  = 0.70   # Efficiency weight

@dataclass
class BaselineConfig:
    performance_threshold: float                 # used for violation rate reporting
    num_oai_instances: int
    allow_multi_oai: bool = True
    cooldown_steps: int = 4                      # 4*15min = 1h
    margin_mbps: float = 0.0                     # hysteresis band (e.g., forecast MAE)
    T_min_mbps: float = 0.0
    T_max_mbps: float = 1000.0
    T_step_mbps: float = 1.0

class RuleBasedSelector:
    """
    Paper-consistent utility baseline (higher = better):

        U_u(T) = α * P_u(T) + β * E_u(T),
        E_u(T) = 100 * (W_max - W_u(T)) / (W_max - W_min),

    with W_min/max computed *globally* across all options on a throughput grid.
    Crosspoints define thresholds; margin + cooldown avoid ping-pong.
    """

    def __init__(self,
                 dpdk_lookup: Dict[float, Tuple[float, float]],
                 oai_lookup: Dict[str, np.ndarray],
                 cfg: BaselineConfig,
                 w: BaselineWeights = BaselineWeights()):
        self.dpdk_lookup = dpdk_lookup
        self.oai_lookup  = oai_lookup
        self.cfg = cfg
        self.w   = w

        self.thresholds = self._compute_thresholds()   # {k: Th_k}, with Th_0 := 0
        self._enforce_monotone()

        self.last_action: int = 0   # 0=DPDK, 1..K=OAI-k
        self.cooldown: int = 0

    def reset(self, initial_action: int = 0):
        self.last_action = int(initial_action)
        self.cooldown = 0

    def decide(self, T_mbps: float) -> int:
        """Return 0 (DPDK) or k in [1..K_eff] (OAI-k), with hysteresis & cooldown."""
        K = self._K_eff()
        margin = self.cfg.margin_mbps
        Th = self.thresholds

        # region selection
        if T_mbps >= Th[K] + margin:
            candidate = 0
        else:
            candidate = 1
            for k in range(1, K + 1):
                left  = Th.get(k - 1, 0.0) - margin
                right = Th[k] + margin
                if left <= T_mbps < right:
                    candidate = k
                    break

        # cooldown hold
        if self.cooldown > 0:
            self.cooldown -= 1
            return self.last_action

        if candidate == self.last_action:
            return candidate

        # no-switch band near boundary
        if candidate == 0 or self.last_action == 0:
            if abs(T_mbps - Th[K]) <= margin:
                return self.last_action
        else:
            bound = Th[max(candidate, self.last_action)]
            if abs(T_mbps - bound) <= margin:
                return self.last_action

        # commit
        self.last_action = candidate
        self.cooldown = self.cfg.cooldown_steps
        return candidate

    # ---- internals ----

    def _K_eff(self) -> int:
        return self.cfg.num_oai_instances if self.cfg.allow_multi_oai else min(1, self.cfg.num_oai_instances)

    def _compute_thresholds(self) -> Dict[int, float]:
        """
        Compute crosspoints where U_DPDK(T) == U_OAI-k(T) for each k in [1..K_eff].
        Uses refined utility formulation:
            - QoS barrier (sharp drop below target)
            - Fair efficiency normalization (global Wmin/Wmax)
            - Optional utilization shaping for OAI
        """
        K = self._K_eff()
        cfg, W = self.cfg, self.w   # rename to 'W' to avoid collision
        grid = np.arange(cfg.T_min_mbps, cfg.T_max_mbps + 1e-9, cfg.T_step_mbps, dtype=np.float32)

        # --- Collect data for global normalization ---
        P_d, PWR_d = [], []
        P_k, PWR_k = {k: [] for k in range(1, K + 1)}, {k: [] for k in range(1, K + 1)}
        for T in grid:
            perf_d, power_d = dpdk_predict(self.dpdk_lookup, float(T))
            P_d.append(perf_d); PWR_d.append(power_d)
            for k in range(1, K + 1):
                perf_o, power_o = oai_predict(self.oai_lookup, float(T), k)
                P_k[k].append(perf_o); PWR_k[k].append(power_o)

        W_all = np.concatenate([np.asarray(PWR_d, np.float32)] +
                               [np.asarray(PWR_k[k], np.float32) for k in range(1, K + 1)])
        W_min, W_max = float(np.min(W_all)), float(np.max(W_all))
        denom = (W_max - W_min) + 1e-12

        # --- Utility parameters ---
        q_target = cfg.performance_threshold
        lambda_q = 6.0
        T_opt = 0.8 * cfg.T_max_mbps / max(1, K)
        gamma_util = 5.0

        def q_eff(q):
            deficit = max(0.0, q_target - q)
            return q - lambda_q * deficit

        def util_penalty(T, n):
            if n <= 0:
                return 0.0
            u = T / (n * T_opt)
            over = max(0.0, u - 1.0)
            return gamma_util * (over ** 2)

        # --- Compute utilities ---
        U_d = []
        U_o = {k: [] for k in range(1, K + 1)}
        for i, T in enumerate(grid):
            p_d, pw_d = float(P_d[i]), float(PWR_d[i])
            e_d = 100.0 * (W_max - pw_d) / denom
            q_d = q_eff(p_d)
            U_d.append(W.alpha_perf * q_d + W.beta_eff * e_d)

            for k in range(1, K + 1):
                p_o, pw_o = float(P_k[k][i]), float(PWR_k[k][i])
                e = 100.0 * (W_max - pw_o) / denom
                q = q_eff(p_o)
                penalty = util_penalty(T, k)
                U_o[k].append(W.alpha_perf * q + W.beta_eff * e - penalty)

        U_d = np.asarray(U_d, np.float32)
        for k in range(1, K + 1):
            U_o[k] = np.asarray(U_o[k], np.float32)

        # --- Determine thresholds ---
        Th = {1: 0.0}
        for k in range(1, K + 1):
            diff = U_d - U_o[k]
            s = np.sign(diff)
            flips = np.where(np.diff(s) != 0)[0]
            if flips.size > 0:
                i = int(flips[0])
                x0, x1 = float(grid[i]), float(grid[i + 1])
                y0, y1 = float(diff[i]), float(diff[i + 1])
                Th[k] = float(x0 - y0 * (x1 - x0) / (y1 - y0 + 1e-12))
            else:
                Th[k] = float(grid[int(np.argmin(np.abs(diff)))])

        
                # === Diagnostic plot of utilities ===
        plt.figure(figsize=(10, 6))
        plt.plot(grid, U_d, label="DPDK", color="red")
        for k in range(1, K + 1):
            plt.plot(grid, U_o[k], label=f"OAI-{k}", linestyle="--")
        for k, v in Th.items():
            plt.axvline(v, color="gray", linestyle=":", label=f"Th{k}={v:.1f}")
        plt.xlabel("Traffic (Mbps)")
        plt.ylabel("Utility (normalized)")
        plt.title("Utility Curves — Diagnostic")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        os.makedirs("results/figures/baseline", exist_ok=True)
        plt.savefig("results/figures/baseline/utility_diagnostics.png", dpi=150)
        print("✓ Saved utility diagnostic plot to results/figures/baseline/utility_diagnostics.png")

        # Save thresholds to JSON
        import json
        os.makedirs("results/baseline", exist_ok=True)
        threshold_info = {
            "thresholds_mbps": {f"Th_{k}": float(v) for k, v in Th.items()},
            "config": {
                "allow_multi_oai": self.cfg.allow_multi_oai,
                "num_oai_instances": self.cfg.num_oai_instances,
                "effective_instances": K,
                "alpha_perf": self.w.alpha_perf,
                "beta_eff": self.w.beta_eff
            }
        }
        with open("results/baseline/rule_thresholds.json", "w", encoding="utf-8") as f:
            json.dump(threshold_info, f, indent=2)
        print(f"✓ Saved thresholds to results/baseline/rule_thresholds.json")
        print(f"  Computed thresholds for {K} OAI instance(s): {Th}")

        return Th

    def _enforce_monotone(self):
        K = self._K_eff()
        for k in range(2, K + 1):
            if self.thresholds[k] < self.thresholds[k - 1]:
                self.thresholds[k] = self.thresholds[k - 1]
