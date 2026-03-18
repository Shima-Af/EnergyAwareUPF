from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .environment import ManualCooldownEnv


@dataclass
class OfflineOptimalResult:
    requested_actions: np.ndarray
    executed_actions: np.ndarray
    objective_value: float
    evaluation_df: pd.DataFrame
    summary: dict[str, Any]


def _normalize_objective_mode(objective_mode: str) -> str:
    mode = str(objective_mode).strip().lower()
    if mode not in {"discounted", "undiscounted"}:
        raise ValueError(
            f"Unsupported objective_mode='{objective_mode}'. "
            "Expected one of: discounted, undiscounted."
        )
    return mode


def _objective_from_rewards(rewards: np.ndarray, objective_mode: str, gamma: float) -> float:
    rew = np.asarray(rewards, dtype=np.float64)
    mode = _normalize_objective_mode(objective_mode)
    if mode == "undiscounted":
        return float(rew.sum())

    gamma_value = float(gamma)
    if not 0.0 <= gamma_value <= 1.0:
        raise ValueError(f"For discounted objective, gamma must be in [0, 1], got {gamma_value}.")

    if rew.size == 0:
        return 0.0
    discounts = np.power(gamma_value, np.arange(rew.size, dtype=np.float64))
    return float(np.dot(discounts, rew))


def _action_series_from_eval(df: pd.DataFrame) -> pd.Series:
    if "executed_action" in df.columns:
        return pd.to_numeric(df["executed_action"], errors="coerce").fillna(0).astype(int)
    if "executed_code" in df.columns:
        return pd.to_numeric(df["executed_code"], errors="coerce").fillna(0).astype(int)
    if "requested_action" in df.columns:
        return pd.to_numeric(df["requested_action"], errors="coerce").fillna(0).astype(int)
    return pd.Series(np.zeros(len(df), dtype=int))


def _prepare_transition_tables(
    num_actions: int,
    cooldown_period: int,
    type_switch_cost: float,
    scale_up_cost: float,
    scale_down_cost: float,
) -> dict[str, np.ndarray]:
    counter_cap = max(int(cooldown_period), 0)
    n_ctr = counter_cap + 1

    executed_cfg = np.zeros((num_actions, n_ctr, num_actions), dtype=np.int16)
    next_counter = np.zeros((num_actions, n_ctr, num_actions), dtype=np.int16)
    blocked = np.zeros((num_actions, n_ctr, num_actions), dtype=bool)
    type_penalty = np.zeros((num_actions, n_ctr, num_actions), dtype=np.float64)
    scale_penalty = np.zeros((num_actions, n_ctr, num_actions), dtype=np.float64)

    for old_cfg in range(num_actions):
        old_type = 0 if old_cfg == 0 else 1
        for old_ctr in range(n_ctr):
            tick_counter = min(old_ctr + 1, counter_cap)
            is_blocked = tick_counter < cooldown_period
            for req in range(num_actions):
                exe = old_cfg if is_blocked else req
                new_ctr = 0 if exe != old_cfg else tick_counter

                new_type = 0 if exe == 0 else 1

                t_pen = 0.0
                s_pen = 0.0
                if new_type != old_type:
                    t_pen = float(type_switch_cost)
                else:
                    delta_k = exe - old_cfg
                    if delta_k > 0:
                        s_pen = float(scale_up_cost) * float(delta_k)
                    elif delta_k < 0:
                        s_pen = float(scale_down_cost) * float(-delta_k)

                executed_cfg[old_cfg, old_ctr, req] = np.int16(exe)
                next_counter[old_cfg, old_ctr, req] = np.int16(new_ctr)
                blocked[old_cfg, old_ctr, req] = bool(is_blocked and req != old_cfg)
                type_penalty[old_cfg, old_ctr, req] = t_pen
                scale_penalty[old_cfg, old_ctr, req] = s_pen

    return {
        "counter_cap": np.array(counter_cap, dtype=np.int16),
        "executed_cfg": executed_cfg,
        "next_counter": next_counter,
        "blocked": blocked,
        "type_penalty": type_penalty,
        "scale_penalty": scale_penalty,
    }


def solve_offline_optimal_actions(
    traffic_steps: np.ndarray,
    power_table: np.ndarray,
    perf_table: np.ndarray,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    initial_config: int = 0,
    initial_counter: int | None = None,
    objective_mode: str = "discounted",
    gamma: float = 0.995,
) -> dict[str, Any]:
    if power_table.shape != perf_table.shape:
        raise ValueError("power_table and perf_table must have identical shape.")
    if power_table.shape[0] != len(traffic_steps):
        raise ValueError("traffic_steps length must equal the number of table rows.")

    horizon, num_actions = power_table.shape
    cooldown_period = int(env_config.get("cooldown_period", 0))
    performance_threshold = float(env_config["performance_threshold"])

    sec_scale = float(reward_config.get("sec_scale", 1.0))
    sec_eps = float(reward_config.get("sec_eps_mbps", 1e-6))
    qos_lambda = float(reward_config.get("qos_lambda", 0.0))
    type_switch_cost = float(reward_config.get("type_switch_cost", 0.0))
    scale_up_cost = float(reward_config.get("scale_up_cost_per_inst", 0.0))
    scale_down_cost = float(reward_config.get("scale_down_cost_per_inst", 0.0))

    transitions = _prepare_transition_tables(
        num_actions=num_actions,
        cooldown_period=cooldown_period,
        type_switch_cost=type_switch_cost,
        scale_up_cost=scale_up_cost,
        scale_down_cost=scale_down_cost,
    )
    executed_cfg = transitions["executed_cfg"]
    next_counter = transitions["next_counter"]
    n_ctr = executed_cfg.shape[1]

    init_ctr = cooldown_period if initial_counter is None else int(initial_counter)
    init_ctr = max(0, min(init_ctr, n_ctr - 1))

    mode = _normalize_objective_mode(objective_mode)
    gamma_value = float(gamma)
    if mode == "discounted" and not (0.0 <= gamma_value <= 1.0):
        raise ValueError(f"For discounted objective, gamma must be in [0, 1], got {gamma_value}.")
    future_weight = gamma_value if mode == "discounted" else 1.0

    dp = np.full((horizon + 1, num_actions, n_ctr), -np.inf, dtype=np.float64)
    policy = np.zeros((horizon, num_actions, n_ctr), dtype=np.int16)
    dp[horizon, :, :] = 0.0

    for t in range(horizon - 1, -1, -1):
        sec_vec = power_table[t, :] / np.maximum(float(traffic_steps[t]), sec_eps)
        qos_pen_vec = qos_lambda * np.maximum(performance_threshold - perf_table[t, :], 0.0)
        base_vec = -sec_scale * sec_vec - qos_pen_vec

        for old_cfg in range(num_actions):
            for old_ctr in range(n_ctr):
                best_value = -np.inf
                best_action = 0
                best_pref = -1
                for req in range(num_actions):
                    exe = int(executed_cfg[old_cfg, old_ctr, req])
                    nxt_ctr = int(next_counter[old_cfg, old_ctr, req])
                    step_reward = (
                        float(base_vec[exe])
                        - float(transitions["type_penalty"][old_cfg, old_ctr, req])
                        - float(transitions["scale_penalty"][old_cfg, old_ctr, req])
                    )
                    cand = step_reward + future_weight * float(dp[t + 1, exe, nxt_ctr])
                    pref = 1 if req == old_cfg else 0
                    if cand > best_value + 1e-12:
                        best_value = cand
                        best_action = req
                        best_pref = pref
                    elif abs(cand - best_value) <= 1e-12:
                        if pref > best_pref or (pref == best_pref and req < best_action):
                            best_value = cand
                            best_action = req
                            best_pref = pref
                dp[t, old_cfg, old_ctr] = best_value
                policy[t, old_cfg, old_ctr] = np.int16(best_action)

    requested = np.zeros(horizon, dtype=np.int16)
    executed = np.zeros(horizon, dtype=np.int16)
    blocked = np.zeros(horizon, dtype=bool)

    old_cfg = int(initial_config)
    old_ctr = init_ctr
    for t in range(horizon):
        req = int(policy[t, old_cfg, old_ctr])
        exe = int(executed_cfg[old_cfg, old_ctr, req])
        requested[t] = np.int16(req)
        executed[t] = np.int16(exe)
        blocked[t] = bool(transitions["blocked"][old_cfg, old_ctr, req])
        old_ctr = int(next_counter[old_cfg, old_ctr, req])
        old_cfg = exe

    return {
        "requested_actions": requested.astype(int),
        "executed_actions": executed.astype(int),
        "cooldown_blocked": blocked,
        "objective_value": float(dp[0, int(initial_config), init_ctr]),
        "transitions": transitions,
    }


def simulate_requested_actions_from_tables(
    requested_actions: np.ndarray,
    traffic_steps: np.ndarray,
    power_table: np.ndarray,
    perf_table: np.ndarray,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    initial_config: int = 0,
    initial_counter: int | None = None,
    objective_mode: str = "discounted",
    gamma: float = 0.995,
) -> dict[str, Any]:
    requested_actions = np.asarray(requested_actions, dtype=int)
    horizon, num_actions = power_table.shape
    if len(requested_actions) != horizon:
        raise ValueError("requested_actions length must match table horizon.")

    cooldown_period = int(env_config.get("cooldown_period", 0))
    performance_threshold = float(env_config["performance_threshold"])

    sec_scale = float(reward_config.get("sec_scale", 1.0))
    sec_eps = float(reward_config.get("sec_eps_mbps", 1e-6))
    qos_lambda = float(reward_config.get("qos_lambda", 0.0))
    type_switch_cost = float(reward_config.get("type_switch_cost", 0.0))
    scale_up_cost = float(reward_config.get("scale_up_cost_per_inst", 0.0))
    scale_down_cost = float(reward_config.get("scale_down_cost_per_inst", 0.0))

    transitions = _prepare_transition_tables(
        num_actions=num_actions,
        cooldown_period=cooldown_period,
        type_switch_cost=type_switch_cost,
        scale_up_cost=scale_up_cost,
        scale_down_cost=scale_down_cost,
    )
    executed_cfg = transitions["executed_cfg"]
    next_counter = transitions["next_counter"]
    n_ctr = executed_cfg.shape[1]

    old_cfg = int(initial_config)
    old_ctr = cooldown_period if initial_counter is None else int(initial_counter)
    old_ctr = max(0, min(old_ctr, n_ctr - 1))

    rewards = np.zeros(horizon, dtype=np.float64)
    executed = np.zeros(horizon, dtype=np.int16)
    blocked = np.zeros(horizon, dtype=bool)

    for t in range(horizon):
        req = int(requested_actions[t])
        if req < 0 or req >= num_actions:
            raise ValueError(f"Requested action {req} out of range [0, {num_actions - 1}].")

        exe = int(executed_cfg[old_cfg, old_ctr, req])
        sec = float(power_table[t, exe]) / max(float(traffic_steps[t]), sec_eps)
        qos_pen = qos_lambda * max(performance_threshold - float(perf_table[t, exe]), 0.0)

        reward = (
            -sec_scale * sec
            - float(transitions["type_penalty"][old_cfg, old_ctr, req])
            - float(transitions["scale_penalty"][old_cfg, old_ctr, req])
            - qos_pen
        )

        rewards[t] = reward
        executed[t] = np.int16(exe)
        blocked[t] = bool(transitions["blocked"][old_cfg, old_ctr, req])

        old_ctr = int(next_counter[old_cfg, old_ctr, req])
        old_cfg = exe

    return {
        "requested_actions": requested_actions.astype(int),
        "executed_actions": executed.astype(int),
        "cooldown_blocked": blocked,
        "rewards": rewards,
        "objective_value": _objective_from_rewards(rewards, objective_mode=objective_mode, gamma=gamma),
    }


def _build_prediction_tables(env: ManualCooldownEnv, step_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_actions = int(env.action_space.n)
    horizon = int(len(step_indices))
    perf_table = np.zeros((horizon, num_actions), dtype=np.float64)
    power_table = np.zeros((horizon, num_actions), dtype=np.float64)

    for row, idx in enumerate(step_indices):
        traffic = float(env.traffic_data[int(idx)])
        for action_code in range(num_actions):
            if action_code == 0:
                perf, power = env._get_prediction(traffic, 0, 0)
            else:
                perf, power = env._get_prediction(traffic, 1, int(action_code))
            perf_table[row, action_code] = float(perf)
            power_table[row, action_code] = float(power)

    return perf_table, power_table


def replay_requested_actions(
    env_payload: dict[str, Any],
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    requested_actions: np.ndarray,
) -> pd.DataFrame:
    env = ManualCooldownEnv(
        **env_payload,
        env_config=env_config,
        reward_config=reward_config,
    )
    _obs, _info = env.reset()

    rows: list[dict[str, Any]] = []
    for action in requested_actions:
        _obs, reward, terminated, truncated, info = env.step(int(action))
        row = dict(info)
        row["reward"] = float(reward)
        rows.append(row)
        if terminated or truncated:
            break

    df = pd.DataFrame(rows)
    exec_codes = _action_series_from_eval(df)
    df["executed_upf_label"] = exec_codes.apply(lambda c: "DPDK" if int(c) == 0 else f"{int(c)}xOAI")
    return df


def summarize_eval_dataframe(
    df: pd.DataFrame,
    performance_threshold: float,
    step_hours: float = 0.25,
) -> dict[str, Any]:
    action = _action_series_from_eval(df)
    prev = action.shift(1)

    power = pd.to_numeric(df.get("power"), errors="coerce")
    traffic = pd.to_numeric(df.get("traffic"), errors="coerce")
    sec = pd.to_numeric(df.get("sec"), errors="coerce")
    if sec.isna().all():
        sec = power / np.maximum(traffic, 1e-6)
    qos = pd.to_numeric(df.get("performance"), errors="coerce")

    if "cooldown_blocked" in df.columns:
        cooldown_blocked = int(pd.to_numeric(df["cooldown_blocked"], errors="coerce").fillna(0).astype(bool).sum())
    elif {"requested_action", "executed_action"}.issubset(df.columns):
        req = pd.to_numeric(df["requested_action"], errors="coerce")
        exe = pd.to_numeric(df["executed_action"], errors="coerce")
        cooldown_blocked = int(req.ne(exe).fillna(False).sum())
    else:
        cooldown_blocked = 0

    switch_dpdk_to_oai = int(((prev == 0) & (action > 0)).sum())
    switch_oai_to_dpdk = int(((prev > 0) & (action == 0)).sum())
    scale_up = int(((prev > 0) & (action > 0) & (action > prev)).sum())
    scale_down = int(((prev > 0) & (action > 0) & (action < prev)).sum())
    switching_count = int(action.diff().fillna(0).ne(0).sum())

    if qos.notna().any():
        violation_rate = float((qos < float(performance_threshold)).mean())
        avg_qos = float(qos.mean())
    else:
        qpen = pd.to_numeric(df.get("qos_penalty"), errors="coerce")
        violation_rate = float((qpen > 0).mean()) if qpen.notna().any() else float("nan")
        avg_qos = float("nan")

    total_power = float(power.sum())
    steps = int(len(df))

    return {
        "steps": steps,
        "switch_dpdk_to_oai": switch_dpdk_to_oai,
        "switch_oai_to_dpdk": switch_oai_to_dpdk,
        "scale_up": scale_up,
        "scale_down": scale_down,
        "cooldown_blocked": cooldown_blocked,
        "switching_count": switching_count,
        "scaling_count": int(scale_up + scale_down),
        "avg_sec": float(sec.mean()),
        "total_power": total_power,
        "avg_qos": avg_qos,
        "violation_rate": violation_rate,
        "avg_power_w": float(total_power / max(steps, 1)),
        "total_energy_wh": float(total_power * step_hours),
        "total_reward": float(pd.to_numeric(df.get("reward"), errors="coerce").sum()),
    }


def baseline_style_dataframe(df: pd.DataFrame, step_hours: float = 0.25) -> pd.DataFrame:
    action = _action_series_from_eval(df)
    upf_cfg = action.apply(lambda c: "('DPDK', 1)" if int(c) == 0 else f"('OAI', {int(c)})")
    out = pd.DataFrame(
        {
            "Actual_Traffic": pd.to_numeric(df.get("traffic"), errors="coerce"),
            "Predicted_Traffic": pd.to_numeric(df.get("predicted_traffic"), errors="coerce"),
            "UPF_Config": upf_cfg,
            "Energy_Wh_Actual": pd.to_numeric(df.get("power"), errors="coerce") * float(step_hours),
            "QoS_Score_Actual": pd.to_numeric(df.get("performance"), errors="coerce"),
            "Requested_Action": pd.to_numeric(df.get("requested_action"), errors="coerce"),
            "Executed_Action": action,
            "Cooldown_Blocked": pd.to_numeric(df.get("cooldown_blocked"), errors="coerce").fillna(0).astype(bool),
            "Reward": pd.to_numeric(df.get("reward"), errors="coerce"),
            "SEC": pd.to_numeric(df.get("sec"), errors="coerce"),
        }
    )
    return out


def parse_upf_config_code(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    text = str(val).strip()
    if text.startswith("('DPDK'"):
        return 0.0
    if text.startswith("('OAI'"):
        try:
            pieces = text.strip("()")
            right = pieces.split(",", maxsplit=1)[1]
            return float(int(right.strip()))
        except Exception:
            return np.nan
    return np.nan


def summarize_baseline_csv(
    csv_path: str | Path,
    performance_threshold: float,
    step_hours: float = 0.25,
) -> dict[str, Any]:
    df = pd.read_csv(csv_path)

    traffic = pd.to_numeric(df.get("Actual_Traffic"), errors="coerce")
    qos = pd.to_numeric(df.get("QoS_Score_Actual"), errors="coerce")

    if "Power_W_Actual" in df.columns:
        power = pd.to_numeric(df.get("Power_W_Actual"), errors="coerce")
    else:
        power = pd.to_numeric(df.get("Energy_Wh_Actual"), errors="coerce") / float(step_hours)

    sec = power / np.maximum(traffic, 1e-6)
    actions = df.get("UPF_Config", pd.Series([np.nan] * len(df))).map(parse_upf_config_code)

    prev = actions.shift(1)
    switch_dpdk_to_oai = int(((prev == 0) & (actions > 0)).sum())
    switch_oai_to_dpdk = int(((prev > 0) & (actions == 0)).sum())
    scale_up = int(((prev > 0) & (actions > 0) & (actions > prev)).sum())
    scale_down = int(((prev > 0) & (actions > 0) & (actions < prev)).sum())

    switching_count = int(actions.diff().fillna(0).ne(0).sum())
    total_power = float(power.sum())
    steps = int(len(df))

    return {
        "steps": steps,
        "switch_dpdk_to_oai": switch_dpdk_to_oai,
        "switch_oai_to_dpdk": switch_oai_to_dpdk,
        "scale_up": scale_up,
        "scale_down": scale_down,
        "cooldown_blocked": 0,
        "switching_count": switching_count,
        "scaling_count": int(scale_up + scale_down),
        "avg_sec": float(sec.mean()),
        "total_power": total_power,
        "avg_qos": float(qos.mean()),
        "violation_rate": float((qos < float(performance_threshold)).mean()),
        "avg_power_w": float(total_power / max(steps, 1)),
        "total_energy_wh": float(total_power * step_hours),
        "total_reward": float("nan"),
    }


def _evaluation_step_indices(traffic_data: np.ndarray, window_size: int) -> np.ndarray:
    start = int(window_size) - 1
    stop = int(len(traffic_data)) - 1
    if stop <= start:
        return np.array([], dtype=np.int32)
    return np.arange(start, stop, dtype=np.int32)


def compute_offline_optimal(
    env_payload: dict[str, Any],
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    objective_tolerance: float = 1e-5,
    objective_mode: str = "discounted",
    gamma: float = 0.995,
) -> OfflineOptimalResult:
    env_for_tables = ManualCooldownEnv(
        **env_payload,
        env_config=env_config,
        reward_config=reward_config,
    )

    step_indices = _evaluation_step_indices(env_for_tables.traffic_data, int(env_for_tables.window_size))
    traffic_steps = env_for_tables.traffic_data[step_indices].astype(np.float64)
    perf_table, power_table = _build_prediction_tables(env_for_tables, step_indices)

    solved = solve_offline_optimal_actions(
        traffic_steps=traffic_steps,
        power_table=power_table,
        perf_table=perf_table,
        env_config=env_config,
        reward_config=reward_config,
        initial_config=0,
        initial_counter=int(env_for_tables.cooldown_period),
        objective_mode=objective_mode,
        gamma=gamma,
    )

    replay_df = replay_requested_actions(
        env_payload=env_payload,
        env_config=env_config,
        reward_config=reward_config,
        requested_actions=solved["requested_actions"],
    )

    replay_summary = summarize_eval_dataframe(
        replay_df,
        performance_threshold=float(env_config["performance_threshold"]),
    )

    replay_rewards = pd.to_numeric(replay_df.get("reward"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    replay_objective = _objective_from_rewards(replay_rewards, objective_mode=objective_mode, gamma=gamma)
    obj = float(solved["objective_value"])
    if not np.isclose(replay_objective, obj, atol=objective_tolerance, rtol=0.0):
        raise ValueError(
            f"Offline-optimal objective mismatch ({_normalize_objective_mode(objective_mode)}): "
            f"DP={obj:.10f}, replay={replay_objective:.10f}."
        )

    replay_summary["objective_mode"] = _normalize_objective_mode(objective_mode)
    replay_summary["objective_gamma"] = float(gamma) if replay_summary["objective_mode"] == "discounted" else np.nan

    replay_exec = _action_series_from_eval(replay_df).to_numpy(dtype=int)
    if len(replay_exec) != len(solved["executed_actions"]):
        raise ValueError("Executed action length mismatch between DP reconstruction and replay.")
    if not np.array_equal(replay_exec, np.asarray(solved["executed_actions"], dtype=int)):
        raise ValueError("Executed action mismatch between DP reconstruction and replay.")

    return OfflineOptimalResult(
        requested_actions=np.asarray(solved["requested_actions"], dtype=int),
        executed_actions=np.asarray(solved["executed_actions"], dtype=int),
        objective_value=obj,
        evaluation_df=replay_df,
        summary=replay_summary,
    )


def brute_force_best_objective(
    traffic_steps: np.ndarray,
    power_table: np.ndarray,
    perf_table: np.ndarray,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    initial_config: int = 0,
    initial_counter: int | None = None,
    objective_mode: str = "discounted",
    gamma: float = 0.995,
) -> float:
    horizon, num_actions = power_table.shape
    best = -np.inf
    for seq in product(range(num_actions), repeat=horizon):
        sim = simulate_requested_actions_from_tables(
            requested_actions=np.array(seq, dtype=int),
            traffic_steps=traffic_steps,
            power_table=power_table,
            perf_table=perf_table,
            env_config=env_config,
            reward_config=reward_config,
            initial_config=initial_config,
            initial_counter=initial_counter,
            objective_mode=objective_mode,
            gamma=gamma,
        )
        best = max(best, float(sim["objective_value"]))
    return float(best)
