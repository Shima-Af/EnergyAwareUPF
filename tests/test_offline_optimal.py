"""
Unit tests for the offline-optimal dynamic-programming baseline
"""

import numpy as np

from src.offline_optimal import brute_force_best_objective, solve_offline_optimal_actions


class TestOfflineOptimalSolver:

    def test_cooldown_zero_matches_bruteforce(self):
        rng = np.random.default_rng(123)
        horizon = 5
        num_actions = 3

        traffic = np.array([85.0, 110.0, 130.0, 95.0, 75.0], dtype=np.float64)
        power = rng.uniform(0.45, 1.05, size=(horizon, num_actions)).astype(np.float64)
        perf = rng.uniform(0.86, 0.98, size=(horizon, num_actions)).astype(np.float64)

        env_cfg = {
            "performance_threshold": 0.90,
            "cooldown_period": 0,
        }
        reward_cfg = {
            "sec_scale": 100.0,
            "sec_eps_mbps": 1e-6,
            "qos_lambda": 30.0,
            "type_switch_cost": 0.03,
            "scale_up_cost_per_inst": 0.012,
            "scale_down_cost_per_inst": 0.003,
        }

        solved = solve_offline_optimal_actions(
            traffic_steps=traffic,
            power_table=power,
            perf_table=perf,
            env_config=env_cfg,
            reward_config=reward_cfg,
            initial_config=0,
            initial_counter=0,
        )

        brute = brute_force_best_objective(
            traffic_steps=traffic,
            power_table=power,
            perf_table=perf,
            env_config=env_cfg,
            reward_config=reward_cfg,
            initial_config=0,
            initial_counter=0,
        )

        assert np.isclose(float(solved["objective_value"]), float(brute), atol=1e-8, rtol=0.0)

    def test_single_action_sequence_constant(self):
        traffic = np.array([40.0, 45.0, 50.0, 55.0], dtype=np.float64)
        power = np.array([[0.81], [0.82], [0.83], [0.84]], dtype=np.float64)
        perf = np.array([[0.96], [0.95], [0.97], [0.96]], dtype=np.float64)

        env_cfg = {
            "performance_threshold": 0.90,
            "cooldown_period": 4,
        }
        reward_cfg = {
            "sec_scale": 100.0,
            "sec_eps_mbps": 1e-6,
            "qos_lambda": 30.0,
            "type_switch_cost": 0.03,
            "scale_up_cost_per_inst": 0.012,
            "scale_down_cost_per_inst": 0.003,
        }

        solved = solve_offline_optimal_actions(
            traffic_steps=traffic,
            power_table=power,
            perf_table=perf,
            env_config=env_cfg,
            reward_config=reward_cfg,
            initial_config=0,
            initial_counter=4,
        )

        assert np.all(np.asarray(solved["requested_actions"], dtype=int) == 0)
        assert np.all(np.asarray(solved["executed_actions"], dtype=int) == 0)

    def test_identical_configs_avoid_switching(self):
        traffic = np.array([70.0, 80.0, 90.0, 100.0], dtype=np.float64)
        pcol = np.array([0.75, 0.76, 0.77, 0.78], dtype=np.float64)
        qcol = np.array([0.95, 0.95, 0.95, 0.95], dtype=np.float64)

        power = np.column_stack([pcol, pcol, pcol])
        perf = np.column_stack([qcol, qcol, qcol])

        env_cfg = {
            "performance_threshold": 0.90,
            "cooldown_period": 2,
        }
        reward_cfg = {
            "sec_scale": 100.0,
            "sec_eps_mbps": 1e-6,
            "qos_lambda": 30.0,
            "type_switch_cost": 0.05,
            "scale_up_cost_per_inst": 0.020,
            "scale_down_cost_per_inst": 0.010,
        }

        solved = solve_offline_optimal_actions(
            traffic_steps=traffic,
            power_table=power,
            perf_table=perf,
            env_config=env_cfg,
            reward_config=reward_cfg,
            initial_config=0,
            initial_counter=2,
        )

        assert np.all(np.asarray(solved["executed_actions"], dtype=int) == 0)
