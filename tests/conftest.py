"""
Pytest configuration and shared fixtures
"""
import pytest
import numpy as np
import yaml
import tempfile
import os
from pathlib import Path

@pytest.fixture
def sample_config():
    """Minimal valid configuration for testing"""
    return {
        'paths': {
            'traffic_data_csv': 'data/traffic_test.csv',
            'prediction_model_dir': 'models/',
            'best_model_save_path': 'saved_models/test',
            'results_dir': 'eval/',
        },
        'training': {
            'test_size': 0.2,
            'seed': 42,
            'num_cpu': 1,
            'total_timesteps': 1000,
            'eval_freq_denom': 1000,
            'n_eval_episodes': 1,
            'deterministic_eval': True,
        },
        'environment': {
            'traffic_column': 'Traffic_Mbps_scaled',
            'forecast_column': 'Forecast_Mbps',
            'window_size': 96,
            'forecast_horizon': 1,
            'performance_threshold': 5.0,
            'cooldown_period': 5,
            'num_oai_instances': 2,
            'observation_schema': 'hybrid',
            'use_dyn_features': True,
            'use_capacity_features': False,
            'use_powergap_features': False,
            'use_calendar_features': True,
        },
        'reward': {
            'sec_scale': 1.0,
            'sec_eps_mbps': 1e-6,
            'qos_lambda': 1.0,
            'type_switch_cost': 0.1,
            'scale_cost': 0.05,
        },
        'agent': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
        },
        'simulation_mode': {
            'type': 'precompute',
        },
        'approximator_models': {
            'dpdk': {'type': 'keras'},
            'oai': {'type': 'keras'},
        },
    }

@pytest.fixture
def sample_traffic_data():
    """Generate synthetic traffic data for testing"""
    np.random.seed(42)
    n_samples = 500
    traffic = np.random.uniform(10, 100, n_samples).astype(np.float32)
    forecast = traffic + np.random.normal(0, 5, n_samples).astype(np.float32)
    return traffic, forecast

@pytest.fixture
def temp_config_file(sample_config, tmp_path):
    """Create temporary config file"""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return str(config_path)

@pytest.fixture
def mock_lookups():
    """Mock DPDK and OAI lookup tables"""
    traffic_values = np.linspace(0, 100, 50)
    dpdk_lookup = {
        t: (float(t * 0.9), float(50 + t * 0.3)) 
        for t in traffic_values
    }
    oai_lookup = {
        'keys': traffic_values.astype(np.float32),
        'perf': (traffic_values * 0.85).astype(np.float32),
        'power': (40 + traffic_values * 0.25).astype(np.float32),
    }
    return dpdk_lookup, oai_lookup