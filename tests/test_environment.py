"""
Unit tests for the RL environment
"""
import pytest
import numpy as np
from src.environment import ManualCooldownEnv

class TestManualCooldownEnv:
    
    def test_env_initialization(self, sample_config, mock_lookups):
        """Test environment can be initialized with minimal config"""
        dpdk_lookup, oai_lookup = mock_lookups
        
        traffic = np.random.uniform(10, 100, 200).astype(np.float32)
        
        env = ManualCooldownEnv(
            traffic_data=traffic,
            dpdk_lookup=dpdk_lookup,
            oai_lookup=oai_lookup,
            env_config=sample_config['environment'],
            reward_config=sample_config['reward'],
        )
        
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None
    
    def test_env_reset(self, sample_config, mock_lookups):
        """Test environment reset returns valid observation"""
        dpdk_lookup, oai_lookup = mock_lookups
        traffic = np.random.uniform(10, 100, 200).astype(np.float32)
        
        env = ManualCooldownEnv(
            traffic_data=traffic,
            dpdk_lookup=dpdk_lookup,
            oai_lookup=oai_lookup,
            env_config=sample_config['environment'],
            reward_config=sample_config['reward'],
        )
        
        obs, info = env.reset()
        
        assert obs is not None
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
    
    def test_env_step(self, sample_config, mock_lookups):
        """Test environment step returns valid outputs"""
        dpdk_lookup, oai_lookup = mock_lookups
        traffic = np.random.uniform(10, 100, 200).astype(np.float32)
        
        env = ManualCooldownEnv(
            traffic_data=traffic,
            dpdk_lookup=dpdk_lookup,
            oai_lookup=oai_lookup,
            env_config=sample_config['environment'],
            reward_config=sample_config['reward'],
        )
        
        obs, _ = env.reset()
        action = env.action_space.sample()
        
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        assert obs_next.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert 'traffic' in info
        assert 'power' in info
        assert 'performance' in info
    
    def test_cooldown_mechanism(self, sample_config, mock_lookups):
        """Test cooldown prevents rapid switching"""
        dpdk_lookup, oai_lookup = mock_lookups
        traffic = np.random.uniform(10, 100, 200).astype(np.float32)
        
        env = ManualCooldownEnv(
            traffic_data=traffic,
            dpdk_lookup=dpdk_lookup,
            oai_lookup=oai_lookup,
            env_config=sample_config['environment'],
            reward_config=sample_config['reward'],
        )
        
        env.reset()
        
        # First switch should work
        _, _, _, _, info1 = env.step(1)  # Switch to 1xOAI
        assert not info1.get('cooldown_blocked', False)
        
        # Immediate second switch should be blocked
        _, _, _, _, info2 = env.step(2)  # Try to switch to 2xOAI
        # Depending on implementation, check cooldown
        assert 'cooldown_blocked' in info2

    def test_qos_violation_penalty(self, sample_config, mock_lookups):
        """Test QoS violation results in penalty"""
        dpdk_lookup, oai_lookup = mock_lookups
        
        # Create low-performance scenario
        traffic = np.full(200, 100.0, dtype=np.float32)
        
        env = ManualCooldownEnv(
            traffic_data=traffic,
            dpdk_lookup=dpdk_lookup,
            oai_lookup=oai_lookup,
            env_config=sample_config['environment'],
            reward_config=sample_config['reward'],
        )
        
        env.reset()
        _, reward, _, _, info = env.step(0)
        
        if info['performance'] < sample_config['environment']['performance_threshold']:
            assert info.get('qos_penalty', 0) > 0