"""
Unit tests for agent creation and loading
"""
import pytest
import tempfile
from stable_baselines3.common.env_util import make_vec_env
from src import agent
from src.environment import ManualCooldownEnv

class TestAgent:
    
    def test_agent_creation_ppo(self, sample_config, mock_lookups):
        """Test PPO agent creation"""
        dpdk_lookup, oai_lookup = mock_lookups
        traffic = np.random.uniform(10, 100, 200).astype(np.float32)
        
        env_kwargs = {
            'traffic_data': traffic,
            'dpdk_lookup': dpdk_lookup,
            'oai_lookup': oai_lookup,
            'env_config': sample_config['environment'],
            'reward_config': sample_config['reward'],
        }
        
        env = make_vec_env(ManualCooldownEnv, n_envs=1, env_kwargs=env_kwargs)
        
        model = agent.create_agent(env, sample_config)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'learn')
    
    def test_agent_creation_recurrent(self, sample_config, mock_lookups):
        """Test RecurrentPPO agent creation"""
        sample_config['agent']['policy'] = 'MlpLstmPolicy'
        
        dpdk_lookup, oai_lookup = mock_lookups
        traffic = np.random.uniform(10, 100, 200).astype(np.float32)
        
        env_kwargs = {
            'traffic_data': traffic,
            'dpdk_lookup': dpdk_lookup,
            'oai_lookup': oai_lookup,
            'env_config': sample_config['environment'],
            'reward_config': sample_config['reward'],
        }
        
        env = make_vec_env(ManualCooldownEnv, n_envs=1, env_kwargs=env_kwargs)
        
        model = agent.create_agent(env, sample_config)
        
        assert model is not None
        assert hasattr(model.policy, 'is_recurrent')
    
    def test_agent_save_load(self, sample_config, mock_lookups, tmp_path):
        """Test agent saving and loading"""
        dpdk_lookup, oai_lookup = mock_lookups
        traffic = np.random.uniform(10, 100, 200).astype(np.float32)
        
        env_kwargs = {
            'traffic_data': traffic,
            'dpdk_lookup': dpdk_lookup,
            'oai_lookup': oai_lookup,
            'env_config': sample_config['environment'],
            'reward_config': sample_config['reward'],
        }
        
        env = make_vec_env(ManualCooldownEnv, n_envs=1, env_kwargs=env_kwargs)
        
        model = agent.create_agent(env, sample_config)
        
        # Save
        model_path = tmp_path / "test_model"
        model.save(str(model_path))
        
        # Load
        loaded_model = agent.load_agent(f"{model_path}.zip", env=env)
        
        assert loaded_model is not None