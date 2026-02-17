"""
Integration tests for full training/evaluation pipeline
"""
import pytest
import os
import tempfile
from src import utils, agent

class TestTrainingPipeline:
    
    @pytest.mark.slow
    def test_minimal_training_run(self, sample_config, sample_traffic_data, tmp_path, mock_lookups):
        """Test a minimal training run completes without errors"""
        
        # Update config for fast test
        sample_config['training']['total_timesteps'] = 100
        sample_config['training']['num_cpu'] = 1
        sample_config['paths']['best_model_save_path'] = str(tmp_path)
        
        traffic, forecast = sample_traffic_data
        dpdk_lookup, oai_lookup = mock_lookups
        
        # Create minimal data payload
        data_for_env = {
            'train': {
                'traffic_data': traffic[:400],
                'dpdk_lookup': dpdk_lookup,
                'oai_lookup': oai_lookup,
                'forecast_data': forecast[:400],
            },
            'test': {
                'traffic_data': traffic[400:],
                'dpdk_lookup': dpdk_lookup,
                'oai_lookup': oai_lookup,
                'forecast_data': forecast[400:],
            }
        }
        
        train_env, eval_env = utils.create_vectorized_envs(sample_config, data_for_env)
        model = agent.create_agent(train_env, sample_config)
        
        # Short training
        model.learn(total_timesteps=100)
        
        # Save
        model_path = tmp_path / "test_model"
        model.save(str(model_path))
        
        assert os.path.exists(f"{model_path}.zip")