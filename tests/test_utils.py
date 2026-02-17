"""
Unit tests for utility functions
"""
import pytest
import numpy as np
import pandas as pd
from src import utils

class TestUtilityFunctions:
    
    def test_load_config(self, temp_config_file):
        """Test configuration loading"""
        config = utils.load_config(temp_config_file)
        
        assert config is not None
        assert 'environment' in config
        assert 'training' in config
        assert 'reward' in config
    
    def test_traffic_dynamics_computation(self):
        """Test traffic dynamics feature computation"""
        traffic = np.random.uniform(10, 100, 200).astype(np.float32)
        
        dynamics = utils.compute_traffic_dynamics(
            traffic, 
            window_size=96, 
            ema_short=4, 
            ema_long=16
        )
        
        assert 'feature_names' in dynamics
        assert len(dynamics['feature_names']) > 0
        
        # Check all feature arrays have correct length
        for key, value in dynamics.items():
            if key != 'feature_names' and isinstance(value, np.ndarray):
                assert len(value) == len(traffic)
    
    def test_calendar_features(self):
        """Test calendar feature computation"""
        # Create sample dataframe with timestamps
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min')
        })
        
        cal_features = utils.compute_calendar_features(
            df, 
            timestamp_col='timestamp'
        )
        
        assert 'feature_names' in cal_features
        assert 'sin_hour' in cal_features
        assert 'cos_hour' in cal_features
        assert len(cal_features['sin_hour']) == len(df)