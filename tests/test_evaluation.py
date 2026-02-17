"""
Tests for evaluation metrics and baseline comparisons
"""
import pytest
import numpy as np
from src.evaluate import predict_dpdk_from_lookup, predict_oai_from_lookup

class TestEvaluationMetrics:
    
    def test_dpdk_baseline_prediction(self, mock_lookups):
        """Test DPDK baseline predictions"""
        dpdk_lookup, _ = mock_lookups
        
        perf, power = predict_dpdk_from_lookup(dpdk_lookup, 50.0)
        
        assert isinstance(perf, (int, float))
        assert isinstance(power, (int, float))
        assert perf > 0
        assert power > 0
    
    def test_oai_baseline_prediction(self, mock_lookups):
        """Test OAI baseline predictions"""
        _, oai_lookup = mock_lookups
        
        for k in [1, 2]:
            perf, power = predict_oai_from_lookup(oai_lookup, 50.0, k)
            
            assert isinstance(perf, (int, float))
            assert isinstance(power, (int, float))
            assert perf > 0
            assert power > 0