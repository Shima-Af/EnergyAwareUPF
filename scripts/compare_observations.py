#!/usr/bin/env python3
"""
Compare Observation Schemas
============================
Shows example observations from forecast vs hybrid schemas to understand
the structural differences.

Usage:
    python scripts/compare_observations.py
"""
import sys
import os
import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_and_preprocess_data, create_vectorized_envs

def compare_schemas():
    """Compare forecast and hybrid observation spaces."""
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    schemas = ["forecast", "hybrid"]
    
    print("=" * 80)
    print("OBSERVATION SCHEMA COMPARISON: Forecast vs Hybrid")
    print("=" * 80)
    print()
    
    for schema in schemas:
        print(f"\n{'─' * 80}")
        print(f"Schema: {schema.upper()}")
        print('─' * 80)
        
        # Update schema in config
        config["environment"]["observation_schema"] = schema
        
        # Load data
        data_for_env = load_and_preprocess_data(config)
        
        # Show data split info
        train_steps = len(data_for_env['train']['traffic_data'])
        test_steps = len(data_for_env['test']['traffic_data'])
        print(f"\nData Split:")
        print(f"  Train steps: {train_steps}")
        print(f"  Test steps:  {test_steps}")
        
        # Create environment (use train env for demo)
        train_env, eval_env = create_vectorized_envs(config, data_for_env)
        venv = eval_env  # Use eval env (single env, easier to inspect)
        
        # Reset to get initial observation
        obs = venv.reset()
        
        # Get observation space info
        obs_space = venv.observation_space
        
        print(f"\nObservation Space Shape: {obs_space.shape}")
        print(f"Observation Space Low:   {obs_space.low[:10]}... (first 10)")
        print(f"Observation Space High:  {obs_space.high[:10]}... (first 10)")
        print(f"\nActual Observation Shape: {obs.shape}")
        print(f"Sample Observation (first env, first 15 dims):")
        print(f"  {obs[0, :15]}")
        
        # Identify key positions based on schema
        if schema == "forecast":
            print(f"\nStructure breakdown:")
            print(f"  [0]     : forecast_t+H = {obs[0, 0]:.4f}")
            print(f"  [1]     : config_code  = {obs[0, 1]:.0f}")
            print(f"  [2]     : cooldown     = {obs[0, 2]:.0f}")
            print(f"  [3...]  : extra features (dynamics, capacity, calendar...)")
        else:  # hybrid
            W = int(config['environment']['window_size'])
            print(f"\nStructure breakdown (W={W}):")
            print(f"  [0..{W-1}] : history window (traffic_t-{W-1}...traffic_t)")
            print(f"             Sample: {obs[0, :min(5, W)]}... (first 5)")
            print(f"  [{W}]    : forecast_t+H = {obs[0, W]:.4f}")
            print(f"  [{W+1}]  : config_code  = {obs[0, W+1]:.0f}")
            print(f"  [{W+2}]  : cooldown     = {obs[0, W+2]:.0f}")
            print(f"  [{W+3}...]: extra features")
        
        # Step the environment a few times to see dynamics
        print(f"\nAfter taking 3 actions (action=0, DPDK):")
        for step_num in range(1, 4):
            obs, reward, done, info = venv.step([0])  # action 0 = DPDK
            if schema == "forecast":
                print(f"  Step {step_num}: forecast={obs[0, 0]:.4f}, config={obs[0, 1]:.0f}, cooldown={obs[0, 2]:.0f}")
            else:
                W = int(config['environment']['window_size'])
                print(f"  Step {step_num}: history[-1]={obs[0, W-1]:.4f}, forecast={obs[0, W]:.4f}, config={obs[0, W+1]:.0f}")
        
        venv.close()
        train_env.close()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("• Forecast: Agent only sees FUTURE prediction (1 value)")
    print("           → Reactive, no memory of recent trends")
    print()
    print("• Hybrid:   Agent sees PAST+FUTURE (W history + 1 forecast)")
    print("           → Can detect trends, slopes, volatility patterns")
    print("           → Richer input but also higher dimensional")
    print("=" * 80)

if __name__ == "__main__":
    compare_schemas()
