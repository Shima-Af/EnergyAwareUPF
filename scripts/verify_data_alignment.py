#!/usr/bin/env python3
"""
Data Alignment Verification
============================
Verifies that all observation schemas produce identical test set sizes
before running expensive training experiments.

Usage:
    python scripts/verify_data_alignment.py
"""
import sys
import os
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_and_preprocess_data

def verify_alignment():
    """Check that all schemas produce identical train/test splits."""
    
    # Load base config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    schemas = ["instant", "history", "forecast", "hybrid"]
    results = []
    
    print("=" * 70)
    print("DATA ALIGNMENT VERIFICATION")
    print("=" * 70)
    print(f"Config: {config['paths']['traffic_data_csv']}")
    print(f"Test size: {config['training']['test_size']}")
    print()
    
    for schema in schemas:
        # Update schema in config
        config["environment"]["observation_schema"] = schema
        
        # Load and split data
        data_for_env = load_and_preprocess_data(config)
        
        train_steps = len(data_for_env['train']['traffic_data'])
        test_steps = len(data_for_env['test']['traffic_data'])
        
        results.append({
            "schema": schema,
            "train_steps": train_steps,
            "test_steps": test_steps,
        })
        
        print(f"{schema:10s}  train: {train_steps:4d}  test: {test_steps:4d}")
    
    # Check uniformity
    print()
    print("-" * 70)
    
    train_sizes = [r["train_steps"] for r in results]
    test_sizes = [r["test_steps"] for r in results]
    
    if len(set(train_sizes)) == 1 and len(set(test_sizes)) == 1:
        print("✅ PASS: All schemas have identical train/test sizes")
        print(f"   Train: {train_sizes[0]} steps")
        print(f"   Test:  {test_sizes[0]} steps")
        return True
    else:
        print("❌ FAIL: Schemas have different data sizes!")
        print(f"   Train sizes: {set(train_sizes)}")
        print(f"   Test sizes:  {set(test_sizes)}")
        print()
        print("This means experiments will not be fairly comparable.")
        print("Check the trimming logic in src/utils.py:load_traffic_and_split()")
        return False

if __name__ == "__main__":
    success = verify_alignment()
    sys.exit(0 if success else 1)
