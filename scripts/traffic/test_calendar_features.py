#!/usr/bin/env python3
# scripts/traffic/test_calendar_features.py
"""
Quick test script to verify calendar features integration.
"""

import yaml
import pandas as pd

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("="*70)
    print("CALENDAR FEATURES TEST")
    print("="*70)
    
    config = load_config()
    train_config = config['traffic_predictor_training']
    proc_config = config['traffic_preprocessing']
    
    # Check config
    use_calendar = train_config.get('use_calendar_features', False)
    calendar_cols = train_config.get('calendar_feature_columns', [])
    
    print(f"\n1. CONFIG CHECK")
    print(f"   use_calendar_features: {use_calendar}")
    print(f"   calendar_feature_columns: {calendar_cols}")
    
    # Check data
    print(f"\n2. DATA CHECK")
    csv_path = proc_config['processed_csv_path']
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✓ CSV loaded: {csv_path}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        # Check for calendar features
        has_calendar = all(col in df.columns for col in calendar_cols)
        
        if use_calendar:
            if has_calendar:
                print(f"\n   ✓ Calendar features ENABLED and PRESENT in data")
                print(f"   Calendar columns found: {[c for c in calendar_cols if c in df.columns]}")
            else:
                print(f"\n   ⚠️  Calendar features ENABLED but MISSING in data!")
                missing = [c for c in calendar_cols if c not in df.columns]
                print(f"   Missing columns: {missing}")
                print(f"\n   → Run: python scripts/traffic/add_calendar_features.py")
        else:
            print(f"\n   ℹ️  Calendar features DISABLED in config")
            if has_calendar:
                print(f"   (Calendar columns exist in data but won't be used)")
            else:
                print(f"   (No calendar columns in data)")
    
    except FileNotFoundError:
        print(f"   ❌ CSV not found: {csv_path}")
        return
    
    # Check model files
    print(f"\n3. MODEL FILES CHECK")
    paths_config = config['paths']
    model_path = paths_config['traffic_predictor_model_path']
    metadata_path = model_path.replace('.keras', '_metadata.pkl')
    
    import os
    if os.path.exists(model_path):
        print(f"   ✓ Model exists: {model_path}")
        
        if os.path.exists(metadata_path):
            import pickle
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            model_uses_calendar = metadata.get('use_calendar_features', False)
            model_calendar_cols = metadata.get('calendar_feature_columns', [])
            
            print(f"   ✓ Metadata exists")
            print(f"     - Model uses calendar: {model_uses_calendar}")
            if model_uses_calendar:
                print(f"     - Model calendar columns: {model_calendar_cols}")
            
            # Check for mismatch
            if use_calendar != model_uses_calendar:
                print(f"\n   ⚠️  MISMATCH DETECTED!")
                print(f"     Config says: use_calendar={use_calendar}")
                print(f"     Model was trained with: use_calendar={model_uses_calendar}")
                print(f"\n   → You need to retrain the model!")
        else:
            print(f"   ⚠️  No metadata file (old model?)")
            print(f"   → Retrain to create metadata")
    else:
        print(f"   ℹ️  No model yet (needs training)")
    
    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if use_calendar and has_calendar:
        print("✓ Status: READY to train with calendar features")
        print("  Next step: python scripts/traffic/train_traffic_predictor.py")
    elif use_calendar and not has_calendar:
        print("⚠️  Status: Calendar enabled but data missing")
        print("  Next step: python scripts/traffic/add_calendar_features.py")
    elif not use_calendar:
        print("ℹ️  Status: Calendar features disabled (traffic-only mode)")
        print("  To enable: Set use_calendar_features: true in config.yaml")
    
    print("="*70)

if __name__ == '__main__':
    main()
