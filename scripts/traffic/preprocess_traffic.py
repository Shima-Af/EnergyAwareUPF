# scripts/preprocess_traffic.py

import os
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_config(path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    """
    Main script to load raw traffic data, scale it to a target peak,
    and save the processed file for the RL pipeline.
    """
    # 1. Setup: Load config and setup command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess raw traffic data.")
    parser.add_argument("--plot", action="store_true", help="Display a plot of original vs. scaled traffic.")
    args = parser.parse_args()
    
    config = load_config()
    proc_config = config['traffic_preprocessing']

    raw_path = proc_config['raw_csv_path']
    processed_path = proc_config['processed_csv_path']
    
    print(f"--- Starting Traffic Preprocessing ---")
    print(f"Loading raw data from: {raw_path}")

    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"\nERROR: Raw traffic file not found at '{raw_path}'")
        print("Please make sure the file exists and the path in config.yaml is correct.")
        return

    # 2. Preprocessing Logic (adapted from your function)
    df[proc_config['timestamp_col']] = pd.to_datetime(df[proc_config['timestamp_col']], format='%d/%m/%Y %H:%M')
    
    traffic_col = proc_config['traffic_col_bps']
    target_peak_bps = proc_config['target_peak_mbps'] * 1e6 # Convert Mbps from config to bps for calculation
    
    current_max = df[traffic_col].max()
    current_min = df[traffic_col].min()
    
    if current_max <= 0:
        print("Warning: Original maximum traffic is zero or negative. Scaling skipped.")
        df['Traffic_bps_scaled'] = df[traffic_col]
    elif proc_config['scaling_method'] == 'ratio':
        scale_factor = target_peak_bps / current_max
        df['Traffic_bps_scaled'] = df[traffic_col] * scale_factor
        print(f"Applied 'ratio' scaling with factor: {scale_factor:.4f}")
    elif proc_config['scaling_method'] == 'minmax':
        df['Traffic_bps_scaled'] = ((df[traffic_col] - current_min) / (current_max - current_min)) * target_peak_bps
        print("Applied 'minmax' scaling.")
    else:
        print("Warning: Unknown scaling method. Using original traffic.")
        df['Traffic_bps_scaled'] = df[traffic_col]

    # Add the final column needed by the RL environment (in Mbps)
    df['Traffic_Mbps_scaled'] = df['Traffic_bps_scaled'] / 1e6
    
    # 3. Save the Processed Data
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\n✓ Successfully processed and saved data to: {processed_path}")
 
if __name__ == '__main__':
    main()