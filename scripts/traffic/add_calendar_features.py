# scripts/traffic/add_calendar_features.py

import os
import yaml
import pandas as pd
import numpy as np

def load_config(path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def add_calendar_features(df, timestamp_col='timestamp'):
    """
    Add calendar/temporal features to help the model learn time-based patterns.
    
    Args:
        df: DataFrame with traffic data
        timestamp_col: Name of timestamp column
    
    Returns:
        DataFrame with added calendar features
    """
    # Ensure timestamp is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract time components
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    df['quarter'] = df[timestamp_col].dt.quarter
    
    # Cyclical encoding for hour (0-23 wraps around)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week (0-6 wraps around)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Binary features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['is_night'] = df['hour'].between(22, 23) | df['hour'].between(0, 5)
    df['is_night'] = df['is_night'].astype(int)
    
    # Peak hours (typical high-traffic times)
    df['is_morning_peak'] = df['hour'].between(7, 9).astype(int)
    df['is_evening_peak'] = df['hour'].between(17, 20).astype(int)
    
    return df

def main():
    """
    Add calendar features to the processed traffic dataset.
    """
    print("--- Adding Calendar Features to Traffic Data ---")
    
    config = load_config()
    proc_config = config['traffic_preprocessing']
    processed_csv_path = proc_config['processed_csv_path']
    
    print(f"Loading data from: {processed_csv_path}")
    
    # Load data
    try:
        df = pd.read_csv(processed_csv_path)
    except FileNotFoundError:
        print(f"\nERROR: File not found at '{processed_csv_path}'")
        print("Please run the preprocessing script first.")
        return
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Add calendar features
    df = add_calendar_features(df, timestamp_col=proc_config['timestamp_col'])
    
    print(f"\nNew shape: {df.shape}")
    print(f"Added features: {[col for col in df.columns if col not in ['timestamp', 'Traffic_bps', 'Traffic_bps_scaled', 'Traffic_Mbps_scaled', 'Traffic_Predicted_Mbps', 'Traffic_Prediction_Uncertainty']]}")
    
    # Save updated dataset
    df.to_csv(processed_csv_path, index=False)
    print(f"\n✓ Successfully saved data with calendar features to: {processed_csv_path}")
    
    # Show sample
    print(f"\n--- Sample of Data with Calendar Features ---")
    sample_cols = ['timestamp', 'Traffic_Mbps_scaled', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours']
    print(df[sample_cols].head(10).to_string(index=False))

if __name__ == '__main__':
    main()
