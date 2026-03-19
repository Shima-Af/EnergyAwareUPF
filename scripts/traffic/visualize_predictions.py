# scripts/traffic/visualize_predictions.py

import os
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def load_config(path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def plot_predictions(df, start_date=None, end_date=None, save_path=None):
    """
    Plot actual traffic vs predicted traffic with uncertainty bands.
    
    Args:
        df: DataFrame with traffic data and predictions
        start_date: Start date for plot (string or datetime)
        end_date: End date for plot (string or datetime)
        save_path: Path to save the figure
    """
    # Filter data if dates provided
    df_plot = df.copy()
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        df_plot = df_plot[df_plot['timestamp'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df_plot = df_plot[df_plot['timestamp'] <= end_date]
    
    # Drop rows without predictions
    df_plot = df_plot.dropna(subset=['Traffic_Predicted_Mbps'])
    
    if len(df_plot) == 0:
        print("No data to plot in the specified date range.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Actual vs Predicted Traffic
    ax1 = axes[0]
    ax1.plot(df_plot['timestamp'], df_plot['Traffic_Mbps_scaled'], 
             label='Actual Traffic', color='#2E86AB', linewidth=1.5, alpha=0.8)
    ax1.plot(df_plot['timestamp'], df_plot['Traffic_Predicted_Mbps'], 
             label='Predicted Traffic', color='#A23B72', linewidth=1.5, alpha=0.8, linestyle='--')
    
    ax1.set_ylabel('Traffic (Mbps)', fontsize=12)
    ax1.set_title('Traffic Prediction: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Plot 2: Prediction with Uncertainty Bands
    ax2 = axes[1]
    ax2.plot(df_plot['timestamp'], df_plot['Traffic_Mbps_scaled'], 
             label='Actual Traffic', color='#2E86AB', linewidth=1.5, alpha=0.8)
    ax2.plot(df_plot['timestamp'], df_plot['Traffic_Predicted_Mbps'], 
             label='Predicted Traffic', color='#A23B72', linewidth=1.5)
    
    # Add uncertainty bands (± 1 sigma and ± 2 sigma)
    predicted = df_plot['Traffic_Predicted_Mbps'].values
    uncertainty = df_plot['Traffic_Prediction_Uncertainty'].values
    timestamps = df_plot['timestamp'].values
    
    ax2.fill_between(timestamps, 
                     predicted - uncertainty, 
                     predicted + uncertainty,
                     alpha=0.3, color='#F18F01', label='±1σ Uncertainty')
    ax2.fill_between(timestamps, 
                     predicted - 2*uncertainty, 
                     predicted + 2*uncertainty,
                     alpha=0.15, color='#F18F01', label='±2σ Uncertainty')
    
    ax2.set_ylabel('Traffic (Mbps)', fontsize=12)
    ax2.set_title('Traffic Prediction with Uncertainty Bands', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Plot 3: Prediction Error and Uncertainty
    ax3 = axes[2]
    prediction_error = np.abs(df_plot['Traffic_Mbps_scaled'] - df_plot['Traffic_Predicted_Mbps'])
    
    ax3.plot(df_plot['timestamp'], prediction_error, 
             label='Absolute Prediction Error', color='#C73E1D', linewidth=1.0, alpha=0.7)
    ax3.plot(df_plot['timestamp'], df_plot['Traffic_Prediction_Uncertainty'], 
             label='Estimated Uncertainty (σ)', color='#F18F01', linewidth=1.5, linestyle='--')
    
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Error / Uncertainty (Mbps)', fontsize=12)
    ax3.set_title('Prediction Error vs Estimated Uncertainty', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotate x-axis labels
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def print_statistics(df):
    """Print statistics about predictions and uncertainty."""
    df_valid = df.dropna(subset=['Traffic_Predicted_Mbps'])
    
    actual = df_valid['Traffic_Mbps_scaled'].values
    predicted = df_valid['Traffic_Predicted_Mbps'].values
    uncertainty = df_valid['Traffic_Prediction_Uncertainty'].values
    
    errors = np.abs(actual - predicted)
    
    print("\n" + "="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    
    print("\n--- Error Metrics ---")
    print(f"Mean Absolute Error (MAE):     {errors.mean():>10.4f} Mbps")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt((errors**2).mean()):>10.4f} Mbps")
    print(f"Mean Absolute % Error (MAPE):   {(errors / (actual + 1e-8) * 100).mean():>10.2f} %")
    print(f"Max Absolute Error:             {errors.max():>10.4f} Mbps")
    
    print("\n--- Uncertainty Statistics ---")
    print(f"Mean Uncertainty (σ):           {uncertainty.mean():>10.4f} Mbps")
    print(f"Std Uncertainty:                {uncertainty.std():>10.4f} Mbps")
    print(f"Min Uncertainty:                {uncertainty.min():>10.4f} Mbps")
    print(f"Max Uncertainty:                {uncertainty.max():>10.4f} Mbps")
    
    # Calibration: how often does actual fall within uncertainty bands
    within_1sigma = np.sum(errors < uncertainty) / len(errors) * 100
    within_2sigma = np.sum(errors < 2 * uncertainty) / len(errors) * 100
    
    print("\n--- Uncertainty Calibration ---")
    print(f"% within 1σ band:               {within_1sigma:>10.2f} % (expect ~68%)")
    print(f"% within 2σ band:               {within_2sigma:>10.2f} % (expect ~95%)")
    
    print("\n--- Traffic Statistics ---")
    print(f"Mean Actual Traffic:            {actual.mean():>10.4f} Mbps")
    print(f"Mean Predicted Traffic:         {predicted.mean():>10.4f} Mbps")
    print(f"Max Actual Traffic:             {actual.max():>10.4f} Mbps")
    print(f"Max Predicted Traffic:          {predicted.max():>10.4f} Mbps")
    
    print("="*60 + "\n")

def main():
    """
    Main script to visualize traffic predictions and uncertainty.
    """
    parser = argparse.ArgumentParser(description="Visualize traffic predictions and uncertainty.")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date for visualization (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date for visualization (YYYY-MM-DD)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the figure (default: show interactively)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only print statistics, don't create plots")
    args = parser.parse_args()
    
    config = load_config()
    proc_config = config['traffic_preprocessing']
    processed_csv_path = proc_config['processed_csv_path']
    
    print(f"--- Traffic Prediction Visualization ---")
    print(f"Loading data from: {processed_csv_path}")
    
    # Load data
    try:
        df = pd.read_csv(processed_csv_path)
    except FileNotFoundError:
        print(f"\nERROR: File not found at '{processed_csv_path}'")
        print("Please run the prediction script first.")
        return
    
    # Check if predictions exist
    if 'Traffic_Predicted_Mbps' not in df.columns:
        print("\nERROR: No predictions found in the data.")
        print("Please run the prediction script first.")
        return
    
    # Print statistics
    print_statistics(df)
    
    # Create visualization
    if not args.stats_only:
        # Default save path if not provided
        save_path = args.save
        if save_path is None and not args.stats_only:
            save_path = "results/figures/traffic/traffic_predictions.png"
        
        plot_predictions(df, args.start_date, args.end_date, save_path)

if __name__ == '__main__':
    main()
