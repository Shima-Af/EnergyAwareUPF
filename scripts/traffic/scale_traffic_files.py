#!/usr/bin/env python3
"""
Scale traffic files using the same ratio or minmax scaling method.

This script applies traffic scaling (same as preprocess_traffic.py) to given CSV files.
It calculates a scaling factor or minmax normalization and applies it to the specified
traffic column.

Usage:
    python scale_traffic_files.py --input_files file1.csv file2.csv \
        --traffic_col Traffic_bps --target_peak_mbps 400 \
        --scaling_method ratio --output_dir ./scaled_output
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def scale_traffic_ratio(df, traffic_col, target_peak_bps, fixed_scale_factor=None):
    """
    Scale traffic using ratio method: scale_factor = target_peak_bps / current_max
    
    Args:
        df: DataFrame containing traffic data
        traffic_col: Column name with traffic values in bps
        target_peak_bps: Target peak value in bits per second
        fixed_scale_factor: If provided, use this exact factor instead of calculating from file
        
    Returns:
        Tuple of (scaled_df, scale_factor, current_max)
    """
    current_max = df[traffic_col].max()
    
    if current_max <= 0:
        raise ValueError("Original maximum traffic is zero or negative. Cannot scale.")
    
    if fixed_scale_factor is not None:
        scale_factor = fixed_scale_factor
        print(f"  Using fixed scale factor: {scale_factor:.6f}")
    else:
        scale_factor = target_peak_bps / current_max
    
    scaled_df = df.copy()
    scaled_df[f'{traffic_col}_scaled'] = df[traffic_col] * scale_factor
    
    return scaled_df, scale_factor, current_max


def scale_traffic_minmax(df, traffic_col, target_peak_bps):
    """
    Scale traffic using minmax normalization: (x - min) / (max - min) * target_peak
    
    Args:
        df: DataFrame containing traffic data
        traffic_col: Column name with traffic values in bps
        target_peak_bps: Target peak value in bits per second
        
    Returns:
        Tuple of (scaled_df, scale_params, current_min, current_max)
    """
    current_min = df[traffic_col].min()
    current_max = df[traffic_col].max()
    
    if current_max <= current_min:
        raise ValueError("Invalid traffic range for minmax scaling.")
    
    scaled_df = df.copy()
    scaled_df[f'{traffic_col}_scaled'] = (
        (df[traffic_col] - current_min) / (current_max - current_min) * target_peak_bps
    )
    
    return scaled_df, {'min': current_min, 'max': current_max}, current_min, current_max


def process_file(input_path, traffic_col, target_peak_mbps, scaling_method, output_dir=None, fixed_scale_factor=None):
    """
    Load, scale, and save a traffic file.
    
    Args:
        input_path: Path to input CSV file
        traffic_col: Column name with traffic values in bps
        target_peak_mbps: Target peak throughput in Mbps (can be None if using fixed_scale_factor)
        scaling_method: "ratio" or "minmax"
        output_dir: Directory for output file (uses input dir if None)
        fixed_scale_factor: For ratio method, use this exact factor instead of calculating from file
        
    Returns:
        Dictionary with scaling statistics
    """
    target_peak_bps = target_peak_mbps * 1e6 if target_peak_mbps is not None else None
    
    # Load file
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    except Exception as e:
        raise Exception(f"Error reading {input_path}: {e}")
    
    # Check if traffic column exists
    if traffic_col not in df.columns:
        raise ValueError(f"Traffic column '{traffic_col}' not found in {input_path}")
    
    # Apply scaling
    if scaling_method == 'ratio':
        scaled_df, scale_factor, current_max = scale_traffic_ratio(df, traffic_col, target_peak_bps, fixed_scale_factor)
        stats = {
            'file': input_path,
            'method': 'ratio',
            'current_max_bps': current_max,
            'current_max_mbps': current_max / 1e6,
            'target_peak_bps': target_peak_bps,
            'target_peak_mbps': target_peak_mbps,
            'scale_factor': scale_factor,
        }
    elif scaling_method == 'minmax':
        scaled_df, scale_params, current_min, current_max = scale_traffic_minmax(
            df, traffic_col, target_peak_bps
        )
        stats = {
            'file': input_path,
            'method': 'minmax',
            'current_min_bps': current_min,
            'current_min_mbps': current_min / 1e6,
            'current_max_bps': current_max,
            'current_max_mbps': current_max / 1e6,
            'target_peak_bps': target_peak_bps,
            'target_peak_mbps': target_peak_mbps,
        }
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    # Add Mbps scaled column if it exists in original
    if f'{traffic_col}_scaled' in scaled_df.columns:
        scaled_df['Traffic_Mbps_scaled'] = scaled_df[f'{traffic_col}_scaled'] / 1e6
    
    # Determine output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(input_path).stem
        output_path = os.path.join(output_dir, f'{base_name}_scaled.csv')
    else:
        base_name = Path(input_path).stem
        parent_dir = Path(input_path).parent
        output_path = os.path.join(parent_dir, f'{base_name}_scaled.csv')
    
    # Save scaled file
    scaled_df.to_csv(output_path, index=False)
    stats['output_file'] = output_path
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Scale traffic files using ratio or minmax scaling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scale multiple files with FIXED ratio (0.1936 from original run)
  python scripts/traffic/scale_traffic_files.py --input_files data/raw/processed_traffic_youtube_aggregated.csv \
    --traffic_col Traffic_bps --fixed_scale_factor 0.1936 --output_dir data/processed
  
  # Scale single file, calculate ratio from target peak
  python scripts/traffic/scale_traffic_files.py --input_files processed_traffic_youtube_aggregated.csv \\
    --traffic_col Traffic_bps --target_peak_mbps 400 --scaling_method ratio
  
  # Scale multiple files to output directory (each gets own ratio)
  python scripts/traffic/scale_traffic_files.py --input_files file1.csv file2.csv file3.csv \\
    --traffic_col Traffic_bps --target_peak_mbps 500 --output_dir ./scaled_output
        """
    )
    
    parser.add_argument(
        '--input_files',
        nargs='+',
        required=True,
        help='Input CSV file(s) to scale'
    )
    parser.add_argument(
        '--traffic_col',
        default='Traffic_bps',
        help='Column name containing traffic values in bits per second (default: Traffic_bps)'
    )
    parser.add_argument(
        '--target_peak_mbps',
        type=float,
        default=None,
        help='Target peak throughput in Mbps (required unless using --fixed_scale_factor)'
    )
    parser.add_argument(
        '--scaling_method',
        choices=['ratio', 'minmax'],
        default='ratio',
        help='Scaling method to use (default: ratio)'
    )
    parser.add_argument(
        '--fixed_scale_factor',
        type=float,
        default=None,
        help='(Ratio method only) Use this exact scale factor for all files instead of calculating from each file. Example: 0.1936'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Output directory for scaled files (uses input dir if not specified)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed scaling information'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.fixed_scale_factor is None and args.target_peak_mbps is None:
        parser.error("Either --target_peak_mbps or --fixed_scale_factor must be provided")
    
    print("=" * 70)
    print("Traffic File Scaling")
    print("=" * 70)
    print(f"Scaling Method: {args.scaling_method}")
    if args.scaling_method == 'ratio' and args.fixed_scale_factor:
        print(f"Scale Factor: {args.fixed_scale_factor} (FIXED)")
    else:
        print(f"Target Peak: {args.target_peak_mbps} Mbps")
    print(f"Traffic Column: {args.traffic_col}")
    if args.output_dir:
        print(f"Output Directory: {args.output_dir}")
    print()
    
    all_stats = []
    for input_file in args.input_files:
        try:
            stats = process_file(
                input_file,
                args.traffic_col,
                args.target_peak_mbps,
                args.scaling_method,
                args.output_dir,
                args.fixed_scale_factor
            )
            all_stats.append(stats)
            
            print(f"✓ Processed: {input_file}")
            if args.verbose:
                if args.scaling_method == 'ratio':
                    print(f"  Scale Factor: {stats['scale_factor']:.6f}")
                    print(f"  Original Peak: {stats['current_max_mbps']:.2f} Mbps")
                else:
                    print(f"  Original Range: {stats['current_min_mbps']:.2f} - {stats['current_max_mbps']:.2f} Mbps")
                print(f"  Target Peak: {stats['target_peak_mbps']:.2f} Mbps")
                print(f"  Output: {stats['output_file']}")
            print()
        
        except Exception as e:
            print(f"✗ Error processing {input_file}: {e}")
            print()
    
    # Summary
    print("=" * 70)
    print(f"Summary: Successfully scaled {len(all_stats)} file(s)")
    if all_stats:
        print("\nOutput files:")
        for stats in all_stats:
            print(f"  - {stats['output_file']}")


if __name__ == '__main__':
    main()
