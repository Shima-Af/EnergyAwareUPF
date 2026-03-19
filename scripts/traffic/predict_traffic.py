# scripts/traffic/predict_traffic.py

import os
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

def load_config(path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def predict_with_uncertainty(model, X, scaler, actual_values=None, method='ensemble'):
    """
    Estimate prediction uncertainty using multiple methods.
    
    Args:
        model: Trained Keras model
        X: Input data for prediction
        scaler: Fitted MinMaxScaler
        actual_values: Actual scaled traffic values for residual-based uncertainty
        method: 'ensemble', 'residual', or 'percentile'
    
    Returns:
        predictions: Mean predictions
        uncertainty: Uncertainty estimates (standard deviation or residual-based)
    """
    if method == 'ensemble':
        # Monte Carlo Dropout (works even without dropout layers by using training=True)
        n_iterations = 30
        predictions = []
        
        for _ in range(n_iterations):
            # Use training=True to enable any stochastic behavior
            pred = model(X, training=True).numpy()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_prediction = predictions.mean(axis=0)
        std_prediction = predictions.std(axis=0)
        
        return mean_prediction, std_prediction
    
    elif method == 'residual':
        # Use prediction errors to estimate uncertainty
        predictions = model.predict(X, verbose=0)
        
        if actual_values is not None:
            # Calculate residuals on available data
            residuals = np.abs(actual_values[len(actual_values)-len(predictions):] - predictions.flatten())
            
            # Use a rolling window to estimate local uncertainty
            window_size = min(96, len(residuals))  # 24 hours or available data
            uncertainty = np.zeros_like(predictions.flatten())
            
            for i in range(len(predictions)):
                start_idx = max(0, i - window_size)
                end_idx = i + 1
                local_residuals = residuals[start_idx:end_idx]
                # Use 95th percentile of local residuals as uncertainty
                uncertainty[i] = np.percentile(local_residuals, 95) if len(local_residuals) > 0 else residuals.std()
            
            uncertainty = uncertainty.reshape(-1, 1)
        else:
            # No actual values available, use a constant uncertainty based on model variance
            uncertainty = np.ones_like(predictions) * 0.1  # 10% of range as default
        
        return predictions, uncertainty
    
    else:  # 'percentile' method
        # Use quantile regression approximation
        predictions = model.predict(X, verbose=0)
        
        # Estimate uncertainty as a function of prediction magnitude
        # Higher traffic typically has higher variance
        data_range = scaler.data_max_[0] - scaler.data_min_[0]
        
        # Uncertainty proportional to predicted value (heteroscedastic)
        uncertainty = predictions * 0.05 + 0.01  # 5% of prediction + small constant
        
        return predictions, uncertainty

def create_sequences(data, seq_len, calendar_data=None):
    """
    Create sequences for LSTM prediction.
    
    Args:
        data: Scaled traffic data
        seq_len: Sequence length
        calendar_data: Optional scaled calendar features
    
    Returns:
        X: Input sequences (with calendar features if provided)
        indices: Original indices for each prediction
    """
    X = []
    X_cal = []
    indices = []
    
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        indices.append(i)
        
        if calendar_data is not None:
            X_cal.append(calendar_data[i])
    
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Add calendar features if provided
    if calendar_data is not None:
        X_cal = np.array(X_cal)
        # Repeat calendar features for each timestep
        X_cal_repeated = np.repeat(X_cal[:, np.newaxis, :], seq_len, axis=1)
        # Concatenate with traffic data
        X = np.concatenate([X, X_cal_repeated], axis=2)
    
    return X, indices

def main():
    """
    Main script to predict traffic values and add predictions with uncertainty
    to the processed traffic dataset.
    """
    # 1. Setup: Load config and setup command-line arguments
    parser = argparse.ArgumentParser(
        description="Predict traffic and add uncertainty estimates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default input from config, save to default location
  python scripts/traffic/predict_traffic.py
  
  # Use custom input file
  python scripts/traffic/predict_traffic.py --input data/raw/processed_traffic_youtube_aggregated.csv
  
  # Use custom input and output
  python scripts/traffic/predict_traffic.py \\
    --input data/raw/traffic_data.csv \\
    --output data/processed/predictions_output.csv
  
  # Use different uncertainty method
  python scripts/traffic/predict_traffic.py --method ensemble
        """
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Input traffic CSV file (default: uses config.traffic_preprocessing.processed_csv_path)")
    parser.add_argument("--method", type=str, default='residual', 
                        choices=['ensemble', 'residual', 'percentile'],
                        help="Uncertainty estimation method (default: residual)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the CSV with predictions (default: overwrites input file)")
    args = parser.parse_args()
    
    config = load_config()
    train_config = config['traffic_predictor_training']
    paths_config = config['paths']
    proc_config = config['traffic_preprocessing']
    
    # Paths - use --input if provided, otherwise use config
    processed_csv_path = args.input if args.input else proc_config['processed_csv_path']
    output_path = args.output if args.output else processed_csv_path
    
    model_path = paths_config['traffic_predictor_model_path']
    scaler_path = model_path.replace('.keras', '_scaler.pkl')
    metadata_path = model_path.replace('.keras', '_metadata.pkl')
    
    print(f"--- Starting Traffic Prediction ---")
    print(f"Loading data from: {processed_csv_path}")
    print(f"Loading model from: {model_path}")
    
    # 2. Load the processed data
    try:
        df = pd.read_csv(processed_csv_path)
    except FileNotFoundError:
        print(f"\nERROR: Processed traffic file not found at '{processed_csv_path}'")
        print("Please run the preprocessing script first.")
        return
    
    # 3. Load the trained model, scaler, and metadata
    try:
        model = keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"\nERROR: Could not load model: {e}")
        print("Please train the traffic predictor first.")
        return
    
    try:
        with open(scaler_path, 'rb') as f:
            traffic_scaler = pickle.load(f)
        print(f"✓ Traffic scaler loaded successfully")
    except FileNotFoundError:
        print(f"\nERROR: Scaler file not found at '{scaler_path}'")
        print("Please train the traffic predictor first.")
        return
    
    # Load metadata to check if calendar features were used
    use_calendar = False
    calendar_cols = []
    calendar_scaler = None
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        use_calendar = metadata.get('use_calendar_features', False)
        calendar_cols = metadata.get('calendar_feature_columns', [])
        seq_len = metadata.get('sequence_length', train_config['sequence_length'])
        print(f"✓ Model metadata loaded")
        print(f"  - Calendar features: {'Enabled' if use_calendar else 'Disabled'}")
        if use_calendar:
            print(f"  - Calendar columns: {calendar_cols}")
    except FileNotFoundError:
        print(f"⚠️  Metadata file not found, assuming traffic-only mode")
        seq_len = train_config['sequence_length']
    
    # Load calendar scaler if calendar features are used
    if use_calendar:
        calendar_scaler_path = model_path.replace('.keras', '_calendar_scaler.pkl')
        try:
            with open(calendar_scaler_path, 'rb') as f:
                calendar_scaler = pickle.load(f)
            print(f"✓ Calendar scaler loaded successfully")
            
            # Check if calendar features exist in the dataframe
            missing_cols = [col for col in calendar_cols if col not in df.columns]
            if missing_cols:
                print(f"\n⚠️  WARNING: Calendar features missing in data: {missing_cols}")
                print("Please run: python scripts/traffic/add_calendar_features.py")
                print("Falling back to traffic-only mode.\n")
                use_calendar = False
        except FileNotFoundError:
            print(f"⚠️  Calendar scaler not found, disabling calendar features")
            use_calendar = False
    
    # 4. Prepare data for prediction
    traffic_data = df['Traffic_Mbps_scaled'].values.reshape(-1, 1)
    scaled_traffic = traffic_scaler.transform(traffic_data)
    
    # Prepare calendar features if enabled
    scaled_calendar = None
    if use_calendar and calendar_scaler is not None:
        calendar_data = df[calendar_cols].values
        scaled_calendar = calendar_scaler.transform(calendar_data)
        print(f"✓ Calendar features prepared: {scaled_calendar.shape}")
    
    X, indices = create_sequences(scaled_traffic, seq_len, scaled_calendar)
    
    print(f"\nPrepared {len(X)} sequences for prediction")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Input shape: {X.shape}")
    
    # 5. Make predictions with uncertainty estimation
    print(f"Predicting with uncertainty method: '{args.method}'...")
    
    # Get predictions and uncertainty
    mean_predictions, std_predictions = predict_with_uncertainty(
        model, X, traffic_scaler, scaled_traffic, method=args.method
    )
    
    # Inverse transform to get actual traffic values
    mean_predictions_rescaled = traffic_scaler.inverse_transform(mean_predictions)
    # For std, we scale it by the data range (approximation)
    data_range = traffic_scaler.data_max_[0] - traffic_scaler.data_min_[0]
    std_predictions_rescaled = std_predictions.flatten() * data_range
    
    # 6. Add predictions and uncertainty to dataframe
    # Initialize columns with NaN
    df['Traffic_Predicted_Mbps'] = np.nan
    df['Traffic_Prediction_Uncertainty'] = np.nan
    
    # Fill in predictions (starting from index seq_len)
    for i, idx in enumerate(indices):
        df.loc[idx, 'Traffic_Predicted_Mbps'] = mean_predictions_rescaled[i, 0]
        df.loc[idx, 'Traffic_Prediction_Uncertainty'] = std_predictions_rescaled[i]
    
    # 7. Calculate prediction metrics
    valid_mask = ~df['Traffic_Predicted_Mbps'].isna()
    if valid_mask.sum() > 0:
        actual = df.loc[valid_mask, 'Traffic_Mbps_scaled'].values
        predicted = df.loc[valid_mask, 'Traffic_Predicted_Mbps'].values
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
        
        print(f"\n--- Prediction Performance ---")
        print(f"MAE:  {mae:.4f} Mbps")
        print(f"RMSE: {rmse:.4f} Mbps")
        print(f"MAPE: {mape:.2f}%")
        print(f"Mean Uncertainty: {df['Traffic_Prediction_Uncertainty'].mean():.4f} Mbps")
        print(f"Max Uncertainty:  {df['Traffic_Prediction_Uncertainty'].max():.4f} Mbps")
    
    # 8. Save the updated dataframe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Successfully saved predictions to: {output_path}")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Rows with predictions: {valid_mask.sum()}")
    print(f"  - Rows without predictions (first {seq_len} timesteps): {(~valid_mask).sum()}")
    
    # 9. Display sample of results
    print(f"\n--- Sample of Results (first 10 rows with predictions) ---")
    sample_cols = ['timestamp', 'Traffic_Mbps_scaled', 'Traffic_Predicted_Mbps', 'Traffic_Prediction_Uncertainty']
    print(df[sample_cols].dropna().head(10).to_string(index=False))

if __name__ == '__main__':
    main()
