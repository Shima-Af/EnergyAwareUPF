# scripts/train_traffic_predictor.py

import os
import yaml
import copy
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras  # pylint: disable=no-name-in-module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_gpu():
    """Check if GPU is available and print device info."""
    print("\n" + "="*60)
    print("GPU CONFIGURATION")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPUs Available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            # Get GPU memory info if possible
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details:
                    print(f"    Details: {gpu_details}")
            except:
                pass
    else:
        print("⚠ WARNING: No GPU detected! Training will be MUCH slower on CPU.")
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print("="*60 + "\n")
    
    return len(gpus) > 0

def create_model_from_config(config, input_shape):
    model = keras.Sequential()
    arch = copy.deepcopy(config['architecture'])

    for i, layer_cfg in enumerate(arch):
        layer_type = layer_cfg.pop('type').lower()

        # Add input_shape only to the first layer that accepts it (LSTM, Dense)
        if i == 0 and layer_type in ['lstm', 'dense']:
            layer_cfg['input_shape'] = input_shape

        if layer_type == 'lstm':
            # Keras LSTM accepts dropout/recurrent_dropout; they only act when training=True
            model.add(keras.layers.LSTM(**layer_cfg))

        elif layer_type == 'dense':
            model.add(keras.layers.Dense(**layer_cfg))

        elif layer_type == 'dropout':
            model.add(keras.layers.Dropout(**layer_cfg))

        elif layer_type == 'spatial_dropout1d':
            model.add(keras.layers.SpatialDropout1D(**layer_cfg))

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='mse'
    )
    return model

def main():
    print("--- Training Traffic Predictor Model ---")
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    config = load_config()
    train_config = config['traffic_predictor_training']
    paths_config = config['paths']
    
    # 1. Load and prepare data
    print("Loading data...")
    df = pd.read_csv(config['traffic_preprocessing']['processed_csv_path'])
    
    # Check if using calendar features
    use_calendar = train_config.get('use_calendar_features', False)
    calendar_cols = train_config.get('calendar_feature_columns', [])
    
    if use_calendar:
        # Check if calendar features exist in the dataframe
        missing_cols = [col for col in calendar_cols if col not in df.columns]
        if missing_cols:
            print(f"\n⚠️  WARNING: Calendar features enabled but missing columns: {missing_cols}")
            print("Please run: python scripts/traffic/add_calendar_features.py")
            print("Falling back to traffic-only mode.\n")
            use_calendar = False
        else:
            print(f"✓ Using calendar features: {calendar_cols}")
    else:
        print("Using traffic data only (calendar features disabled)")
    
    # Prepare traffic data
    traffic_data = df['Traffic_Mbps_scaled'].values.reshape(-1, 1)

    # Normalize the traffic data between 0 and 1 for the LSTM
    print("Normalizing data...")
    from sklearn.preprocessing import MinMaxScaler
    traffic_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_traffic = traffic_scaler.fit_transform(traffic_data)
    
    # Prepare calendar features if enabled
    if use_calendar:
        calendar_data = df[calendar_cols].values
        calendar_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_calendar = calendar_scaler.fit_transform(calendar_data)
        print(f"Calendar features shape: {scaled_calendar.shape}")
    else:
        scaled_calendar = None
        calendar_scaler = None

    # 2. Create sequences
    seq_len = train_config['sequence_length']
    print(f"Creating sequences with length {seq_len}...")
    X, y, X_cal = [], [], []
    
    for i in range(seq_len, len(scaled_traffic)):
        # Traffic sequences
        X.append(scaled_traffic[i-seq_len:i, 0])
        y.append(scaled_traffic[i, 0])
        
        # Calendar features for current timestep (not sequences)
        if use_calendar:
            X_cal.append(scaled_calendar[i])
    
    X, y = np.array(X), np.array(y)
    
    if use_calendar:
        X_cal = np.array(X_cal)
        # Reshape X to be [samples, timesteps, 1] for traffic
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        # Concatenate calendar features to each timestep
        # Repeat calendar features for each timestep in the sequence
        X_cal_repeated = np.repeat(X_cal[:, np.newaxis, :], seq_len, axis=1)
        # Combine: [samples, timesteps, 1+num_calendar_features]
        X = np.concatenate([X, X_cal_repeated], axis=2)
        num_features = 1 + len(calendar_cols)
        print(f"Combined features: 1 traffic + {len(calendar_cols)} calendar = {num_features} total")
    else:
        # Reshape X to be [samples, timesteps, features] for LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        num_features = 1
    
    print(f"Sequence shape: {X.shape}")
    print(f"Memory usage: ~{X.nbytes / 1024 / 1024:.2f} MB")

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_config['test_size'], random_state=train_config['random_state'], shuffle=False
    )
    print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples.")

    # 4. Create and train model
    print("\nBuilding model...")
    model = create_model_from_config(train_config, input_shape=(X_train.shape[1], num_features))
    model.summary()

    # Optimize batch size for GPU
    batch_size = train_config['batch_size']
    if has_gpu and batch_size < 64:
        print(f"\n💡 GPU detected! Increasing batch size from {batch_size} to 64 for better GPU utilization.")
        batch_size = 64
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {train_config['learning_rate']}")
    print(f"  Sequence length: {seq_len}")

    # Callbacks for better training
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20,  # Increased patience for larger model
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by half
        patience=10,  # Wait 10 epochs before reducing
        min_lr=1e-6,
        verbose=1
    )
    
    # Progress bar callback
    print("\nStarting training...")
    print("="*60)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=train_config['epochs'],
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # 5. Save the trained model and scaler(s)
    model_path = paths_config['traffic_predictor_model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\n✓ Traffic predictor model saved successfully to: {model_path}")
    
    # Save the traffic scaler
    scaler_path = model_path.replace('.keras', '_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(traffic_scaler, f)
    print(f"✓ Traffic predictor scaler saved successfully to: {scaler_path}")
    
    # Save calendar scaler if used
    if use_calendar and calendar_scaler is not None:
        calendar_scaler_path = model_path.replace('.keras', '_calendar_scaler.pkl')
        with open(calendar_scaler_path, 'wb') as f:
            pickle.dump(calendar_scaler, f)
        print(f"✓ Calendar features scaler saved successfully to: {calendar_scaler_path}")
    
    # Save config metadata
    metadata_path = model_path.replace('.keras', '_metadata.pkl')
    metadata = {
        'use_calendar_features': use_calendar,
        'calendar_feature_columns': calendar_cols if use_calendar else [],
        'sequence_length': seq_len,
        'num_features': num_features
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Model metadata saved successfully to: {metadata_path}")

if __name__ == '__main__':
    main()