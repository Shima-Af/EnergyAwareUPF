# Traffic Prediction Scripts

This directory contains scripts for preprocessing traffic data, training prediction models, making predictions with uncertainty estimates, and visualizing the results.

## Scripts Overview

## ===================================================================
## 4. CALENDAR FEATURES (OPTIONAL)
## ===================================================================

### What Are Calendar Features?

Calendar features are time-based attributes that help the model understand temporal patterns:
- **Hour of day**: Traffic is different at 3 AM vs 9 AM
- **Day of week**: Weekdays vs weekends have different patterns
- **Business hours**: Work hours vs off-hours
- **Peak times**: Morning/evening rush hours

### When to Use Calendar Features?

**Use calendar features when:**
- ✅ Your traffic has strong daily/weekly patterns
- ✅ You want the model to learn "9 AM Mondays are busy"
- ✅ You have sparse data and need explicit time information

**Don't use calendar features when:**
- ❌ Using autoregressive models (LSTM with long sequences)
- ❌ Past traffic already contains time patterns implicitly
- ❌ Dataset is small (< 1 month of data)

**Note**: For LSTM models with 96-timestep sequences (24 hours), calendar features typically provide **minimal improvement** (1-3%) because the traffic history already encodes time patterns.

### How to Enable Calendar Features

**Step 1: Add calendar features to your dataset**
```bash
python scripts/traffic/add_calendar_features.py
```

This adds these columns to your CSV:
- `hour`, `day_of_week`, `day_of_month`, `month`, `quarter`
- `hour_sin`, `hour_cos` (cyclical encoding for hour)
- `dow_sin`, `dow_cos` (cyclical encoding for day of week)
- `is_weekend`, `is_business_hours`, `is_night`
- `is_morning_peak`, `is_evening_peak`

**Step 2: Enable in config**
```yaml
# In config.yaml
traffic_predictor_training:
  use_calendar_features: true  # Change from false to true
  
  calendar_feature_columns:
    - "hour_sin"
    - "hour_cos"
    - "dow_sin"
    - "dow_cos"
    - "is_weekend"
    - "is_business_hours"
```

**Step 3: Retrain the model**
```bash
python scripts/traffic/train_traffic_predictor.py
```

**Step 4: Generate predictions** (automatically uses calendar features)
```bash
python scripts/traffic/predict_traffic.py --method ensemble
```

### Script: `add_calendar_features.py`

Adds temporal features to the traffic dataset.

**Usage:**
```bash
python scripts/traffic/add_calendar_features.py
```

**What it does:**
- Reads processed traffic CSV
- Extracts time components from timestamp
- Adds cyclical encodings (sin/cos) for periodic features
- Adds binary flags (weekend, business hours, etc.)
- Saves updated CSV with calendar features

**Output**: Updated `processed_traffic_normalized.csv` with 14 additional columns

---

## ===================================================================
## 1. SCRIPTS OVERVIEW (Updated)
## ===================================================================

### 1. `preprocess_traffic.py`
Preprocesses raw traffic data by scaling it to a target peak throughput.

**Usage:**
```bash
python scripts/traffic/preprocess_traffic.py [--plot]
```

**Options:**
- `--plot`: Display a plot of original vs. scaled traffic

**What it does:**
- Loads raw traffic data from CSV
- Scales traffic to target peak (configurable in `config.yaml`)
- Saves processed data with scaled traffic in Mbps

**Configuration (config.yaml):**
```yaml
traffic_preprocessing:
  raw_csv_path: "data/raw/processed_traffic_sum_pivoted.csv"
  processed_csv_path: "data/processed/processed_traffic_normalized.csv"
  scaling_method: "ratio"  # or "minmax"
  target_peak_mbps: 400.0
```

---

### 2. `train_traffic_predictor.py`
Trains an LSTM-based traffic prediction model.

**Usage:**
```bash
python scripts/traffic/train_traffic_predictor.py
```

**What it does:**
- Loads processed traffic data
- Creates sequences for LSTM training
- Trains the model with early stopping
- Saves the trained model and scaler

**Configuration (config.yaml):**
```yaml
traffic_predictor_training:
  sequence_length: 24        # Number of timesteps to look back
  test_size: 0.2            # Train/test split ratio
  random_state: 42          # Random seed
  epochs: 100               # Max epochs
  batch_size: 32
  learning_rate: 0.001
  architecture:
    - type: "lstm"
      units: 64
      activation: "tanh"
      return_sequences: true
    - type: "lstm"
      units: 32
      activation: "tanh"
    - type: "dense"
      units: 1
```

**Outputs:**
- `saved_models/prediction_models/traffic_predictor.keras`
- `saved_models/prediction_models/traffic_predictor_scaler.pkl`

---

### 3. `predict_traffic.py` ⭐ NEW
Generates traffic predictions with uncertainty estimates and adds them to the dataset.

**Usage:**
```bash
python scripts/traffic/predict_traffic.py [options]
```

**Options:**
- `--method {ensemble,residual,percentile}`: Uncertainty estimation method (default: residual)
  - **residual**: Uses historical prediction errors (recommended)
  - **ensemble**: Monte Carlo dropout (requires model with dropout layers)
  - **percentile**: Proportional to prediction magnitude
- `--output PATH`: Output CSV path (default: overwrites processed file)

**Examples:**
```bash
# Use residual-based uncertainty (recommended)
python scripts/traffic/predict_traffic.py --method residual

# Save to a different file
python scripts/traffic/predict_traffic.py --output data/processed/traffic_with_predictions.csv

# Try ensemble method
python scripts/traffic/predict_traffic.py --method ensemble
```

**What it does:**
1. Loads the trained traffic predictor model and scaler
2. Creates sequences from processed traffic data
3. Makes predictions for each timestep
4. Estimates uncertainty using the selected method
5. Adds two new columns to the dataset:
   - `Traffic_Predicted_Mbps`: Predicted traffic value
   - `Traffic_Prediction_Uncertainty`: Uncertainty estimate (standard deviation)
6. Saves the updated dataset

**Output Columns:**
- `timestamp`: Timestamp of the data point
- `Traffic_bps`: Original traffic in bps
- `Traffic_bps_scaled`: Scaled traffic in bps
- `Traffic_Mbps_scaled`: Scaled traffic in Mbps (actual value)
- `Traffic_Predicted_Mbps`: **Predicted traffic in Mbps** ⭐ NEW
- `Traffic_Prediction_Uncertainty`: **Uncertainty estimate (σ)** ⭐ NEW

**Note:** The first `sequence_length` rows (default: 24) will have NaN predictions since there's insufficient history.

---

### 4. `visualize_predictions.py` ⭐ NEW
Visualizes traffic predictions and uncertainty estimates.

**Usage:**
```bash
python scripts/traffic/visualize_predictions.py [options]
```

**Options:**
- `--start-date YYYY-MM-DD`: Start date for visualization
- `--end-date YYYY-MM-DD`: End date for visualization
- `--save PATH`: Save figure to path (default: results/figures/traffic_predictions.png)
- `--stats-only`: Only print statistics, don't create plots

**Examples:**
```bash
# Visualize full dataset
python scripts/traffic/visualize_predictions.py

# Visualize specific date range
python scripts/traffic/visualize_predictions.py --start-date 2019-04-01 --end-date 2019-04-14

# Save to custom location
python scripts/traffic/visualize_predictions.py --save my_predictions.png

# Only print statistics
python scripts/traffic/visualize_predictions.py --stats-only
```

**What it creates:**
1. **Plot 1**: Actual vs Predicted Traffic
   - Line plot comparing actual and predicted traffic
2. **Plot 2**: Predictions with Uncertainty Bands
   - Shows ±1σ and ±2σ uncertainty bands around predictions
3. **Plot 3**: Prediction Error vs Uncertainty
   - Compares actual prediction errors with estimated uncertainty

**Statistics Printed:**
- Error metrics (MAE, RMSE, MAPE)
- Uncertainty statistics (mean, std, min, max)
- Calibration (% of predictions within uncertainty bands)
- Traffic statistics

---

## Typical Workflow

1. **Preprocess raw traffic data:**
   ```bash
   python scripts/traffic/preprocess_traffic.py
   ```

2. **Train the traffic predictor:**
   ```bash
   python scripts/traffic/train_traffic_predictor.py
   ```

3. **Generate predictions with uncertainty:**
   ```bash
   python scripts/traffic/predict_traffic.py --method residual
   ```

4. **Visualize results:**
   ```bash
   python scripts/traffic/visualize_predictions.py
   ```

---

## Understanding Uncertainty Methods

### Residual Method (Recommended)
- Uses historical prediction errors to estimate uncertainty
- Calculates a rolling window of past errors
- Uses 95th percentile of local residuals as uncertainty
- **Pros**: Calibrated to actual model performance, data-driven
- **Cons**: Requires sufficient historical data

### Ensemble Method
- Uses Monte Carlo Dropout for stochastic predictions
- Makes multiple forward passes with dropout enabled
- Standard deviation across passes = uncertainty
- **Pros**: Model-based uncertainty, captures epistemic uncertainty
- **Cons**: Requires model with dropout layers, computationally expensive

### Percentile Method
- Assumes uncertainty proportional to prediction magnitude
- Higher traffic → higher uncertainty (heteroscedastic)
- **Pros**: Fast, no historical data needed
- **Cons**: Less calibrated, assumes specific error structure

---

## Key Files

### Input Files
- `data/raw/processed_traffic_sum_pivoted.csv`: Raw traffic data
- `config.yaml`: Configuration for all scripts

### Intermediate Files
- `data/processed/processed_traffic_normalized.csv`: Preprocessed traffic with scaled values

### Model Files
- `saved_models/prediction_models/traffic_predictor.keras`: Trained LSTM model
- `saved_models/prediction_models/traffic_predictor_scaler.pkl`: MinMax scaler

### Output Files
- `data/processed/processed_traffic_normalized.csv`: Updated with predictions and uncertainty
- `results/figures/traffic_predictions.png`: Visualization of predictions

---

## Performance Metrics

Based on the current model (as of Nov 10, 2025):

- **MAE**: 14.79 Mbps
- **RMSE**: 19.98 Mbps
- **MAPE**: 14.62%
- **Mean Uncertainty**: 198.67 Mbps (residual method)
- **Calibration**: 100% within 1σ band (well-calibrated)

---

## Troubleshooting

### "Model file not found"
Run the training script first:
```bash
python scripts/traffic/train_traffic_predictor.py
```

### "Processed traffic file not found"
Run the preprocessing script first:
```bash
python scripts/traffic/preprocess_traffic.py
```

### Uncertainty values are too low/high
- Try different uncertainty methods: `--method residual`, `--method ensemble`, `--method percentile`
- For residual method, check if you have sufficient historical data
- Adjust the model architecture or training parameters in `config.yaml`

### Predictions are inaccurate
- Check if the model was trained properly
- Increase `sequence_length` for more context
- Increase model capacity (more LSTM units or layers)
- Collect more training data
- Try different preprocessing scaling methods

---

## Future Enhancements

Possible improvements:
- [ ] Add confidence intervals (e.g., 95% prediction intervals)
- [ ] Support for multi-step ahead predictions
- [ ] Online learning/model updates
- [ ] Ensemble of multiple models
- [ ] Attention mechanisms for interpretability
- [ ] Exogenous features (time of day, day of week, etc.)

---

## References

- LSTM Networks: Hochreiter & Schmidhuber (1997)
- Uncertainty Quantification: Gal & Ghahramani (2016) - "Dropout as a Bayesian Approximation"
- Time Series Forecasting: Makridakis et al. (2020) - "M4 Competition"
