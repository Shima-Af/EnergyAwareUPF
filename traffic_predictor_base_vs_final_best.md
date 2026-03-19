# Traffic Predictor — Winner Only (Export)

Date: 2026-03-19

## Winner model

- Model ID: `delta_calendar_huber`
- Role: **Final best model for export/reporting**

## Winner architecture

`SpatialDropout1D(0.1) -> LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2) -> LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2) -> Dropout(0.2) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, linear)`

## Winner setup

- Calendar features: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`, `is_business_hours`
- Target mode: `delta` (residual)
- Loss: `Huber`
- Learning rate: `0.0005`
- Sequence length: `96`
- Temporal split: `train/val/test = 0.65 / 0.15 / 0.20`

## Winner accuracy (held-out test)

| Model ID | n_test | RMSE | MAE | R2 | MAPE_safe | sMAPE | WAPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `delta_calendar_huber` | 672 | 19.119999 | 13.835485 | 0.897953 | 17.424974 | 16.642336 | 13.752468 |
