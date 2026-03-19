#!/usr/bin/env python3
"""
Compute held-out traffic prediction accuracy for the saved traffic predictor.

Usage:
    python -m scripts.compute_traffic_prediction_accuracy --out-dir . --print-table

Optional:
    python -m scripts.compute_traffic_prediction_accuracy \
        --data-csv data/processed/processed_traffic_normalized.csv \
        --model-path saved_models/prediction_models/traffic_predictor.keras \
        --out-dir .
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.utils import load_config


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def format_architecture(architecture: List[Dict]) -> str:
    parts: List[str] = []
    for layer in architecture:
        layer_type = str(layer.get("type", "")).lower()
        if layer_type == "dense":
            units = layer.get("units", "?")
            activation = layer.get("activation", "linear")
            parts.append(f"Dense({units}, {activation})")
        elif layer_type == "dropout":
            parts.append(f"Dropout({layer.get('rate', '?')})")
        elif layer_type == "spatial_dropout1d":
            parts.append(f"SpatialDropout1D({layer.get('rate', '?')})")
        elif layer_type == "lstm":
            units = layer.get("units", "?")
            rs = layer.get("return_sequences", False)
            parts.append(f"LSTM({units}, return_sequences={rs})")
        else:
            parts.append(str(layer))
    return " → ".join(parts) if parts else "(not specified)"


def create_sequences(
    scaled_traffic: np.ndarray,
    seq_len: int,
    scaled_calendar: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[float] = []
    indices: List[int] = []
    X_cal: List[np.ndarray] = []

    for i in range(seq_len, len(scaled_traffic)):
        X.append(scaled_traffic[i - seq_len : i, 0])
        y.append(float(scaled_traffic[i, 0]))
        indices.append(i)
        if scaled_calendar is not None:
            X_cal.append(scaled_calendar[i])

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    idx_arr = np.array(indices, dtype=np.int32)

    X_arr = np.reshape(X_arr, (X_arr.shape[0], X_arr.shape[1], 1))

    if scaled_calendar is not None:
        X_cal_arr = np.array(X_cal, dtype=np.float32)
        X_cal_repeated = np.repeat(X_cal_arr[:, np.newaxis, :], seq_len, axis=1)
        X_arr = np.concatenate([X_arr, X_cal_repeated], axis=2)

    return X_arr, y_arr, idx_arr


def build_summary_markdown(
    accuracy_df: pd.DataFrame,
    config_path: Path,
    data_csv: Path,
    model_path: Path,
    scaler_path: Path,
    metadata_path: Path,
    calendar_scaler_path: Path,
    split_kwargs: Dict,
    seq_len: int,
    use_calendar_features: bool,
    calendar_feature_columns: List[str],
    traffic_architecture: List[Dict],
    traffic_cfg: Dict,
    model_layers_str: str,
    mape_eps: float,
    accuracy_csv_name: str,
    predictions_csv_name: str,
) -> str:
    split_name = (
        f"holdout_test_seq{seq_len}_testsize{split_kwargs['test_size']}_"
        f"rs{split_kwargs['random_state']}_shuffleFalse"
    )

    lines = [
        "# Traffic Prediction Accuracy (Digital Twin)",
        "",
        "## Files and pipeline used",
        "",
        f"- Configuration: `{config_path}`",
        "- Training/prediction logic basis: provided `train_traffic_predictor.py` and `predict_traffic.py` scripts",
        f"- Saved traffic predictor artifact: `{model_path}`",
        f"- Traffic scaler artifact: `{scaler_path}`",
        f"- Metadata artifact: `{metadata_path}`",
        f"- Calendar scaler artifact: `{calendar_scaler_path}`",
        f"- Traffic dataset: `{data_csv}`",
        "",
        "## Training architecture (from scripts + config + saved artifact)",
        "",
        f"- Configured traffic predictor architecture: `{format_architecture(traffic_architecture)}`",
        f"- Saved model layer stack: `{model_layers_str}`",
        f"- Sequence length used: `{seq_len}`",
        f"- Calendar features used: `{use_calendar_features}`",
    ]

    if calendar_feature_columns:
        lines.append(f"- Calendar feature columns: `{calendar_feature_columns}`")

    lines.extend(
        [
            "",
            "## Held-out split and preprocessing used",
            "",
            "- Sequence construction follows training script: uses past `seq_len` timesteps to predict next timestep.",
            f"- Split rule: `train_test_split(test_size={split_kwargs['test_size']}, random_state={split_kwargs['random_state']}, shuffle=False)`",
            "- Input preprocessing: `MinMaxScaler` transform on full traffic series before sequence construction.",
            "- Target space during training/evaluation: `Traffic_Mbps_scaled`.",
            "- Inverse-transform step undoes only `MinMaxScaler` normalization and returns values in `Traffic_Mbps_scaled` units.",
            "- This does **not** undo upstream traffic-load scaling performed during data preprocessing.",
            f"- Evaluation split name: `{split_name}`",
            "",
            "## Training hyperparameters (config)",
            "",
            f"- learning_rate={traffic_cfg.get('learning_rate')}",
            f"- epochs={traffic_cfg.get('epochs')}",
            f"- batch_size={traffic_cfg.get('batch_size')}",
            f"- test_size={traffic_cfg.get('test_size')}",
            f"- random_state={traffic_cfg.get('random_state')}",
            "- validation_data uses the exact held-out split (per training script).",
            "",
            "## MAPE handling",
            "",
            f"- Safe MAPE uses epsilon `{mape_eps}` in `Traffic_Mbps_scaled` units:",
            f"  - `MAPE_safe = mean(|y_true - y_pred| / max(|y_true|, {mape_eps})) * 100`",
            "",
            "## Accuracy table",
            "",
            "| model_name | target | split_name | n_samples | rmse | mape | r2 | mae |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    table_df = accuracy_df.copy().reset_index(drop=True)
    for _, row in table_df.iterrows():
        lines.append(
            "| "
            f"{row['model_name']} | {row['target']} | {row['split_name']} | {int(row['n_samples'])} | "
            f"{row['rmse']:.6f} | {row['mape']:.6f} | {row['r2']:.6f} | {row['mae']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Traceability artifacts",
            "",
            f"- Summary metrics CSV: `{accuracy_csv_name}`",
            f"- Per-sample predictions CSV: `{predictions_csv_name}`",
            "",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute held-out accuracy for the traffic predictor model on the training-script split."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--data-csv",
        default=None,
        help="Override traffic CSV path (default: config.traffic_preprocessing.processed_csv_path)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override traffic predictor model path (default: config.paths.traffic_predictor_model_path)",
    )
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--mape-eps", type=float, default=1e-3, help="Safe MAPE epsilon in Mbps")
    parser.add_argument("--print-table", action="store_true", help="Print final table to stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    config = load_config(str(config_path))

    traffic_cfg = config["traffic_predictor_training"]
    proc_cfg = config["traffic_preprocessing"]
    paths_cfg = config["paths"]

    data_csv = Path(args.data_csv) if args.data_csv else Path(proc_cfg["processed_csv_path"])
    model_path = (
        Path(args.model_path) if args.model_path else Path(paths_cfg["traffic_predictor_model_path"])
    )
    if not data_csv.is_absolute():
        data_csv = (project_root / data_csv).resolve()
    if not model_path.is_absolute():
        model_path = (project_root / model_path).resolve()

    scaler_path = Path(str(model_path).replace(".keras", "_scaler.pkl"))
    metadata_path = Path(str(model_path).replace(".keras", "_metadata.pkl"))
    calendar_scaler_path = Path(str(model_path).replace(".keras", "_calendar_scaler.pkl"))

    df = pd.read_csv(data_csv)

    if "Traffic_Mbps_scaled" not in df.columns:
        raise KeyError(f"Traffic_Mbps_scaled missing in {data_csv}")

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    with open(scaler_path, "rb") as f:
        traffic_scaler = pickle.load(f)

    use_calendar = bool(metadata.get("use_calendar_features", False))
    calendar_cols = list(metadata.get("calendar_feature_columns", []))
    seq_len = int(metadata.get("sequence_length", traffic_cfg.get("sequence_length", 96)))

    scaled_traffic = traffic_scaler.transform(df[["Traffic_Mbps_scaled"]].to_numpy(dtype=np.float32))

    scaled_calendar = None
    if use_calendar:
        if not calendar_scaler_path.exists():
            raise FileNotFoundError(
                f"Calendar features enabled in metadata but scaler not found: {calendar_scaler_path}"
            )
        missing_cols = [c for c in calendar_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Calendar columns missing in dataset: {missing_cols}")
        with open(calendar_scaler_path, "rb") as f:
            calendar_scaler = pickle.load(f)
        scaled_calendar = calendar_scaler.transform(df[calendar_cols].to_numpy(dtype=np.float32))

    X, y_scaled, indices = create_sequences(
        scaled_traffic=scaled_traffic,
        seq_len=seq_len,
        scaled_calendar=scaled_calendar,
    )

    split_kwargs = {
        "test_size": float(traffic_cfg["test_size"]),
        "random_state": int(traffic_cfg["random_state"]),
        "shuffle": False,
    }

    (
        _,
        X_test,
        _,
        y_test_scaled,
        _,
        test_indices,
    ) = train_test_split(X, y_scaled, indices, **split_kwargs)

    model = keras.models.load_model(model_path)
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    y_true = traffic_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    y_pred = traffic_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    split_name = (
        f"holdout_test_seq{seq_len}_testsize{split_kwargs['test_size']}_"
        f"rs{split_kwargs['random_state']}_shuffleFalse"
    )

    accuracy_df = pd.DataFrame(
        [
            {
                "model_name": "traffic_predictor.keras",
                "target": "Traffic_Mbps_scaled (MinMax inverse only; load-scaling preserved)",
                "split_name": split_name,
                "n_samples": int(len(y_true)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mape": safe_mape(y_true, y_pred, eps=float(args.mape_eps)),
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
            }
        ]
    )

    pred_df = pd.DataFrame(
        {
            "model_name": "traffic_predictor.keras",
            "target": "Traffic_Mbps_scaled (MinMax inverse only; load-scaling preserved)",
            "split_name": split_name,
            "sample_idx": np.arange(len(y_true), dtype=int),
            "time_index": test_indices.astype(int),
            "timestamp": df.loc[test_indices, "timestamp"].astype(str).to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": np.abs(y_true - y_pred),
            "ape_safe_pct": np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), float(args.mape_eps)) * 100.0,
        }
    )

    model_layers_str = " → ".join(
        [
            layer.__class__.__name__
            + (
                f"({layer.units})"
                if hasattr(layer, "units")
                else f"({getattr(layer, 'rate', '')})" if hasattr(layer, "rate") else ""
            )
            for layer in model.layers
        ]
    )

    summary_text = build_summary_markdown(
        accuracy_df=accuracy_df,
        config_path=config_path,
        data_csv=data_csv,
        model_path=model_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        calendar_scaler_path=calendar_scaler_path,
        split_kwargs=split_kwargs,
        seq_len=seq_len,
        use_calendar_features=use_calendar,
        calendar_feature_columns=calendar_cols,
        traffic_architecture=traffic_cfg.get("architecture", []),
        traffic_cfg=traffic_cfg,
        model_layers_str=model_layers_str,
        mape_eps=float(args.mape_eps),
        accuracy_csv_name="traffic_prediction_accuracy.csv",
        predictions_csv_name="traffic_prediction_predictions_vs_target.csv",
    )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    accuracy_csv = out_dir / "traffic_prediction_accuracy.csv"
    summary_md = out_dir / "traffic_prediction_accuracy_summary.md"
    preds_csv = out_dir / "traffic_prediction_predictions_vs_target.csv"

    accuracy_df.to_csv(accuracy_csv, index=False)
    pred_df.to_csv(preds_csv, index=False)
    summary_md.write_text(summary_text, encoding="utf-8")

    print(f"Wrote {accuracy_csv}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {preds_csv}")

    if args.print_table:
        print()
        print(accuracy_df.to_string(index=False))


if __name__ == "__main__":
    main()
