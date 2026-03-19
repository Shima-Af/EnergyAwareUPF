#!/usr/bin/env python3
"""
Isolated traffic predictor improvement pipeline.

This script creates a copy of the original traffic dataset, trains multiple
improved traffic-prediction variants in an isolated folder, evaluates them
on a strict temporal hold-out split, and writes comparison artifacts.

Usage:
    /home/ubuntu/EnergyAwareUPF/.venv/bin/python \
      /home/ubuntu/EnergyAwareUPF/traffic_predictor/run_improved_traffic_predictor.py
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
import shutil

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Approach:
    name: str
    description: str
    seq_len: int
    use_calendar: bool
    target_mode: str  # "absolute" or "delta"
    loss: str         # "mse" or "huber"
    learning_rate: float
    batch_size: int
    epochs: int
    early_stopping_patience: int
    reduce_lr_patience: int


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])

    out["hour"] = out[timestamp_col].dt.hour
    out["day_of_week"] = out[timestamp_col].dt.dayofweek

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(np.float32)
    out["is_business_hours"] = out["hour"].between(9, 17).astype(np.float32)

    return out


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(200.0 * np.mean(np.abs(y_pred - y_true) / denom))


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = max(np.sum(np.abs(y_true)), eps)
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def format_architecture(architecture: List[Dict]) -> str:
    parts: List[str] = []
    for layer in architecture:
        layer_type = str(layer.get("type", "")).lower()
        if layer_type == "spatial_dropout1d":
            parts.append(f"SpatialDropout1D({layer.get('rate', '?')})")
        elif layer_type == "lstm":
            parts.append(
                f"LSTM({layer.get('units', '?')}, return_sequences={layer.get('return_sequences', False)})"
            )
        elif layer_type == "dropout":
            parts.append(f"Dropout({layer.get('rate', '?')})")
        elif layer_type == "dense":
            parts.append(f"Dense({layer.get('units', '?')}, {layer.get('activation', 'linear')})")
        else:
            parts.append(str(layer))
    return " → ".join(parts)


def build_model(input_shape: Tuple[int, int], architecture: List[Dict], learning_rate: float, loss_name: str):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    for layer_cfg in architecture:
        cfg = dict(layer_cfg)
        layer_type = cfg.pop("type").lower()
        if layer_type == "spatial_dropout1d":
            model.add(keras.layers.SpatialDropout1D(**cfg))
        elif layer_type == "lstm":
            model.add(keras.layers.LSTM(**cfg))
        elif layer_type == "dropout":
            model.add(keras.layers.Dropout(**cfg))
        elif layer_type == "dense":
            model.add(keras.layers.Dense(**cfg))
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    loss = keras.losses.Huber() if loss_name.lower() == "huber" else "mse"
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
    )
    return model


def split_boundaries(n_total: int, val_frac: float, test_frac: float) -> Tuple[int, int]:
    n_test = int(round(n_total * test_frac))
    n_val = int(round(n_total * val_frac))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Invalid split sizes: no training data left")
    return n_train, n_train + n_val


def prepare_dataset(
    df: pd.DataFrame,
    approach: Approach,
    calendar_cols: List[str],
    val_frac: float,
    test_frac: float,
) -> Dict:
    traffic = df["Traffic_Mbps_scaled"].to_numpy(dtype=np.float32)
    n_total = len(traffic)
    train_end, val_end = split_boundaries(n_total=n_total, val_frac=val_frac, test_frac=test_frac)

    if train_end <= approach.seq_len:
        raise ValueError("Not enough training data for selected sequence length")

    traffic_scaler = MinMaxScaler(feature_range=(0, 1))
    traffic_scaler.fit(traffic[:train_end].reshape(-1, 1))
    traffic_scaled = traffic_scaler.transform(traffic.reshape(-1, 1)).flatten().astype(np.float32)

    calendar_scaled = None
    calendar_scaler = None
    if approach.use_calendar:
        missing = [c for c in calendar_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing calendar columns: {missing}")
        calendar_arr = df[calendar_cols].to_numpy(dtype=np.float32)
        calendar_scaler = MinMaxScaler(feature_range=(0, 1))
        calendar_scaler.fit(calendar_arr[:train_end])
        calendar_scaled = calendar_scaler.transform(calendar_arr).astype(np.float32)

    split_data = {
        "train": {"X": [], "y": [], "y_abs": [], "last": [], "idx": []},
        "val": {"X": [], "y": [], "y_abs": [], "last": [], "idx": []},
        "test": {"X": [], "y": [], "y_abs": [], "last": [], "idx": []},
    }

    for t in range(approach.seq_len, n_total):
        x_traffic = traffic_scaled[t - approach.seq_len : t]
        x = x_traffic.reshape(-1, 1)

        if approach.use_calendar and calendar_scaled is not None:
            cal_now = calendar_scaled[t]
            cal_rep = np.repeat(cal_now[np.newaxis, :], approach.seq_len, axis=0)
            x = np.concatenate([x, cal_rep], axis=1)

        y_abs = float(traffic_scaled[t])
        last_val = float(x_traffic[-1])
        if approach.target_mode == "delta":
            y_target = y_abs - last_val
        else:
            y_target = y_abs

        if t < train_end:
            split_key = "train"
        elif t < val_end:
            split_key = "val"
        else:
            split_key = "test"

        split_data[split_key]["X"].append(x)
        split_data[split_key]["y"].append(y_target)
        split_data[split_key]["y_abs"].append(y_abs)
        split_data[split_key]["last"].append(last_val)
        split_data[split_key]["idx"].append(t)

    for key in ["train", "val", "test"]:
        split_data[key]["X"] = np.array(split_data[key]["X"], dtype=np.float32)
        split_data[key]["y"] = np.array(split_data[key]["y"], dtype=np.float32)
        split_data[key]["y_abs"] = np.array(split_data[key]["y_abs"], dtype=np.float32)
        split_data[key]["last"] = np.array(split_data[key]["last"], dtype=np.float32)
        split_data[key]["idx"] = np.array(split_data[key]["idx"], dtype=np.int32)

    split_data["meta"] = {
        "train_end": train_end,
        "val_end": val_end,
        "n_total": n_total,
        "traffic_scaler": traffic_scaler,
        "calendar_scaler": calendar_scaler,
        "traffic_scaled_full": traffic_scaled,
    }
    return split_data


def reconstruct_abs_scaled(y_pred_raw: np.ndarray, last_scaled: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "delta":
        y_abs = last_scaled + y_pred_raw
    else:
        y_abs = y_pred_raw
    return np.clip(y_abs, 0.0, 1.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_safe": safe_mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }


def plot_overlay(actual: np.ndarray, predicted: np.ndarray, timestamps: pd.Series, title: str, out_png: Path, out_pdf: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
    x = pd.to_datetime(timestamps)
    ax.plot(x, actual, label="Actual Traffic", linewidth=1.8)
    ax.plot(x, predicted, label="Predicted Traffic", linewidth=1.6, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic_Mbps_scaled")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    summary_path: Path,
    approaches: List[Approach],
    architecture: List[Dict],
    data_copy_path: Path,
    val_frac: float,
    test_frac: float,
    comparison_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    best_row: pd.Series,
    results_dir: Path,
):
    lines: List[str] = []
    lines.append("# Improved Traffic Predictor: Isolated Experiments")
    lines.append("")
    lines.append("## Isolation and data safety")
    lines.append("")
    lines.append(f"- Original dataset was NOT modified.")
    lines.append(f"- Copied working dataset: `{data_copy_path}`")
    lines.append(f"- All models/results saved under: `{results_dir.parent}`")
    lines.append("")
    lines.append("## Base architecture")
    lines.append("")
    lines.append(f"- Architecture template: `{format_architecture(architecture)}`")
    lines.append("")
    lines.append("## Improved approaches evaluated")
    lines.append("")
    for ap in approaches:
        lines.append(
            f"- `{ap.name}`: {ap.description} | seq_len={ap.seq_len}, calendar={ap.use_calendar}, "
            f"target_mode={ap.target_mode}, loss={ap.loss}, lr={ap.learning_rate}"
        )
    lines.append("")
    lines.append("## Split strategy")
    lines.append("")
    lines.append(f"- Temporal split (no shuffle): train / val / test = {1 - val_frac - test_frac:.2f} / {val_frac:.2f} / {test_frac:.2f}")
    lines.append("- Scalers fit on TRAIN segment only (leakage-free).")
    lines.append("")
    lines.append("## Model comparison (test set)")
    lines.append("")
    lines.append("| model_name | rmse | mae | r2 | mape_safe | smape | wape | n_test |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in comparison_df.iterrows():
        lines.append(
            f"| {r['model_name']} | {r['rmse']:.6f} | {r['mae']:.6f} | {r['r2']:.6f} | "
            f"{r['mape_safe']:.6f} | {r['smape']:.6f} | {r['wape']:.6f} | {int(r['n_test'])} |"
        )
    lines.append("")
    lines.append("## Baselines (on best-model split)")
    lines.append("")
    lines.append("| baseline | rmse | mae | r2 | mape_safe | smape | wape |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in baseline_df.iterrows():
        lines.append(
            f"| {r['baseline']} | {r['rmse']:.6f} | {r['mae']:.6f} | {r['r2']:.6f} | "
            f"{r['mape_safe']:.6f} | {r['smape']:.6f} | {r['wape']:.6f} |"
        )
    lines.append("")
    lines.append("## Best updated accuracy")
    lines.append("")
    lines.append(
        f"- Best model: `{best_row['model_name']}` with RMSE={best_row['rmse']:.6f}, "
        f"MAE={best_row['mae']:.6f}, R2={best_row['r2']:.6f}, "
        f"MAPE_safe={best_row['mape_safe']:.6f}, sMAPE={best_row['smape']:.6f}, WAPE={best_row['wape']:.6f}."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `results/model_accuracy_comparison.csv`")
    lines.append("- `results/baseline_accuracy_comparison.csv`")
    lines.append("- `results/all_model_predictions.csv`")
    lines.append("- `results/best_model_predictions.csv`")
    lines.append("- `results/best_model_actual_vs_predicted.png`")
    lines.append("- `results/best_model_actual_vs_predicted.pdf`")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    work_root = repo_root / "traffic_predictor"
    data_dir = work_root / "data"
    models_dir = work_root / "models"
    results_dir = work_root / "results"

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(repo_root / "config.yaml")
    traffic_cfg = config["traffic_predictor_training"]

    src_data_path = (repo_root / config["traffic_preprocessing"]["processed_csv_path"]).resolve()
    data_copy_path = data_dir / "processed_traffic_normalized_copy.csv"
    shutil.copy2(src_data_path, data_copy_path)

    # Add calendar features only to copied dataset
    df_base = pd.read_csv(data_copy_path)
    df_work = add_calendar_features(df_base, timestamp_col=config["traffic_preprocessing"]["timestamp_col"])
    data_copy_with_calendar = data_dir / "processed_traffic_normalized_copy_with_calendar.csv"
    df_work.to_csv(data_copy_with_calendar, index=False)

    # Save snapshot of config for reproducibility
    with (work_root / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    architecture = traffic_cfg["architecture"]
    calendar_cols = list(traffic_cfg.get("calendar_feature_columns", []))

    seed = int(traffic_cfg.get("random_state", 42))
    keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    val_frac = 0.15
    test_frac = float(traffic_cfg.get("test_size", 0.2))

    approaches = [
        Approach(
            name="baseline_abs_nocal_mse",
            description="Leak-free scaling + temporal split + absolute target",
            seq_len=96,
            use_calendar=False,
            target_mode="absolute",
            loss="mse",
            learning_rate=1e-3,
            batch_size=int(traffic_cfg.get("batch_size", 128)),
            epochs=120,
            early_stopping_patience=15,
            reduce_lr_patience=7,
        ),
        Approach(
            name="delta_nocal_huber",
            description="Residual/delta target + Huber loss",
            seq_len=96,
            use_calendar=False,
            target_mode="delta",
            loss="huber",
            learning_rate=5e-4,
            batch_size=int(traffic_cfg.get("batch_size", 128)),
            epochs=120,
            early_stopping_patience=15,
            reduce_lr_patience=7,
        ),
        Approach(
            name="abs_calendar_mse",
            description="Calendar features enabled (copy only) + absolute target",
            seq_len=96,
            use_calendar=True,
            target_mode="absolute",
            loss="mse",
            learning_rate=1e-3,
            batch_size=int(traffic_cfg.get("batch_size", 128)),
            epochs=120,
            early_stopping_patience=15,
            reduce_lr_patience=7,
        ),
        Approach(
            name="delta_calendar_huber",
            description="Calendar features + residual target + Huber loss",
            seq_len=96,
            use_calendar=True,
            target_mode="delta",
            loss="huber",
            learning_rate=5e-4,
            batch_size=int(traffic_cfg.get("batch_size", 128)),
            epochs=120,
            early_stopping_patience=15,
            reduce_lr_patience=7,
        ),
    ]

    comparison_rows: List[Dict] = []
    all_predictions_rows: List[pd.DataFrame] = []
    baseline_rows_for_best: List[Dict] = []

    best = {
        "rmse": float("inf"),
        "row": None,
        "pred_df": None,
        "timestamps": None,
        "y_true": None,
        "y_pred": None,
        "baseline_df": None,
    }

    for ap in approaches:
        keras.backend.clear_session()
        data_pack = prepare_dataset(
            df=df_work,
            approach=ap,
            calendar_cols=calendar_cols,
            val_frac=val_frac,
            test_frac=test_frac,
        )

        X_train = data_pack["train"]["X"]
        y_train = data_pack["train"]["y"]
        X_val = data_pack["val"]["X"]
        y_val = data_pack["val"]["y"]
        X_test = data_pack["test"]["X"]
        y_test_abs_scaled = data_pack["test"]["y_abs"]
        last_test_scaled = data_pack["test"]["last"]
        idx_test = data_pack["test"]["idx"]

        model = build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            architecture=architecture,
            learning_rate=ap.learning_rate,
            loss_name=ap.loss,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=ap.early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=ap.reduce_lr_patience,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=ap.epochs,
            batch_size=ap.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        model_path = models_dir / f"{ap.name}.keras"
        model.save(model_path)

        # Save training history
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(results_dir / f"history_{ap.name}.csv", index=False)

        y_pred_raw = model.predict(X_test, verbose=0).flatten().astype(np.float32)
        y_pred_abs_scaled = reconstruct_abs_scaled(
            y_pred_raw=y_pred_raw,
            last_scaled=last_test_scaled,
            target_mode=ap.target_mode,
        )

        scaler = data_pack["meta"]["traffic_scaler"]
        y_true = scaler.inverse_transform(y_test_abs_scaled.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred_abs_scaled.reshape(-1, 1)).flatten()

        metrics = evaluate_predictions(y_true, y_pred)
        row = {
            "model_name": ap.name,
            "description": ap.description,
            "seq_len": ap.seq_len,
            "use_calendar": ap.use_calendar,
            "target_mode": ap.target_mode,
            "loss": ap.loss,
            "learning_rate": ap.learning_rate,
            "batch_size": ap.batch_size,
            "epochs_config": ap.epochs,
            "epochs_ran": len(history.history.get("loss", [])),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            **metrics,
        }
        comparison_rows.append(row)

        pred_df = pd.DataFrame(
            {
                "model_name": ap.name,
                "time_index": idx_test,
                "timestamp": df_work.loc[idx_test, "timestamp"].astype(str).to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
                "abs_error": np.abs(y_true - y_pred),
                "ape_safe_pct": np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-3) * 100.0,
            }
        )
        pred_df.to_csv(results_dir / f"predictions_{ap.name}.csv", index=False)
        all_predictions_rows.append(pred_df)

        # Baselines on this exact test split
        traffic_scaled_full = data_pack["meta"]["traffic_scaled_full"]
        persist_abs_scaled = last_test_scaled
        persist = scaler.inverse_transform(persist_abs_scaled.reshape(-1, 1)).flatten()
        baseline_persist = {
            "model_name": ap.name,
            "baseline": "naive_persistence_t-1",
            **evaluate_predictions(y_true, persist),
        }

        # Daily seasonal baseline at 96-step lag (if available)
        daily_scaled = np.array(
            [traffic_scaled_full[i - 96] if i - 96 >= 0 else np.nan for i in idx_test],
            dtype=np.float32,
        )
        valid_daily = ~np.isnan(daily_scaled)
        if valid_daily.any():
            daily = scaler.inverse_transform(daily_scaled[valid_daily].reshape(-1, 1)).flatten()
            baseline_daily_metrics = evaluate_predictions(y_true[valid_daily], daily)
        else:
            baseline_daily_metrics = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape_safe": np.nan, "smape": np.nan, "wape": np.nan}
        baseline_daily = {
            "model_name": ap.name,
            "baseline": "naive_daily_t-96",
            **baseline_daily_metrics,
        }

        # Track best by RMSE
        if metrics["rmse"] < best["rmse"]:
            best["rmse"] = metrics["rmse"]
            best["row"] = row
            best["pred_df"] = pred_df
            best["timestamps"] = df_work.loc[idx_test, "timestamp"].astype(str).to_numpy()
            best["y_true"] = y_true
            best["y_pred"] = y_pred
            best["baseline_df"] = pd.DataFrame([baseline_persist, baseline_daily])

    comparison_df = pd.DataFrame(comparison_rows).sort_values("rmse").reset_index(drop=True)
    comparison_df.to_csv(results_dir / "model_accuracy_comparison.csv", index=False)

    all_pred_df = pd.concat(all_predictions_rows, ignore_index=True)
    all_pred_df.to_csv(results_dir / "all_model_predictions.csv", index=False)

    if best["pred_df"] is None or best["row"] is None or best["baseline_df"] is None:
        raise RuntimeError("No model results were produced.")

    best_pred_df = best["pred_df"].copy()
    best_pred_df.to_csv(results_dir / "best_model_predictions.csv", index=False)
    best_baseline_df = best["baseline_df"].copy()
    best_baseline_df.to_csv(results_dir / "baseline_accuracy_comparison.csv", index=False)

    title = (
        f"Best Improved Traffic Predictor: {best['row']['model_name']} "
        f"(RMSE={best['row']['rmse']:.2f}, R2={best['row']['r2']:.3f})"
    )
    plot_overlay(
        actual=best["y_true"],
        predicted=best["y_pred"],
        timestamps=pd.Series(best["timestamps"]),
        title=title,
        out_png=results_dir / "best_model_actual_vs_predicted.png",
        out_pdf=results_dir / "best_model_actual_vs_predicted.pdf",
    )

    write_summary(
        summary_path=work_root / "README_RESULTS.md",
        approaches=approaches,
        architecture=architecture,
        data_copy_path=data_copy_with_calendar,
        val_frac=val_frac,
        test_frac=test_frac,
        comparison_df=comparison_df,
        baseline_df=best_baseline_df,
        best_row=pd.Series(best["row"]),
        results_dir=results_dir,
    )

    run_meta = {
        "approaches": [asdict(a) for a in approaches],
        "best_model": best["row"],
        "data_copy": str(data_copy_with_calendar),
        "result_files": {
            "comparison_csv": str(results_dir / "model_accuracy_comparison.csv"),
            "baseline_csv": str(results_dir / "baseline_accuracy_comparison.csv"),
            "best_predictions_csv": str(results_dir / "best_model_predictions.csv"),
            "plot_png": str(results_dir / "best_model_actual_vs_predicted.png"),
            "summary_md": str(work_root / "README_RESULTS.md"),
        },
    }
    (work_root / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("✓ Isolated traffic predictor experiments complete")
    print(f"  Data copy: {data_copy_with_calendar}")
    print(f"  Best model: {best['row']['model_name']}")
    print(
        "  Metrics: "
        f"RMSE={best['row']['rmse']:.6f}, MAE={best['row']['mae']:.6f}, "
        f"R2={best['row']['r2']:.6f}, MAPE_safe={best['row']['mape_safe']:.6f}, "
        f"sMAPE={best['row']['smape']:.6f}, WAPE={best['row']['wape']:.6f}"
    )
    print(f"  Comparison CSV: {results_dir / 'model_accuracy_comparison.csv'}")


if __name__ == "__main__":
    main()
