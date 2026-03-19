#!/usr/bin/env python3
"""
Compute held-out profiling-model accuracy for digital-twin surrogates.

Usage:
    python -m scripts.compute_profiling_model_accuracy

Optional:
    python -m scripts.compute_profiling_model_accuracy \
        --dpdk-csv /path/to/dpdk_processed.csv \
        --oai-csv /path/to/oai_processed.csv \
        --out-dir .
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.utils import UPFApproximatorBundle, load_config


DEFAULT_FALLBACK_DIRS = [
    Path("/home/ubuntu/DDT_UPF_Selection_DRL/data/processed"),
    Path("/home/ubuntu/xRL_UpfSelection/data/processed"),
    Path("/home/ubuntu/dynamic-upf-selection-rl/data/processed"),
]


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
        elif layer_type == "batch_norm":
            parts.append("BatchNorm")
        elif layer_type == "dropout":
            rate = layer.get("rate", "?")
            parts.append(f"Dropout({rate})")
        elif layer_type == "lstm":
            units = layer.get("units", "?")
            rs = layer.get("return_sequences", False)
            parts.append(f"LSTM({units}, return_sequences={rs})")
        elif layer_type == "spatial_dropout1d":
            rate = layer.get("rate", "?")
            parts.append(f"SpatialDropout1D({rate})")
        else:
            parts.append(str(layer))
    return " → ".join(parts) if parts else "(not specified)"


def resolve_dataset(
    preferred: Path,
    filename: str,
    fallback_dirs: List[Path],
) -> Path:
    if preferred.exists():
        return preferred
    for base in fallback_dirs:
        candidate = base / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve dataset. Missing preferred path {preferred} and no fallback for {filename}."
    )


def evaluate_mode(
    mode_name: str,
    upf_name: str,
    csv_path: Path,
    model_dir: Path,
    feature_col: str,
    power_col: str,
    qos_col: str,
    split_kwargs: Dict,
    mape_eps: float,
) -> Tuple[List[Dict], List[Dict]]:
    df = pd.read_csv(csv_path)

    required = [feature_col, power_col, qos_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{mode_name}: missing columns {missing} in {csv_path}")

    X = df[[feature_col]].to_numpy(dtype=np.float32)
    y_power = df[power_col].to_numpy(dtype=np.float32)
    y_qos = df[qos_col].to_numpy(dtype=np.float32)

    _, X_test, _, ypow_test, _, yqos_test = train_test_split(
        X, y_power, y_qos, **split_kwargs
    )

    bundle = UPFApproximatorBundle(
        upf_name=upf_name,
        model_dir=str(model_dir),
        model_type="keras",
    )
    yqos_pred, ypow_pred = bundle.predict_batch(X_test.flatten())

    metrics_rows: List[Dict] = []
    prediction_rows: List[Dict] = []

    metric_specs = [
        ("attributed power", ypow_test.astype(np.float64), ypow_pred.astype(np.float64)),
        ("QoS score", yqos_test.astype(np.float64), yqos_pred.astype(np.float64)),
    ]

    split_name = (
        f"holdout_test_testsize{split_kwargs['test_size']}_"
        f"rs{split_kwargs['random_state']}_shuffleTrue"
    )

    for target_name, y_true, y_pred in metric_specs:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        mape = safe_mape(y_true, y_pred, eps=mape_eps)

        metrics_rows.append(
            {
                "mode": mode_name,
                "target": target_name,
                "split_name": split_name,
                "n_samples": int(len(y_true)),
                "rmse": rmse,
                "mape": mape,
                "r2": r2,
                "mae": mae,
            }
        )

        for idx, (x_val, yt, yp) in enumerate(zip(X_test.flatten(), y_true, y_pred)):
            prediction_rows.append(
                {
                    "mode": mode_name,
                    "target": target_name,
                    "split_name": split_name,
                    "sample_idx": idx,
                    "offered_load": float(x_val),
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "abs_error": float(abs(yt - yp)),
                    "ape_safe_pct": float(abs(yt - yp) / max(abs(yt), mape_eps) * 100.0),
                }
            )

    return metrics_rows, prediction_rows


def build_summary_markdown(
    accuracy_df: pd.DataFrame,
    config_path: Path,
    model_dir: Path,
    dpdk_csv: Path,
    usr_csv: Path,
    split_kwargs: Dict,
    feature_col: str,
    power_col: str,
    qos_col: str,
    mape_eps: float,
    power_train_cfg: Dict,
    perf_train_cfg: Dict,
    power_default_arch: List[Dict],
    power_dpdk_arch: List[Dict],
    power_oai_arch: List[Dict],
    performance_arch: List[Dict],
    accuracy_csv_name: str,
    predictions_csv_name: str,
) -> str:
    split_name = (
        f"holdout_test_testsize{split_kwargs['test_size']}_"
        f"rs{split_kwargs['random_state']}_shuffleTrue"
    )

    lines = [
        "# Profiling Model Accuracy (Digital Twin)",
        "",
        "## Files and pipeline used",
        "",
        f"- Configuration: `{config_path}`",
        "- Runtime profiling inference path: `src/utils.py` (`UPFApproximatorBundle.predict_batch`)",
        "- Saved profiling artifacts (loaded, not refit):",
        f"  - `{model_dir / 'dpdk_upf_power_model.keras'}`",
        f"  - `{model_dir / 'dpdk_upf_performance_model.keras'}`",
        f"  - `{model_dir / 'oai_upf_power_model.keras'}`",
        f"  - `{model_dir / 'oai_upf_performance_model.keras'}`",
        f"  - scaler files in `{model_dir}`",
        "- Profiling datasets:",
        f"  - DPDK: `{dpdk_csv}`",
        f"  - USR: `{usr_csv}`",
        "",
        "## Training architectures (from scripts + config)",
        "",
        "- Power model training behavior (as in `train_power_models.py`):",
        "  - Uses `power_model_training.default` then applies `power_model_training.upf_specific[mode]` overrides.",
        f"  - DPDK attributed power architecture: `{format_architecture(power_dpdk_arch)}`",
        f"  - USR attributed power architecture: `{format_architecture(power_oai_arch)}`",
        f"  - Default power architecture (fallback): `{format_architecture(power_default_arch)}`",
        "",
        "- QoS model training behavior (as in `train_performance_models.py`):",
        "  - Uses one shared `performance_model_training.architecture` for both DPDK and USR.",
        f"  - QoS architecture: `{format_architecture(performance_arch)}`",
        "",
        "- Training hyperparameters from config/scripts:",
        f"  - Power: lr={power_train_cfg.get('learning_rate')}, epochs={power_train_cfg.get('epochs')}, batch_size={power_train_cfg.get('batch_size')}, early_stopping_patience={power_train_cfg.get('early_stopping_patience')}, lr_patience={power_train_cfg.get('lr_patience')}",
        f"  - QoS: lr={perf_train_cfg.get('learning_rate')}, epochs={perf_train_cfg.get('epochs')}, batch_size={perf_train_cfg.get('batch_size')}, early_stopping_patience={perf_train_cfg.get('early_stopping_patience')}, lr_patience={perf_train_cfg.get('lr_patience')}",
        "  - Both scripts use the held-out split as `validation_data` during training.",
        "",
        "## Held-out split and preprocessing used",
        "",
        f"- Split rule: `train_test_split(test_size={split_kwargs['test_size']}, random_state={split_kwargs['random_state']}, shuffle=True)`",
        f"- Input feature: `{feature_col}`",
        "- Targets:",
        f"  - attributed power: `{power_col}`",
        f"  - QoS score: `{qos_col}`",
        "- Input preprocessing: `log1p(offered_load)` then `StandardScaler` transform",
        "- Target preprocessing: inverse `StandardScaler` transform after prediction",
        f"- Evaluation split name: `{split_name}`",
        "",
        "## MAPE handling",
        "",
        f"- Safe MAPE uses epsilon `{mape_eps}` in target units:",
        f"  - `MAPE_safe = mean(|y_true - y_pred| / max(|y_true|, {mape_eps})) * 100`",
        "",
        "## Accuracy table",
        "",
        "| mode | target | split_name | n_samples | rmse | mape | r2 | mae |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    table_df = accuracy_df.copy().sort_values(["mode", "target"]).reset_index(drop=True)
    for _, row in table_df.iterrows():
        lines.append(
            "| "
            f"{row['mode']} | {row['target']} | {row['split_name']} | {int(row['n_samples'])} | "
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
        description="Compute held-out accuracy for DPDK/USR profiling models (attributed power and QoS score)."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--dpdk-csv", default=None, help="Override DPDK profiling CSV path")
    parser.add_argument("--oai-csv", default=None, help="Override OAI/USR profiling CSV path")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument(
        "--feature-col",
        default="Objective Throughput (DL/UL)",
        help="Offered-load feature column used by profiling models",
    )
    parser.add_argument(
        "--mape-eps",
        type=float,
        default=1e-3,
        help="Epsilon for safe MAPE denominator",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print final accuracy table to stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    config = load_config(str(config_path))
    prediction_cfg = config["prediction_model_data"]
    power_train_cfg = config["power_model_training"]["default"]
    power_upf_cfg = config["power_model_training"].get("upf_specific", {})
    perf_train_cfg = config["performance_model_training"]

    power_default_arch = power_train_cfg.get("architecture", [])
    power_dpdk_arch = power_upf_cfg.get("dpdk", {}).get("architecture", power_default_arch)
    power_oai_arch = power_upf_cfg.get("oai", {}).get("architecture", power_default_arch)
    if not power_oai_arch:
        power_oai_arch = power_default_arch
    performance_arch = perf_train_cfg.get("architecture", [])

    preferred_dpdk = Path(args.dpdk_csv) if args.dpdk_csv else Path(prediction_cfg["dpdk_csv"])
    preferred_oai = Path(args.oai_csv) if args.oai_csv else Path(prediction_cfg["oai_csv"])
    if not preferred_dpdk.is_absolute():
        preferred_dpdk = (project_root / preferred_dpdk).resolve()
    if not preferred_oai.is_absolute():
        preferred_oai = (project_root / preferred_oai).resolve()

    dpdk_csv = resolve_dataset(preferred_dpdk, "dpdk_processed.csv", DEFAULT_FALLBACK_DIRS)
    usr_csv = resolve_dataset(preferred_oai, "oai_processed.csv", DEFAULT_FALLBACK_DIRS)

    model_dir = Path(config["paths"]["prediction_model_dir"])
    if not model_dir.is_absolute():
        model_dir = (project_root / model_dir).resolve()

    split_kwargs = {
        "test_size": float(power_train_cfg["test_size"]),
        "random_state": int(power_train_cfg["random_state"]),
        "shuffle": True,
    }

    power_col = prediction_cfg["power_col"]
    qos_col = prediction_cfg["qos_col"]

    metrics_rows: List[Dict] = []
    prediction_rows: List[Dict] = []

    for mode_name, upf_name, csv_path in [
        ("DPDK", "dpdk", dpdk_csv),
        ("USR", "oai", usr_csv),
    ]:
        mode_metrics, mode_predictions = evaluate_mode(
            mode_name=mode_name,
            upf_name=upf_name,
            csv_path=csv_path,
            model_dir=model_dir,
            feature_col=args.feature_col,
            power_col=power_col,
            qos_col=qos_col,
            split_kwargs=split_kwargs,
            mape_eps=float(args.mape_eps),
        )
        metrics_rows.extend(mode_metrics)
        prediction_rows.extend(mode_predictions)

    accuracy_df = pd.DataFrame(metrics_rows).sort_values(["mode", "target"]).reset_index(drop=True)
    predictions_df = pd.DataFrame(prediction_rows)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    accuracy_csv = out_dir / "profiling_model_accuracy.csv"
    summary_md = out_dir / "profiling_model_accuracy_summary.md"
    predictions_csv = out_dir / "profiling_model_predictions_vs_target.csv"

    accuracy_df.to_csv(accuracy_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)

    summary_text = build_summary_markdown(
        accuracy_df=accuracy_df,
        config_path=config_path,
        model_dir=model_dir,
        dpdk_csv=dpdk_csv,
        usr_csv=usr_csv,
        split_kwargs=split_kwargs,
        feature_col=args.feature_col,
        power_col=power_col,
        qos_col=qos_col,
        mape_eps=float(args.mape_eps),
        power_train_cfg=power_train_cfg,
        perf_train_cfg=perf_train_cfg,
        power_default_arch=power_default_arch,
        power_dpdk_arch=power_dpdk_arch,
        power_oai_arch=power_oai_arch,
        performance_arch=performance_arch,
        accuracy_csv_name=accuracy_csv.name,
        predictions_csv_name=predictions_csv.name,
    )
    summary_md.write_text(summary_text, encoding="utf-8")

    print(f"Wrote {accuracy_csv}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {predictions_csv}")

    if args.print_table:
        print()
        print(accuracy_df.to_string(index=False))


if __name__ == "__main__":
    main()
