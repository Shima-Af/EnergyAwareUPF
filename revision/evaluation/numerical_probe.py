"""Controlled profiling-reference diagnostics in a fresh process per backend.

This consumes the archived profiling prediction CSV, not traffic evaluation
results. Backend variants diagnose arithmetic; none selects a controller/model.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import warnings
from pathlib import Path

from .twin import CPU_ENV, ROOT, artifact_inventory, sha256

PROFILES = ("cpu_scalar", "cpu_predict32", "cpu_onednn_predict32",
            "gpu_predict32_tf32_on", "gpu_predict32_tf32_off", "gpu_scalar_tf32_off")


def probe(profile: str, reference: Path):
    inventory = artifact_inventory()
    settings = dict(CPU_ENV)
    gpu = profile.startswith("gpu")
    settings["CUDA_VISIBLE_DEVICES"] = "0" if gpu else "-1"
    settings["TF_ENABLE_ONEDNN_OPTS"] = "1" if profile == "cpu_onednn_predict32" else "0"
    settings["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.update(settings)
    import numpy as np
    import tensorflow as tf
    import joblib

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(profile == "gpu_predict32_tf32_on")
    if not gpu:
        tf.config.set_visible_devices([], "GPU")
    elif not tf.config.list_physical_devices("GPU"):
        return {"profile": profile, "status": "gpu_unavailable", "environment": settings}, []
    if gpu:
        tf.config.set_soft_device_placement(False)  # A GPU control must not fall back to CPU.
    components, captured = {}, []
    for item in inventory:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = (tf.keras.models.load_model(item["absolute_path"], compile=False)
                     if item["role"] == "model" else joblib.load(item["absolute_path"]))
        captured.extend({"path": item["path"], "category": w.category.__name__, "message": str(w.message)}
                        for w in caught)
        components[item["mode"], item["target"], item["role"]] = value
    rows = list(csv.DictReader(reference.open()))
    output, groups = [], {}
    device = "/GPU:0" if gpu else "/CPU:0"
    for mode, label in [("dpdk", "DPDK"), ("usr", "USR")]:
        for target, label_target in [("power", "attributed power"), ("qos", "QoS score")]:
            selected = [r for r in rows if r["mode"] == label and r["target"] == label_target]
            x = np.asarray([float(r["offered_load"]) for r in selected], dtype=np.float32).reshape(-1, 1)
            batch_logged = np.log1p(x)
            scalar_logged = np.vstack([np.log1p(np.asarray([[v]], dtype=np.float32)) for v in x[:, 0]])
            logged = scalar_logged if "scalar" in profile else batch_logged
            sx, sy = components[mode, target, "input_scaler"], components[mode, target, "target_scaler"]
            scaled = sx.transform(logged.copy())
            explicit = logged.copy()
            if sx.with_mean:
                explicit -= sx.mean_
            if sx.with_std:
                explicit /= sx.scale_
            model = components[mode, target, "model"]
            with tf.device(device):
                if "scalar" in profile:
                    # Deduplicate identical loads; exactly one input per forward pass.
                    unique = {float(v): model(tf.convert_to_tensor([[v]], dtype=tf.float32), training=False).numpy()[0, 0]
                              for v in np.unique(scaled)}
                    network = np.asarray([unique[float(v)] for v in scaled[:, 0]], dtype=np.float32).reshape(-1, 1)
                else:
                    network = model.predict(scaled, batch_size=32, verbose=0)
            predicted = sy.inverse_transform(network.copy()).ravel()
            explicit_inverse = network.copy()
            if sy.with_std:
                explicit_inverse *= sy.scale_
            if sy.with_mean:
                explicit_inverse += sy.mean_
            reference_values = np.asarray([float(r["y_pred"]) for r in selected], dtype=np.float64)
            delta = predicted.astype(np.float64) - reference_values
            invalid = (~np.isfinite(predicted) | (predicted < 0)
                       | ((predicted > 1) if target == "qos" else False))
            groups[f"{mode}_{target}"] = {
                "rows": len(selected), "unique_loads": len(np.unique(x)),
                "exact_matches": int(np.count_nonzero(delta == 0)),
                "max_abs_delta": float(np.max(np.abs(delta))),
                "mean_delta": float(np.mean(delta)),
                "rmse_delta": float(np.sqrt(np.mean(delta ** 2))),
                "invalid_output_rows": int(np.count_nonzero(invalid)),
                "batch_vs_single_input_log1p_max_delta": float(np.max(np.abs(batch_logged - scalar_logged))),
                "input_scaler_matches_explicit_saved_statistics_bitwise": bool(np.array_equal(scaled, explicit)),
                "inverse_scaler_matches_explicit_saved_statistics_bitwise": bool(np.array_equal(predicted, explicit_inverse.ravel())),
            }
            for row, value, difference, bad in zip(selected, predicted, delta, invalid):
                output.append({"profile": profile, "mode": mode, "target": target,
                               "sample_idx": row["sample_idx"], "offered_load_mbps": row["offered_load"],
                               "archived_prediction": float(row["y_pred"]), "fresh_prediction": float(value),
                               "delta": float(difference), "invalid_prediction": bool(bad)})
    return {
        "profile": profile, "status": "completed", "reference_path": str(reference.resolve()),
        "reference_sha256": sha256(reference), "source": "archived profiling predictions, not traffic final-test evaluation",
        "environment": settings, "device_requested": device,
        "logical_devices": [str(d) for d in tf.config.list_logical_devices()],
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "dependencies": {n: importlib.metadata.version(n) for n in
                         ("tensorflow", "keras", "numpy", "scikit-learn", "joblib")},
        "artifact_sha256": {i["path"]: i["sha256"] for i in inventory},
        "warnings": captured, "groups": groups,
    }, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "profiling_model_predictions_vs_target.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / "revision"):
        raise ValueError("Diagnostic outputs must be new files under revision/")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary, rows = probe(args.profile, args.reference)
    with (args.output_dir / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if rows:
        with (args.output_dir / "predictions.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"profile": args.profile, "status": summary["status"], "groups": summary.get("groups", {})}))


if __name__ == "__main__":
    main()
