"""Read-only, fixed-profile inference for the four original UPF surrogates.

TensorFlow is imported lazily after establishing a dedicated deterministic CPU
profile. Use a fresh process if TensorFlow was already initialized elsewhere.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .contract import ACTIONS, Outcome
from .data import fingerprint

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT / "revision/configs/twin_artifacts_v1.json"
CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_DETERMINISTIC_OPS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "TF_CPP_MIN_LOG_LEVEL": "2",
}
_RUNTIME = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialization_metadata(path: Path, role: str) -> dict:
    if role == "model":
        with zipfile.ZipFile(path) as archive:
            return json.loads(archive.read("metadata.json"))
    version = re.search(rb"_sklearn_version.{0,8}?(\d+\.\d+\.\d+)", path.read_bytes())
    if version is None:
        raise ValueError(f"No sklearn serialization version found: {path}")
    return {"sklearn_version": version.group(1).decode()}


def artifact_inventory(config_path: Path = DEFAULT_ARTIFACTS) -> tuple[dict, ...]:
    config = json.loads(config_path.read_text())
    if config["schema_version"] != 1 or config["inference_policy"] != "cpu_float32_single_sample_v1":
        raise ValueError("Unsupported twin manifest or inference policy")
    expected = {(m, t, r) for m in ("dpdk", "usr") for t in ("power", "qos")
                for r in ("model", "input_scaler", "target_scaler")}
    rows = config["artifacts"]
    if len(rows) != 12 or {(r["mode"], r["target"], r["role"]) for r in rows} != expected:
        raise ValueError("Expected four models and their eight corresponding scalers")
    inventory = []
    for row in rows:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT / "saved_models/prediction_models"):
            raise ValueError("Twin artifacts must be original repository profiling artifacts")
        actual = sha256(path)
        if actual != row["sha256"]:
            raise ValueError(f"Artifact SHA-256 mismatch: {path}")
        metadata = serialization_metadata(path, row["role"])
        if metadata != row["serialization"]:
            raise ValueError(f"Serialization metadata mismatch: {path}")
        inventory.append({**row, "absolute_path": str(path), "bytes": path.stat().st_size})
    return tuple(inventory)


def _runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    if "tensorflow" in sys.modules:
        raise RuntimeError("TensorFlow was imported before the twin CPU profile; use a fresh process")
    for name, value in CPU_ENV.items():
        os.environ[name] = value
    import numpy as np
    import tensorflow as tf
    import joblib

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    if tf.config.get_visible_devices("GPU"):
        raise RuntimeError("CPU inference profile unexpectedly exposes a GPU")
    metadata = {
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "tensorflow", "keras", "scikit-learn", "joblib", "h5py")},
        "environment": dict(CPU_ENV),
        "tensorflow_build": tf.sysconfig.get_build_info(),
        "visible_devices": [str(d) for d in tf.config.get_visible_devices()],
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "tensor_float_32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "model_call": "model(one_float32_sample, training=False); compile=False on load",
        "batch_policy": "one sample per forward pass, independent of caller batching",
        "input_pipeline": "float32(load/k) -> float32 log1p -> saved input scaler",
        "output_pipeline": "float32 model output -> saved target inverse scaler; no clipping",
        "serialized_tensorflow_version": None,
        "serialized_tensorflow_version_note": "Not recorded in the .keras metadata; not inferred from Keras version",
    }
    _RUNTIME = np, tf, joblib, metadata
    return _RUNTIME


@dataclass(frozen=True)
class RawPrediction:
    action: int
    load_mbps: float
    instance_count: int
    per_instance_load_mbps: float
    float32_model_load_mbps: float
    per_instance_power_w: float
    power_w: float
    qos: float

    @property
    def failures(self) -> tuple[str, ...]:
        failures = []
        if not math.isfinite(self.power_w) or self.power_w < 0:
            failures.append("nonfinite_or_negative_attributed_power")
        if not math.isfinite(self.qos) or not 0 <= self.qos <= 1:
            failures.append("nonfinite_or_out_of_range_qos")
        return tuple(failures)


class InvalidPrediction(ValueError):
    def __init__(self, raw: RawPrediction, twin_id: str):
        self.raw, self.twin_id = raw, twin_id
        super().__init__(json.dumps({"error": "invalid_surrogate_prediction", "twin_id": twin_id,
                                     "prediction": asdict(raw), "failures": raw.failures}))


class FittedUPFTwin:
    """Balanced-load approximation, using exactly the pinned historical artifacts."""

    def __init__(self, config_path: Path = DEFAULT_ARTIFACTS):
        self.config_path = config_path.resolve()
        self.artifacts = artifact_inventory(config_path)  # Verify before unpickling/loading.
        self._np, self._tf, joblib, runtime = _runtime()
        self._runtime_metadata = runtime
        self._components = {}
        self.load_warnings = []
        for item in self.artifacts:
            key = item["mode"], item["target"], item["role"]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if item["role"] == "model":
                    component = self._tf.keras.models.load_model(item["absolute_path"], compile=False)
                else:
                    component = joblib.load(item["absolute_path"])
            self.load_warnings.extend({"artifact": item["path"], "category": w.category.__name__,
                                       "message": str(w.message)} for w in caught)
            if item["role"] != "model" and component.n_features_in_ != 1:
                raise ValueError(f"Expected a single-feature scaler: {item['path']}")
            self._components[key] = component
        self._cache = {}
        self.identity = fingerprint({
            "artifacts": [{k: i[k] for k in ("mode", "target", "role", "path", "sha256")}
                          for i in self.artifacts],
            "runtime": runtime, "adapter_sha256": sha256(Path(__file__)),
            "balanced_load": "per_instance_load=load/k; total_power=k*power; qos=per_instance_qos",
        })

    @property
    def manifest(self) -> dict:
        return {"twin_id": self.identity, "artifact_config_sha256": sha256(self.config_path),
                "adapter_path": str(Path(__file__).resolve()), "adapter_sha256": sha256(Path(__file__)),
                "artifacts": self.artifacts, "runtime": self._runtime_metadata,
                "load_warnings": self.load_warnings,
                "invalid_prediction_policy": "retain raw diagnostics; raise before evaluation accounting"}

    def verify_artifacts(self) -> None:
        for item in self.artifacts:
            if sha256(Path(item["absolute_path"])) != item["sha256"]:
                raise ValueError(f"Artifact changed after loading: {item['path']}")

    def _predict_instance(self, mode: str, load: float) -> tuple[float, float]:
        np = self._np
        with np.errstate(over="ignore"):
            value = np.float32(load)
        if not np.isfinite(value):
            raise ValueError("Load is not representable as finite float32 for the saved models")
        key = mode, value.tobytes()
        if key not in self._cache:
            logged = np.log1p(np.asarray([[value]], dtype=np.float32))
            outputs = {}
            for target in ("power", "qos"):
                scaled = self._components[mode, target, "input_scaler"].transform(logged.copy())
                model = self._components[mode, target, "model"]
                predicted = model(self._tf.convert_to_tensor(scaled, dtype=self._tf.float32), training=False).numpy()
                result = self._components[mode, target, "target_scaler"].inverse_transform(predicted)
                if result.shape != (1, 1):
                    raise ValueError("Unexpected profiling-model output shape")
                outputs[target] = float(result[0, 0])
            self._cache[key] = outputs["power"], outputs["qos"]
        return self._cache[key]

    def raw_prediction(self, action: int, load_mbps: float) -> RawPrediction:
        if type(action) is not int or action not in ACTIONS:
            raise ValueError("Action must be an integer in {0,1,2}")
        if not math.isfinite(load_mbps) or load_mbps < 0:
            raise ValueError("Load must be finite, nonnegative Mbps")
        k = 1 if action == 0 else action
        per_load = load_mbps / k
        power, qos = self._predict_instance("dpdk" if action == 0 else "usr", per_load)
        return RawPrediction(action, load_mbps, k, per_load, float(self._np.float32(per_load)),
                             power, k * power, qos)

    def outcome(self, action: int, load_mbps: float) -> Outcome:
        raw = self.raw_prediction(action, load_mbps)
        if raw.failures:
            raise InvalidPrediction(raw, self.identity)
        return Outcome(raw.power_w, raw.qos)

    def clear_cache(self) -> None:
        """Allow uncached repeatability checks without changing artifacts or parameters."""
        self._cache.clear()
