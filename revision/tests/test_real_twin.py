"""Integration checks using saved models; no real final-test rollout is performed."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from revision.evaluation.contract import State, actuate
from revision.evaluation.static import StaticController
from revision.evaluation.twin import (
    DEFAULT_ARTIFACTS, ROOT, FittedUPFTwin, InvalidPrediction, artifact_inventory,
)
from revision.evaluation.validate_static import diagnostic_json, run_static_validation, validation_inputs


@pytest.fixture(scope="module")
def real_twin():
    return FittedUPFTwin()


@pytest.fixture(scope="module")
def real_validation():
    return validation_inputs()


@pytest.fixture(scope="module")
def real_static_run(real_twin, real_validation):
    h, contract, metadata = real_validation
    return run_static_validation(real_twin, h, contract, metadata["protocol_sha256"])


def test_all_twelve_original_artifact_hashes_and_serializer_versions_are_pinned():
    inventory = artifact_inventory()
    assert len(inventory) == 12
    assert len({r["absolute_path"] for r in inventory}) == 12
    for item in inventory:
        assert hashlib.sha256(Path(item["absolute_path"]).read_bytes()).hexdigest() == item["sha256"]
        if item["role"] == "model":
            assert item["serialization"]["keras_version"] == "3.11.3"
        else:
            assert item["serialization"]["sklearn_version"] == "1.7.2"


def test_artifact_hash_mismatch_fails_before_model_loading(tmp_path):
    manifest = json.loads(DEFAULT_ARTIFACTS.read_text())
    manifest["artifacts"][0]["sha256"] = "0" * 64
    fake = tmp_path / "tampered_manifest.json"
    fake.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FittedUPFTwin(fake)


def test_missing_model_or_scaler_is_not_silently_substituted(tmp_path):
    manifest = json.loads(DEFAULT_ARTIFACTS.read_text())
    manifest["artifacts"].pop()
    fake = tmp_path / "missing_component.json"
    fake.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="four models"):
        artifact_inventory(fake)


def test_inference_runtime_is_explicit_cpu_and_records_compatibility_warnings(real_twin):
    metadata = real_twin.manifest
    runtime = metadata["runtime"]
    assert all("CPU" in device for device in runtime["visible_devices"])
    assert runtime["tensor_float_32_enabled"] is False
    assert runtime["intra_op_threads"] == runtime["inter_op_threads"] == 1
    assert "training=False" in runtime["model_call"]
    assert runtime["serialized_tensorflow_version"] is None
    mismatched_scalers = sum(i["serialization"]["sklearn_version"] != runtime["dependencies"]["scikit-learn"]
                            for i in metadata["artifacts"] if i["role"] != "model")
    assert len([w for w in metadata["load_warnings"] if w["category"] == "InconsistentVersionWarning"]) == mismatched_scalers


@pytest.mark.parametrize("total_load", [40.0, 100.0, 200.0, 400.0])
def test_balanced_load_scales_power_and_keeps_per_instance_qos(real_twin, total_load):
    single = real_twin.outcome(1, total_load / 2)
    double = real_twin.outcome(2, total_load)
    raw = real_twin.raw_prediction(2, total_load)
    assert raw.per_instance_load_mbps == total_load / 2
    assert raw.instance_count == 2
    assert double.power_w == 2 * single.power_w
    assert double.qos == single.qos


@pytest.mark.parametrize("action", [0, 1, 2])
def test_outputs_match_independently_applied_original_transform_chain(real_twin, action):
    np = real_twin._np
    mode = "dpdk" if action == 0 else "usr"
    k = max(1, action)
    values = {}
    logged = np.log1p(np.asarray([[100.0 / k]], dtype=np.float32))
    for target in ("power", "qos"):
        components = real_twin._components
        scaled = components[mode, target, "input_scaler"].transform(logged.copy())
        network = components[mode, target, "model"](scaled, training=False).numpy()
        values[target] = float(components[mode, target, "target_scaler"].inverse_transform(network)[0, 0])
    outcome = real_twin.outcome(action, 100)
    assert (outcome.power_w, outcome.qos) == (k * values["power"], values["qos"])
    assert math.isfinite(outcome.power_w) and outcome.power_w >= 0
    assert 0 <= outcome.qos <= 1


def test_uncached_repeats_and_reloading_are_bitwise_consistent(real_twin):
    inputs = [(0, 100.0), (1, 100.0), (2, 100.0)]
    first = [real_twin.outcome(*x) for x in inputs]
    weights = [a.copy() for key, model in real_twin._components.items() if key[2] == "model" for a in model.get_weights()]
    real_twin.clear_cache()
    assert [real_twin.outcome(*x) for x in reversed(inputs)][::-1] == first
    other = FittedUPFTwin()
    assert other.identity == real_twin.identity
    assert [other.outcome(*x) for x in inputs] == first
    after = [a for key, model in real_twin._components.items() if key[2] == "model" for a in model.get_weights()]
    assert all(real_twin._np.array_equal(a, b) for a, b in zip(weights, after))
    real_twin.verify_artifacts()


def test_fresh_process_prediction_and_identity_match(real_twin):
    code = (
        "from revision.evaluation.twin import FittedUPFTwin; import json; "
        "from dataclasses import asdict; t=FittedUPFTwin(); "
        "print(json.dumps({'identity':t.identity,'outcomes':[asdict(t.outcome(a,100.0)) for a in (0,1,2)]}))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    assert data["identity"] == real_twin.identity
    assert data["outcomes"] == [asdict(real_twin.outcome(a, 100.0)) for a in (0, 1, 2)]


@pytest.mark.parametrize("action,load", [(1, 2.0), (2, 10.0)])
def test_known_negative_power_is_reported_and_rejected_without_repair(real_twin, action, load):
    raw = real_twin.raw_prediction(action, load)
    assert raw.power_w < 0
    assert "nonfinite_or_negative_attributed_power" in raw.failures
    with pytest.raises(InvalidPrediction) as error:
        real_twin.outcome(action, load)
    assert error.value.raw == raw


@pytest.mark.parametrize("power,qos", [(float("nan"), .9), (1.0, 1.01), (1.0, -.01)])
def test_invalid_raw_outputs_are_not_clipped(real_twin, monkeypatch, power, qos):
    monkeypatch.setattr(real_twin, "_predict_instance", lambda mode, load: (power, qos))
    with pytest.raises(InvalidPrediction) as error:
        real_twin.outcome(1, 100)
    if math.isnan(power):
        assert math.isnan(error.value.raw.power_w)
    else:
        assert error.value.raw.qos == qos


@pytest.mark.parametrize("action,load", [(3, 100), (True, 100), (1, -1), (1, float("nan"))])
def test_invalid_model_inputs_fail(real_twin, action, load):
    with pytest.raises(ValueError):
        real_twin.outcome(action, load)


def test_static_configuration_identity_and_reset_are_deterministic(prepared):
    from revision.evaluation.contract import observe
    h = prepared.horizons["validation"]
    observation = observe(h.context(15), "instant", State(0, 4), None, None)
    identities = []
    for action in (0, 1, 2):
        controller = StaticController(action)
        identities.append(controller.identity)
        for seed in (42, 123):
            controller.reset(seed=seed)
            assert controller.predict(observation, deterministic=True) == action
        with pytest.raises(ValueError):
            controller.predict(observation, deterministic=False)
    assert len(set(identities)) == 3


def test_real_validation_horizons_energy_events_and_repeated_evaluation(real_validation, real_static_run):
    horizon, contract, _ = real_validation
    report, predictions, records = real_static_run
    assert report["all_three_successful"]
    assert report["final_test_evaluated"] is False
    assert len(horizon.indices) == 504
    common_times = None
    for action, label in [(0, "DPDK"), (1, "USR(1)"), (2, "USR(2)")]:
        rows = [r for r in records if r["configuration"] == label]
        pred = [r for r in predictions if r["configuration"] == label]
        times = tuple(r["timestamp"] for r in rows)
        if common_times is None:
            common_times = times
        assert times == common_times == tuple(t.isoformat() for t in horizon.timestamps)
        assert all(not r["failures"] for r in pred)
        assert all(r["executed_action"] == action for r in rows)
        assert all(not r["blocked"] for r in rows)
        assert rows[0]["previous_action"] == 0
        assert sum(r["type_switch"] for r in rows) == int(action != 0)
        assert sum(r["usr_instances_launched"] for r in rows) == action
        assert not any(r["scale_up"] or r["scale_down"] for r in rows)
        assert report["baselines"][label]["energy_wh"] == math.fsum(r["power_w"] * .25 for r in rows)
        assert report["baselines"][label]["deterministic_repeat_equal"] is True
        if action:
            assert [r["next_cooldown_elapsed_slots"] for r in rows[:5]] == [1, 2, 3, 4, 4]
            assert rows[0]["orchestration_term"] == action * contract.usr_launch_cost
    assert common_times[0] == "2019-04-22T00:00:00"
    assert common_times[-1] == "2019-04-27T05:45:00"


@pytest.mark.parametrize("role", ["train", "final_test"])
def test_task2a_runner_refuses_non_validation_before_any_model_inference(prepared, contract, role):
    class ForbiddenTwin:
        def raw_prediction(self, *args):
            raise AssertionError("Must reject this horizon before calling models")

    with pytest.raises(ValueError, match="VALIDATION only"):
        run_static_validation(ForbiddenTwin(), prepared.horizons[role], contract, "a" * 64)


def test_static_adapter_respects_cooldown_instead_of_overriding_environment(contract):
    initial_locked = State(0, 0)
    blocked = actuate(initial_locked, StaticController(2).action, contract)
    assert blocked.blocked and blocked.executed == 0
    assert blocked.usr_instances_launched == 0


def test_invalid_validation_never_produces_partial_energy_totals(real_twin, prepared, contract, monkeypatch):
    monkeypatch.setattr(real_twin, "_predict_instance", lambda mode, load: (float("nan"), .9))
    report, predictions, records = run_static_validation(real_twin, prepared.horizons["validation"],
                                                         contract, "a" * 64)
    assert not report["all_three_successful"]
    assert predictions and not records
    for result in report["baselines"].values():
        assert result["status"] == "failed_invalid_predictions"
        assert "energy_wh" not in result
        assert result["invalid_slots"] == len(prepared.horizons["validation"].indices)
    encoded = json.dumps(diagnostic_json(report), allow_nan=False)
    assert '"nonfinite_float": "nan"' in encoded
