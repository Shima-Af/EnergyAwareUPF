"""Run the three static configurations on Netflix validation only, never final test."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from .contract import Contract, evaluate, require_comparable
from .data import DataSpec, EvaluationHorizon, LoadScale, SplitPlan, Timeline, fingerprint, prepare
from .selection import Candidate
from .static import StaticController
from .twin import DEFAULT_ARTIFACTS, ROOT, FittedUPFTwin, sha256

DEFAULT_PROTOCOL = ROOT / "revision/configs/evaluation_v1.json"
LABELS = {0: "DPDK", 1: "USR(1)", 2: "USR(2)"}


def diagnostic_json(value):
    """Encode nonfinite failures explicitly without altering inference/accounting."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {k: diagnostic_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json(v) for v in value]
    return value


def validation_inputs(config_path: Path = DEFAULT_PROTOCOL):
    config = json.loads(config_path.read_text())
    if (config["load_scale"]["fit_dataset"] != "netflix"
            or config["load_scale"]["fit_partition"] != "train"
            or config["load_scale"]["clip"] is not False):
        raise ValueError("Validation requires the approved Netflix training-only scaling policy")
    plan = SplitPlan(**{k: datetime.fromisoformat(v) for k, v in config["split"].items()})
    raw = Timeline.read_csv(ROOT / config["datasets"]["netflix"], "netflix",
                            timedelta(minutes=config["slot_minutes"]))
    scale = LoadScale.fit_training(plan.partition(raw)["train"],
                                   config["load_scale"]["reference_peak_mbps"])
    data = prepare(raw, plan, DataSpec(**config["data_spec"]), scale=scale)
    horizon = data.horizons["validation"]
    contract = Contract(**config["contract"])
    metadata = {
        "protocol_path": str(config_path.resolve()), "protocol_sha256": sha256(config_path),
        "dataset_path": str((ROOT / config["datasets"]["netflix"]).resolve()),
        "dataset_sha256": raw.source_sha256, "partition": "validation",
        "horizon_sha256": horizon.identity, "slots": len(horizon.indices),
        "first_timestamp": horizon.timestamps[0].isoformat(),
        "last_timestamp": horizon.timestamps[-1].isoformat(),
        "end_exclusive": (horizon.timestamps[-1] + raw.slot).isoformat(),
        "slot_seconds": raw.slot.total_seconds(),
        "load_scale_multiplier": scale.multiplier,
        "load_scale_fit_last_label": scale.fit.last_training_label.isoformat(),
        "initial_configuration": "DPDK", "initial_cooldown_elapsed_slots": contract.cooldown_slots,
        "contract": asdict(contract), "schema": "instant",
        "final_test_evaluated": False, "youtube_evaluated": False,
    }
    return horizon, contract, metadata


def run_static_validation(twin: FittedUPFTwin, horizon: EvaluationHorizon, contract: Contract,
                          protocol_sha256: str):
    if horizon.partition.role != "validation" or horizon.partition.timeline.usage != "development":
        raise ValueError("Task 2A runner accepts development VALIDATION only")
    predictions, records, results, summaries = [], [], [], {}
    for action, label in LABELS.items():
        failures = []
        for index in horizon.indices:
            load = horizon.partition.timeline.load_mbps[index]
            raw = twin.raw_prediction(action, load)
            row = {"timestamp": horizon.partition.timeline.timestamps[index].isoformat(),
                   "source_index": index, "configuration": label, **asdict(raw),
                   "failures": ";".join(raw.failures)}
            predictions.append(row)
            if raw.failures:
                failures.append(row)
        controller = StaticController(action)
        candidate = Candidate(f"static_{action}", (
            ("twin_bundle_and_runtime", twin.identity), ("controller", controller.identity),
            ("protocol", protocol_sha256), ("contract", fingerprint(asdict(contract))),
        ))
        if failures:
            summaries[label] = {"status": "failed_invalid_predictions", "candidate_id": candidate.identity,
                                "invalid_slots": len(failures), "failures": failures}
            continue  # No accounting for invalid outputs, no repair or partial energy total.
        kwargs = dict(candidate=candidate, contract=contract, schema="instant")
        result = evaluate(horizon, controller, twin, **kwargs)
        repeated = evaluate(horizon, controller, twin, **kwargs)
        if result != repeated:
            raise RuntimeError(f"Non-deterministic repeated static evaluation: {label}")
        expected_events = 0 if action == 0 else 1
        if (result.summary["type_switches"] != expected_events
                or result.summary["usr_instances_launched"] != action
                or result.summary["scale_up_events"] != 0 or result.summary["scale_down_events"] != 0
                or result.summary["blocked_requests"] != 0):
            raise RuntimeError(f"Static initial-state/event accounting mismatch: {label}")
        if any(s.transition.executed != action for s in result.steps):
            raise RuntimeError(f"Static action was not executed on every validation slot: {label}")
        expected_energy = math.fsum(s.outcome.power_w * horizon.partition.timeline.slot.total_seconds() / 3600
                                    for s in result.steps)
        if result.summary["energy_wh"] != expected_energy:
            raise RuntimeError("Energy accounting is inconsistent with W × hours")
        summaries[label] = {"status": "success", "candidate_id": candidate.identity,
                            "deterministic_repeat_equal": True, **result.summary,
                            "min_power_w": min(s.outcome.power_w for s in result.steps),
                            "max_power_w": max(s.outcome.power_w for s in result.steps),
                            "min_qos": min(s.outcome.qos for s in result.steps),
                            "max_qos": max(s.outcome.qos for s in result.steps)}
        results.append(result)
        for step in result.steps:
            records.append({"timestamp": step.context.decision_at.isoformat(), "source_index": step.index,
                            "configuration": label, "load_mbps": step.load_mbps,
                            "previous_action": step.transition.previous, "requested_action": step.transition.requested,
                            "executed_action": step.transition.executed,
                            "type_switch": step.transition.type_switch, "scale_up": step.transition.scale_up,
                            "scale_down": step.transition.scale_down,
                            "blocked": step.transition.blocked,
                            "usr_instances_launched": step.transition.usr_instances_launched,
                            "next_cooldown_elapsed_slots": step.transition.next_state.elapsed_slots,
                            **asdict(step.outcome), **asdict(step.accounting)})
    if results:
        require_comparable(tuple(results))
    twin.verify_artifacts()
    return {"baselines": summaries, "all_three_successful": len(results) == 3,
            "common_horizon_sha256": horizon.identity, "final_test_evaluated": False}, predictions, records


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / "revision"):
        raise ValueError("Task 2A outputs must be new files under revision/")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    horizon, contract, metadata = validation_inputs(args.protocol)
    twin = FittedUPFTwin(args.artifacts)
    summary, predictions, records = run_static_validation(twin, horizon, contract, metadata["protocol_sha256"])
    summary["evaluation"] = metadata
    summary["twin_id"] = twin.identity
    for name, content in [("artifact_manifest.json", twin.manifest), ("summary.json", summary)]:
        with (args.output_dir / name).open("x") as handle:
            json.dump(diagnostic_json(content), handle, indent=2, allow_nan=False)
            handle.write("\n")
    write_csv(args.output_dir / "predictions.csv", predictions)
    write_csv(args.output_dir / "steps.csv", records)
    print(json.dumps(diagnostic_json(summary), indent=2, allow_nan=False))
    if not summary["all_three_successful"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
