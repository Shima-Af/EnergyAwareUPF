"""Inspect timelines and emit preparation metadata, never policy/twin outcomes.

Run from the repository root with ``python -m revision.evaluation.inspect_timeline``.
Only an explicitly requested, new file under revision/ may be written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from .contract import Contract
from .data import DataSpec, LoadScale, SplitPlan, Timeline, prepare

ROOT = Path(__file__).resolve().parents[2]


def inspect(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    if (config["load_scale"]["fit_dataset"] != "netflix"
            or config["load_scale"]["fit_partition"] != "train"
            or config["load_scale"]["clip"] is not False
            or config["input_normalization"] != "training_minmax_no_clipping"
            or config["timestamp_semantics"] != "interval_start_source_clock_timezone_unverified"
            or config["transfer"] != "youtube_is_reporting_only_reuse_netflix_transforms_no_fitting_or_selection"
            or config["selection_objective"] != "total_reward"):
        raise ValueError("Unsupported protocol setting; implementation and documentation must agree")
    plan = SplitPlan(**{k: datetime.fromisoformat(v) for k, v in config["split"].items()})
    spec = DataSpec(**config["data_spec"])
    contract = Contract(**config["contract"])
    timelines = {name: Timeline.read_csv(ROOT / path, name, timedelta(minutes=config["slot_minutes"]),
                                       usage="transfer" if name == "youtube" else "development")
                 for name, path in config["datasets"].items()}
    scale = LoadScale.fit_training(plan.partition(timelines["netflix"])["train"],
                                   config["load_scale"]["reference_peak_mbps"])
    main = prepare(timelines["netflix"], plan, spec, scale=scale)
    transfer = prepare(timelines["youtube"], plan, spec, scale=scale, normalizer=main.normalizer)
    if timelines["netflix"].timestamps != timelines["youtube"].timestamps:
        raise ValueError("Transfer and source timeline timestamps differ")
    report = {
        "protocol_id": config["protocol_id"],
        "artifact_kind": "data_timeline_inspection_not_experimental_results",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "timestamp_semantics": config["timestamp_semantics"],
        "warmup_completed_slots": spec.warmup,
        "load_scale_multiplier": scale.multiplier,
        "scale_fitted_through": scale.fit.last_training_label.isoformat(),
        "normalizer": {"minimum_mbps": main.normalizer.minimum, "span_mbps": main.normalizer.span,
                       "fitted_through": main.normalizer.fit.last_training_label.isoformat()},
        "contract": contract.__dict__,
        "selection_objective": config["selection_objective"],
        "datasets": {},
        "archived_csv_inventory": [],
    }
    for name, data in [("netflix", main), ("youtube", transfer)]:
        raw = timelines[name]
        dataset = {"source": config["datasets"][name], "source_sha256": raw.source_sha256,
                   "rows": len(raw.timestamps), "first_timestamp": raw.timestamps[0].isoformat(),
                   "last_timestamp": raw.timestamps[-1].isoformat(),
                   "end_exclusive": (raw.timestamps[-1] + raw.slot).isoformat(),
                   "raw_aggregate_peak_mbps": max(raw.load_mbps),
                   "role": "development_and_reporting" if name == "netflix" else "transfer_reporting_only",
                   "partitions": {}}
        for role, p in data.partitions.items():
            h = data.horizons[role]
            dataset["partitions"][role] = {
                "source_indices_half_open": [p.start, p.stop], "label_rows": p.stop - p.start,
                "label_start": p.timestamps[0].isoformat(), "label_last": p.timestamps[-1].isoformat(),
                "evaluation_rows": len(h.indices), "evaluation_start": h.timestamps[0].isoformat(),
                "evaluation_last": h.timestamps[-1].isoformat(),
                "horizon_sha256": h.identity,
                "scaled_peak_mbps": max(p.loads),
                "above_reference_400_mbps": sum(x > 400 for x in p.loads),
            }
        report["datasets"][name] = dataset
    for path in sorted((ROOT / "data/processed").glob("*.csv")):
        rows = list(csv.DictReader(path.read_text().splitlines()))
        timeline = Timeline.read_csv(path, path.stem, timedelta(minutes=config["slot_minutes"]))
        predictions = [r.get("Traffic_Predicted_Mbps", "") for r in rows]
        first = next((i for i, v in enumerate(predictions) if v and v.lower() != "nan"), None)
        report["archived_csv_inventory"].append({
            "path": str(path.relative_to(ROOT)), "sha256": timeline.source_sha256,
            "rows": len(rows), "first_timestamp": timeline.timestamps[0].isoformat(),
            "last_timestamp": timeline.timestamps[-1].isoformat(),
            "raw_workload_sha256": hashlib.sha256(json.dumps(
                [t.isoformat() for t in timeline.timestamps] + list(timeline.load_mbps)).encode()).hexdigest(),
            "first_archived_forecast_row": first,
            "archived_forecast_used": False,
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "revision/configs/evaluation_v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contents = json.dumps(inspect(args.config), indent=2) + "\n"
    if args.output is None:
        print(contents, end="")
    else:
        if not args.output.resolve().is_relative_to(ROOT / "revision"):
            raise ValueError("Inspection outputs must be new files under revision/")
        with args.output.open("x") as handle:
            handle.write(contents)


if __name__ == "__main__":
    main()
