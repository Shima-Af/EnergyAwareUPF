"""Summarize controlled profiling comparisons without running any inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .twin import ROOT

VARIANTS = ("cpu_scalar", "cpu_predict32", "cpu_onednn_predict32",
            "compat_sklearn172", "compat_keras3113", "compat_both")


def summarize(directory: Path) -> dict:
    summaries, rows = {}, {}
    for variant in VARIANTS:
        summaries[variant] = json.loads((directory / variant / "summary.json").read_text())
        with (directory / variant / "predictions.csv").open() as handle:
            rows[variant] = {(r["mode"], r["target"], r["sample_idx"]): r for r in csv.DictReader(handle)}
    baseline = summaries["cpu_predict32"]
    for summary in summaries.values():
        if (summary["artifact_sha256"] != baseline["artifact_sha256"]
                or summary["reference_sha256"] != baseline["reference_sha256"]):
            raise ValueError("Numerical comparisons require identical artifacts and reference data")
    base_rows = rows["cpu_predict32"]
    comparisons = {}
    for name, indexed in rows.items():
        if indexed.keys() != base_rows.keys():
            raise ValueError("Profiling comparison rows differ")
        groups = {}
        for mode, target in (("dpdk", "power"), ("dpdk", "qos"), ("usr", "power"), ("usr", "qos")):
            keys = [k for k in indexed if k[:2] == (mode, target)]
            if any(indexed[k]["offered_load_mbps"] != base_rows[k]["offered_load_mbps"] for k in keys):
                raise ValueError("Profiling comparisons require identical input loads")
            differences = [float(indexed[k]["fresh_prediction"]) - float(base_rows[k]["fresh_prediction"])
                           for k in keys]
            groups[f"{mode}_{target}"] = {
                "max_abs_difference_from_current_cpu_batch32": max(abs(x) for x in differences),
                "changed_rows": sum(x != 0 for x in differences),
            }
        comparisons[name] = groups
    invalid = [r for r in rows["cpu_scalar"].values() if r["invalid_prediction"] == "True"]
    gpu_log = directory / "gpu_predict32_tf32_on.log"
    gpu_summary = directory / "gpu_predict32_tf32_on/summary.json"
    if gpu_summary.exists():
        gpu_attempt = json.loads(gpu_summary.read_text())
    elif gpu_log.exists():
        failure = next((line for line in reversed(gpu_log.read_text().splitlines()) if "Error:" in line), "unknown")
        gpu_attempt = {"status": "failed_before_predictions", "profile": "gpu_predict32_tf32_on",
                       "error": failure, "log": str(gpu_log), "cpu_fallback_used": False}
    else:
        gpu_attempt = {"status": "not_tested", "profile": "gpu_predict32_tf32_on"}
    gpu_attempt["unexecuted_profiles"] = [p for p in ("gpu_predict32_tf32_off", "gpu_scalar_tf32_off")
                                         if not (directory / p / "summary.json").exists()]
    return {
        "reference_kind": "profiling measurement prediction archive; no policy final-test outputs consumed",
        "reference_path": baseline["reference_path"], "reference_sha256": baseline["reference_sha256"],
        "completed_profiles": {name: summary["dependencies"] for name, summary in summaries.items()},
        "controlled_comparisons": comparisons,
        "production_cpu_scalar_vs_archive": summaries["cpu_scalar"]["groups"],
        "invalid_profiled_outputs": {
            "rows": len(invalid), "mode_target": "usr_power",
            "unique_per_instance_loads_mbps": sorted({float(r["offered_load_mbps"]) for r in invalid}),
            "negative_predictions_already_in_archive": sum(float(r["archived_prediction"]) < 0 for r in invalid),
        },
        "gpu_attempt": gpu_attempt,
        "conclusions": {
            "keras3113_identical_to_current_cpu_batch32": all(
                g["changed_rows"] == 0 for g in comparisons["compat_keras3113"].values()),
            "root_cause_of_larger_archive_gap": "Unresolved; tested CPU kernels and serializer versions do not close it",
            "default_backend_selection": "Fixed deterministic CPU single-sample profile; not selected by archival agreement",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.directory), indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        if not args.output.resolve().is_relative_to(ROOT / "revision"):
            raise ValueError("Outputs must be new files under revision/")
        with args.output.open("x") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
