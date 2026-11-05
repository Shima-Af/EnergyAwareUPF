"""Post-selection diagnostics: train/validation input census and static validation.

This entry point does not fit or select a power model and has no final-test option.
"""

import argparse
import csv
import json
from pathlib import Path

from .profiling_validity import workload_census
from revision.evaluation.diagnostic_power_twin import DiagnosticPowerTwin
from revision.evaluation.twin import ROOT, FittedUPFTwin, sha256
from revision.evaluation.validate_static import DEFAULT_PROTOCOL, run_static_validation, validation_inputs, write_csv
from revision.profiling.refit_usr_power import write_json


def run(model_path: Path, selection_path: Path, output: Path):
    original = FittedUPFTwin()
    diagnostic = DiagnosticPowerTwin(original, model_path, selection_path)
    workload_results = workload_census(diagnostic, output, DEFAULT_PROTOCOL)
    write_json(output / 'input_validity_census.json', workload_results)
    horizon, contract, metadata = validation_inputs()
    summary, predictions, records = run_static_validation(diagnostic, horizon, contract, metadata['protocol_sha256'])
    prior_path = ROOT / 'revision/reports/task2a/validation/summary.json'
    prior = json.loads(prior_path.read_text())
    if prior['twin_id'] != original.identity or prior['common_horizon_sha256'] != horizon.identity:
        raise ValueError('Task 2A baseline reference has a different original twin or validation horizon')
    prior_rows_path = ROOT / 'revision/reports/task2a/validation/steps.csv'
    with prior_rows_path.open() as handle:
        prior_rows = {(r['configuration'], r['timestamp']): r for r in csv.DictReader(handle)}
    for row in records:
        before = prior_rows[row['configuration'], row['timestamp']]
        if row['qos'] != float(before['qos']):
            raise ValueError('QoS changed despite power-only replacement')
        if row['configuration'] == 'DPDK' and row['power_w'] != float(before['power_w']):
            raise ValueError('DPDK power changed despite USR-only replacement')
    changes = {}
    for label, result in summary['baselines'].items():
        before = prior['baselines'][label]
        if result['status'] != 'success':
            raise ValueError('Invalid replacement validation outcome; no default promotion')
        for event in ('type_switches', 'scale_up_events', 'scale_down_events', 'blocked_requests', 'usr_instances_launched'):
            if result[event] != before[event]:
                raise ValueError('Actuation contract changed during power-only diagnostic')
        changes[label] = {
            'original_energy_wh': before['energy_wh'], 'replacement_energy_wh': result['energy_wh'],
            'energy_change_wh': result['energy_wh'] - before['energy_wh'],
            'energy_change_percent': 100 * (result['energy_wh'] - before['energy_wh']) / before['energy_wh'],
            'original_mean_sec_j_per_mbit': before['mean_sec_j_per_mbit'],
            'replacement_mean_sec_j_per_mbit': result['mean_sec_j_per_mbit'],
            'original_qos_violation_rate': before['qos_violation_rate'],
            'replacement_qos_violation_rate': result['qos_violation_rate'],
            'qos_values_identical_all_slots': True,
        }
    summary.update({'evaluation': metadata, 'twin_id': diagnostic.identity,
                    'task2a_reference_path': str(prior_path), 'task2a_reference_sha256': sha256(prior_path),
                    'changes_from_task2a': changes, 'default_twin_replaced': False,
                    'model_selection_used_these_results': False})
    write_json(output / 'static_validation_summary.json', summary)
    write_json(output / 'diagnostic_twin_manifest.json', diagnostic.manifest)
    write_csv(output / 'static_validation_predictions.csv', predictions)
    write_csv(output / 'static_validation_steps.csv', records)
    diagnostic.verify_artifacts()
    print(json.dumps(changes, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / 'revision'):
        raise ValueError('Diagnostic outputs must stay under revision/')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run(args.model, args.selection, args.output_dir)


if __name__ == '__main__':
    main()
