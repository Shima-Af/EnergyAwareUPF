"""Three frozen power candidates, selected by profiling validation only.

No traffic data are imported by this fitting/selection entry point. Original-model
comparisons and internal-test scoring happen after selection is saved to disk.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .power_model import LOAD, POWER, ProfilingPartition, PowerModel, fit_candidate, metrics, partition_sweeps, select_candidate
from revision.evaluation.twin import ROOT, FittedUPFTwin, sha256
from revision.evaluation.validate_static import write_csv

DEFAULT_CONFIG = ROOT / 'revision/configs/usr_power_refit_v1.json'
CAMPAIGN = Path('/home/ubuntu/UpfProfilingCampaign/UpfProfilingCampaign')


def write_json(path: Path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def verify_provenance(frame, config, output):
    raw_path = CAMPAIGN / 'data/raw/scaphandre/usr/scaph_process_power_consumption_microwatts.csv'
    catalog_path = CAMPAIGN / 'configs/test_cases.csv'
    raw = pd.read_csv(raw_path)
    if raw.pid.nunique() != 1 or set(raw.exe) != {'/openair-upf/bin/oai_upf'}:
        raise ValueError('Raw power process identity requires explicit review')
    raw_times = pd.to_datetime(raw.timestamp, format='ISO8601')
    raw_watts = raw.value.to_numpy(dtype=np.float64) / 1e6
    matches, nearest_equal = [], 0
    for row_id, row in frame.iterrows():
        offsets = (raw_times - row.parsed_timestamp).dt.total_seconds().to_numpy()
        nearest = int(np.argmin(abs(offsets)))
        same = np.isclose(raw_watts, row[POWER], rtol=0, atol=1e-12)
        nearby = np.flatnonzero(same & (abs(offsets) <= 6))
        if len(nearby) == 0:
            raise ValueError(f'Cannot trace processed power to raw microwatts at row {row_id}')
        chosen = int(nearby[np.argmin(abs(offsets[nearby]))])
        is_nearest = bool(same[nearest])
        nearest_equal += is_nearest
        matches.append({'source_row': int(row_id), 'processed_timestamp': row.timestamp,
                        'test_set': int(row['Test Set']), 'test_case': row['Test Case'],
                        'processed_power_w': float(row[POWER]), 'matched_raw_power_w': float(raw_watts[chosen]),
                        'raw_timestamp': raw.timestamp.iloc[chosen],
                        'raw_minus_processed_seconds': float(offsets[chosen]),
                        'nearby_equal_value_matches': len(nearby), 'nearest_timestamp_value_matches': is_nearest})
    catalog = pd.read_csv(catalog_path).set_index('Test Case')
    fields = ['Test Set', LOAD, 'Packet Size', 'Packet Rate']
    for name, block in frame.groupby('Test Case'):
        for field in fields:
            if not np.all(block[field] == catalog.loc[name, field]):
                raise ValueError(f'Case catalog mismatch: {name}, {field}')
    bit_rate_error = np.max(abs(frame['Packet Size'] * 8 * frame['Packet Rate'] / 1e6 - frame[LOAD]))
    if bit_rate_error != 0:
        raise ValueError('Packet size/rate does not support the documented Mbps labels')
    copies = []
    for project in ('xRL_UpfSelection', 'dynamic-upf-selection-rl'):
        path = Path('/home/ubuntu') / project / 'data/processed/oai_processed.csv'
        digest = sha256(path)
        if digest != config['source_sha256']:
            raise ValueError('Recovered profiling copies differ; no implicit source substitution')
        copies.append({'path': str(path), 'sha256': digest})
    write_csv(output / 'power_label_raw_matches.csv', matches)
    return {
        'processed_source': config['source_csv'], 'processed_sha256': config['source_sha256'],
        'matching_copies': copies,
        'raw_source': str(raw_path), 'raw_sha256': sha256(raw_path),
        'raw_rows': len(raw), 'raw_pid': int(raw.pid.iloc[0]), 'raw_executable': '/openair-upf/bin/oai_upf',
        'raw_metric': 'scaph_process_power_consumption_microwatts',
        'target_conversion': 'raw microwatts / 1e6 -> process-attributed watts',
        'all_processed_power_labels_traced': len(matches),
        'nearest_timestamp_values_match': int(nearest_equal),
        'other_nearby_value_matches': len(matches) - int(nearest_equal),
        'max_absolute_matching_timestamp_offset_seconds': max(abs(r['raw_minus_processed_seconds']) for r in matches),
        'match_value_tolerance_w': 1e-12,
        'match_tolerance_note': 'Floating-point serialization check only; not a physical error acceptance threshold',
        'catalog': str(catalog_path), 'catalog_sha256': sha256(catalog_path),
        'all_110_cases_match_catalog': True, 'packet_bitrate_label_max_error_mbps': float(bit_rate_error),
        'input_definition': config['input_units'],
        'uncertainties': [
            'Exact historical LoadCore/Scaphandre join, trimming and aggregation code not authenticated',
            'Some labels match a previous rather than the closest raw sample; zeros can match multiple samples',
            'Input is the campaign DL/UL objective, not an authenticated aggregate DL+UL variable; historical mapping retained',
            'No new physical campaign, packet-size diversity, true idle point, or calibrated metering uncertainty',
            'Grouped limited exports are not substituted for the source CSV',
        ],
    }


def artifact(model, training, config, config_path):
    return {
        'schema_version': 1, 'target_units': 'process_attributed_W', 'input_units': config['input_units'],
        'model': asdict(model), 'target_transform': 'identity_watts_no_target_scaler',
        'input_transform': 'float64 Mbps; affine scale from training maximum only; PWL linear load coordinates',
        'training_sweeps': list(training.sweeps), 'training_rows': len(training.frame),
        'training_source_rows': [int(i) for i in training.frame.index],
        'source_path': config['source_csv'], 'source_sha256': config['source_sha256'],
        'config_path': str(config_path.resolve()), 'config_sha256': sha256(config_path),
        'fit_objective': config['fit_objective'], 'solver': config['solver'],
        'dependencies': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'threadpoolctl')},
        'solver_blas_threads': 1,
        'python': platform.python_version(), 'implementation_sha256': sha256(Path(__file__).with_name('power_model.py')),
        'no_output_clipping': True, 'no_extrapolation': True,
    }


def run(config_path: Path, output: Path, model_dir: Path):
    config = json.loads(config_path.read_text())
    if (config['selection_objective'] != 'validation_row_rmse_w'
            or config['fit_objective'] != 'unweighted_training_row_squared_error_in_watts'
            or config['solver'] != 'scipy.optimize.nnls'
            or config['refit_on_train_plus_validation'] or config['default_twin_replacement']
            or config['traffic_results_used_for_selection']):
        raise ValueError('Unsupported or leakage-prone profiling protocol')
    source = Path(config['source_csv'])
    if sha256(source) != config['source_sha256']:
        raise ValueError('Original profiling source hash changed')
    frame = pd.read_csv(source)
    frame['parsed_timestamp'] = pd.to_datetime(frame.timestamp, format='ISO8601')
    if not np.isfinite(frame[[LOAD, POWER]].to_numpy()).all() or (frame[POWER] < 0).any():
        raise ValueError('Invalid source measurements require review, not automatic repair')
    provenance = verify_provenance(frame, config, output)
    write_json(output / 'measurement_provenance.json', provenance)
    parts = partition_sweeps(frame, config['split'])
    split_rows = [{'source_row': int(i), 'role': role, 'sweep': int(row['Test Set']),
                   'test_case': row['Test Case'], 'timestamp': row.timestamp, 'load_mbps': float(row[LOAD])}
                  for role, p in parts.items() for i, row in p.frame.iterrows()]
    write_csv(output / 'profiling_split.csv', split_rows)
    split_summary = {r: {'sweeps': list(p.sweeps), 'rows': len(p.frame),
                         'case_blocks': int(p.frame['Test Case'].nunique()),
                         'load_levels': int(p.frame[LOAD].nunique()),
                         'first_timestamp': p.frame.timestamp.min(), 'last_timestamp': p.frame.timestamp.max()}
                     for r, p in parts.items()}
    write_json(output / 'profiling_protocol.json', {'config': config, 'config_sha256': sha256(config_path),
                                                  'partitions': split_summary,
                                                  'new_independent_campaign': False})
    models = tuple(fit_candidate(parts['train'], spec, tuple(config['domain_mbps'])) for spec in config['candidates'])
    for model in models:
        write_json(model_dir / f'{model.name}.json', artifact(model, parts['train'], config, config_path))
    selected, validation_scores = select_candidate(models, parts['validation'])
    selected_path = model_dir / 'selected_usr_power.json'
    write_json(selected_path, artifact(selected, parts['train'], config, config_path))
    # This file is written before internal-test scoring, original-model comparison,
    # uncertainty analysis or any traffic/controller entry point is invoked.
    selection = {'selected_name': selected.name, 'selected_model_path': str(selected_path.resolve()),
                 'selected_model_sha256': sha256(selected_path), 'validation_scores': validation_scores,
                 'objective': config['selection_objective'], 'training_sweeps': list(parts['train'].sweeps),
                 'validation_sweeps': list(parts['validation'].sweeps),
                 'internal_test_sweeps': list(parts['internal_test'].sweeps),
                 'internal_test_scored_before_selection': False, 'traffic_used_for_selection': False,
                 'refit_on_train_plus_validation': False, 'config_sha256': sha256(config_path)}
    write_json(output / 'selection_frozen.json', selection)
    print('Frozen profiling validation selection:', selected.name, flush=True)
    all_models = list(models)
    base = FittedUPFTwin()

    class Original:
        name = 'original_saved_usr_power_descriptive_only'
        def predict(self, loads):
            return np.asarray([base.raw_prediction(1, float(x)).power_w for x in loads])

    all_models.append(Original())
    summary, predictions, by_load, by_sweep = {}, [], [], []
    for model in all_models:
        summary[model.name] = {}
        for role, part in parts.items():
            summary[model.name][role] = metrics(model, part.frame, config['low_load_max_mbps'])
            predicted = model.predict(part.frame[LOAD].to_numpy())
            for (index, row), value in zip(part.frame.iterrows(), predicted):
                predictions.append({'model': model.name, 'role': role, 'source_row': int(index),
                                    'sweep': int(row['Test Set']), 'test_case': row['Test Case'],
                                    'load_mbps': float(row[LOAD]), 'measured_power_w': float(row[POWER]),
                                    'predicted_power_w': float(value), 'error_w': float(value - row[POWER])})
            for load, group in part.frame.groupby(LOAD):
                by_load.append({'model': model.name, 'role': role, 'load_mbps': float(load),
                                **metrics(model, group, config['low_load_max_mbps'])})
            for sweep, group in part.frame.groupby('Test Set'):
                by_sweep.append({'model': model.name, 'role': role, 'sweep': int(sweep),
                                 **metrics(model, group, config['low_load_max_mbps'])})
    write_csv(output / 'profiling_predictions.csv', predictions)
    write_csv(output / 'errors_by_load.csv', by_load)
    write_csv(output / 'errors_by_sweep.csv', by_sweep)
    selected_spec = next(s for s in config['candidates'] if s['name'] == selected.name)
    jackknife = []
    for omitted in parts['train'].sweeps:
        remaining = tuple(s for s in parts['train'].sweeps if s != omitted)
        training = ProfilingPartition('train', remaining, parts['train'].frame[parts['train'].frame['Test Set'].isin(remaining)])
        jackknife.append(fit_candidate(training, selected_spec, tuple(config['domain_mbps'])))
    domain = tuple(config['domain_mbps'])
    grid = np.unique(np.concatenate([np.geomspace(*domain, 10001), np.linspace(domain[0], domain[1], 10001),
                                     np.arange(.01, 20.001, .01), np.asarray(selected.knots_mbps)]))
    # Floating-point grid generation can overstep 20, but never the overall domain.
    physical_checks = {m.name: {'domain_mbps': domain, 'grid_samples': len(grid),
                               'minimum_power_w': float(np.min(m.predict(grid))),
                               'finite_and_nonnegative': bool(np.isfinite(m.predict(grid)).all() and (m.predict(grid) >= 0).all()),
                               'continuous_domain_guarantee': 'nonnegative affine coefficients and nonnegative loads'
                               if m.family == 'affine' else 'convex interpolation of nonnegative knot powers'} for m in models}
    uncertainty = []
    for load in sorted(frame[LOAD].unique()):
        band = [float(m.predict([load])[0]) for m in jackknife]
        means = parts['train'].frame[parts['train'].frame[LOAD] == load].groupby('Test Set')[POWER].mean()
        uncertainty.append({'load_mbps': float(load), 'selected_power_w': float(selected.predict([load])[0]),
                            'training_sweep_mean_min_w': float(means.min()), 'training_sweep_mean_max_w': float(means.max()),
                            'training_sweep_mean_sd_w': float(means.std()),
                            'jackknife_min_w': min(band), 'jackknife_max_w': max(band), 'training_sweeps': len(means)})
    write_csv(output / 'training_uncertainty.csv', uncertainty)
    low_rows = []
    for load in np.linspace(.01, 20, 2000):
        band = [float(m.predict([load])[0]) for m in jackknife]
        low_rows.append({'load_mbps': float(load), 'selected_power_w': float(selected.predict([load])[0]),
                         'jackknife_min_w': min(band), 'jackknife_max_w': max(band),
                         'original_power_w': base.raw_prediction(1, float(load)).power_w})
    write_csv(output / 'low_load_comparison.csv', low_rows)
    write_json(output / 'metrics.json', {'selected_name': selected.name, 'candidates': summary,
                                        'physical_checks': physical_checks,
                                        'uncertainty': 'Training sweep jackknife envelope: three refits, not calibrated confidence/prediction intervals',
                                        'original_comparison': 'Descriptive only; original model trained on historical row split including all sweeps',
                                        'original_twin_id': base.identity})
    write_json(output / 'original_artifact_manifest.json', base.manifest)
    base.verify_artifacts()
    print('Internal profiling test and descriptive original comparison complete; no traffic used.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args()
    for directory in (args.output_dir, args.model_dir):
        if not directory.resolve().is_relative_to(ROOT / 'revision'):
            raise ValueError('New model/diagnostic outputs must stay under revision/')
        directory.mkdir(parents=True, exist_ok=False)
    run(args.config, args.output_dir, args.model_dir)


if __name__ == '__main__':
    main()
