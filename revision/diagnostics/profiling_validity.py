"""Task 2B: measurements, inference stages, and train/validation validity census.

No fit(), model correction, controller evaluation, or final-test inference occurs.
All outputs require a new directory under revision/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from revision.evaluation.data import DataSpec, LoadScale, SplitPlan, Timeline
from revision.evaluation.twin import ROOT, FittedUPFTwin, RawPrediction, sha256
from revision.evaluation.validate_static import diagnostic_json, write_csv

PROFILING = Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')
TRAINING_SOURCE = Path('/home/ubuntu/xRL_UpfSelection/scripts/PowerPerformance')
DEFAULT_PROTOCOL = ROOT / 'revision/configs/evaluation_v1.json'


def summarize_failures(rows: list[RawPrediction]) -> dict:
    if not rows:
        raise ValueError('Validity denominators must be nonempty')
    invalid = sum(bool(r.failures) for r in rows)
    return {
        'n': len(rows), 'invalid': invalid, 'invalid_fraction': invalid / len(rows),
        'negative_power': sum(math.isfinite(r.power_w) and r.power_w < 0 for r in rows),
        'nonfinite_power': sum(not math.isfinite(r.power_w) for r in rows),
        'qos_out_of_range': sum(math.isfinite(r.qos) and not 0 <= r.qos <= 1 for r in rows),
        'nonfinite_qos': sum(not math.isfinite(r.qos) for r in rows),
        'min_total_load_mbps': min(r.load_mbps for r in rows),
        'max_total_load_mbps': max(r.load_mbps for r in rows),
        'min_per_instance_load_mbps': min(r.per_instance_load_mbps for r in rows),
        'max_per_instance_load_mbps': max(r.per_instance_load_mbps for r in rows),
        'minimum_predicted_power_w': min(r.power_w for r in rows),
    }


def allowed_partition(role: str) -> None:
    if role not in ('train', 'validation'):
        raise ValueError('Task 2B permits training/validation inputs only; no final-test inference')


def measurement_analysis(twin: FittedUPFTwin, output: Path):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    mode_results, levels, blocks, stages = {}, [], [], []
    for mode, prefix, action in [('dpdk', 'dpdk', 0), ('usr', 'oai', 1)]:
        path = PROFILING / f'{prefix}_processed.csv'
        data = pd.read_csv(path)
        load_col, power_col = 'Objective Throughput (DL/UL)', 'power_consumption_watts'
        indices = np.arange(len(data))
        train_idx, validation_idx = train_test_split(indices, test_size=.2, random_state=42, shuffle=True)
        membership = {int(i): 'train' for i in train_idx}
        membership.update({int(i): 'early_stopping_validation' for i in validation_idx})
        train_groups = set(data.iloc[train_idx]['Test Case'].astype(str))
        validation_groups = set(data.iloc[validation_idx]['Test Case'].astype(str))
        near_adjacent = sum(any(j in membership and membership[j] == 'train'
                                and data.iloc[j]['Test Case'] == data.iloc[i]['Test Case']
                                for j in (int(i) - 1, int(i) + 1)) for i in validation_idx)
        validation_neighbors = []
        for group, frame in data.groupby('Test Case'):
            t = pd.Series([datetime.fromisoformat(value) for value in frame.timestamp], index=frame.index)
            training_times = t.loc[[i for i in frame.index if membership[int(i)] == 'train']]
            valid_times = t.loc[[i for i in frame.index if membership[int(i)] != 'train']]
            for ts in valid_times:
                if len(training_times):
                    validation_neighbors.append(float(abs(training_times - ts).min().total_seconds()))
        component = twin._components
        sx, sy = component[mode, 'power', 'input_scaler'], component[mode, 'power', 'target_scaler']
        train = data.iloc[train_idx]
        train_log = np.log1p(train[load_col].to_numpy(dtype=np.float64))
        train_power = train[power_col].to_numpy(dtype=np.float64)
        # Reconstruct sufficient statistics only. No scaler/model is fitted.
        stats_checks = {
            'stored_input_mean': float(sx.mean_[0]), 'reconstructed_input_mean': float(np.mean(train_log)),
            'stored_input_scale': float(sx.scale_[0]), 'reconstructed_input_scale': float(np.std(train_log)),
            'stored_target_mean': float(sy.mean_[0]), 'reconstructed_target_mean': float(np.mean(train_power)),
            'stored_target_scale': float(sy.scale_[0]), 'reconstructed_target_scale': float(np.std(train_power)),
            'stored_training_samples': int(sy.n_samples_seen_), 'reconstructed_training_samples': len(train_idx),
        }
        targets = data[[power_col]].to_numpy(dtype=np.float64)
        roundtrip = sy.inverse_transform(sy.transform(targets.copy()))
        predictions = {float(load): twin.raw_prediction(action, float(load))
                       for load in sorted(data[load_col].unique())}
        for load, frame in data.groupby(load_col, sort=True):
            power = frame[power_col]
            raw = predictions[float(load)]
            levels.append({
                'mode': mode, 'offered_load_mbps': float(load), 'n': len(frame),
                'measured_min_w': float(power.min()), 'measured_q25_w': float(power.quantile(.25)),
                'measured_median_w': float(power.median()), 'measured_mean_w': float(power.mean()),
                'measured_q75_w': float(power.quantile(.75)), 'measured_max_w': float(power.max()),
                'measured_std_w': float(power.std()), 'measured_zero_count': int((power == 0).sum()),
                'predicted_power_w': raw.power_w, 'predicted_qos': raw.qos,
                'prediction_minus_measured_mean_w': raw.power_w - float(power.mean()),
                'invalid_prediction': bool(raw.failures), 'failures': ';'.join(raw.failures),
                'test_sets': int(frame['Test Set'].nunique()),
            })
            logged = np.log1p(np.asarray([[load]], dtype=np.float32))
            scaled = sx.transform(logged.copy())
            network = component[mode, 'power', 'model'](scaled, training=False).numpy()
            zero_threshold = -float(sy.mean_[0]) / float(sy.scale_[0])
            stages.append({
                'mode': mode, 'load_mbps': float(load), 'log1p_load_float32': float(logged[0, 0]),
                'scaled_input_float32': float(scaled[0, 0]), 'standardized_network_output': float(network[0, 0]),
                'standardized_output_for_zero_watts': zero_threshold,
                'inverse_scaled_power_w': float(sy.inverse_transform(network)[0, 0]),
                'target_mean_w': float(sy.mean_[0]), 'target_scale_w': float(sy.scale_[0]),
            })
        for (test_set, test_case, load), frame in data.groupby(['Test Set', 'Test Case', load_col]):
            n_train = sum(membership[int(i)] == 'train' for i in frame.index)
            blocks.append({'mode': mode, 'test_set': int(test_set), 'test_case': test_case,
                           'load_mbps': float(load), 'rows': len(frame), 'train_rows': n_train,
                           'validation_rows': len(frame) - n_train,
                           'measured_mean_power_w': float(frame[power_col].mean()),
                           'first_timestamp': frame.timestamp.min(), 'last_timestamp': frame.timestamp.max()})
        low = data[data[load_col] <= 20]
        prediction_errors = data[power_col].to_numpy() - np.asarray([predictions[float(x)].power_w for x in data[load_col]])
        low_mask = data[load_col].to_numpy() <= 20
        mode_results[mode] = {
            'path': str(path), 'sha256': sha256(path), 'rows': len(data),
            'minimum_measured_power_w': float(data[power_col].min()),
            'negative_measurements': int((data[power_col] < 0).sum()),
            'zero_measurements': int((data[power_col] == 0).sum()),
            'minimum_offered_load_mbps': float(data[load_col].min()),
            'maximum_offered_load_mbps': float(data[load_col].max()),
            'measured_zero_load_rows': int((data[load_col] == 0).sum()),
            'load_levels': int(data[load_col].nunique()), 'test_sets': int(data['Test Set'].nunique()),
            'test_cases': int(data['Test Case'].nunique()),
            'packet_sizes': sorted(int(x) for x in data['Packet Size'].unique()),
            'first_measurement_timestamp': str(data.timestamp.min()),
            'last_measurement_timestamp': str(data.timestamp.max()),
            'low_load_rows_0_to_20_mbps': len(low), 'low_load_levels': int(low[load_col].nunique()),
            'low_load_blocks': int(low['Test Case'].nunique()),
            'all_row_descriptive_rmse_w': float(np.sqrt(np.mean(prediction_errors ** 2))),
            'low_load_descriptive_rmse_w': float(np.sqrt(np.mean(prediction_errors[low_mask] ** 2))),
            'all_row_error_share_from_low_loads': float(np.sum(prediction_errors[low_mask] ** 2) / np.sum(prediction_errors ** 2)),
            'candidate_historical_split': {
                'test_size': .2, 'random_state': 42, 'shuffle': True,
                'training_rows': len(train_idx), 'early_stopping_validation_rows': len(validation_idx),
                'training_case_groups': len(train_groups), 'validation_case_groups': len(validation_groups),
                'case_groups_shared_between_train_and_validation': len(train_groups & validation_groups),
                'test_sets_shared_between_train_and_validation': len(
                    set(data.iloc[train_idx]['Test Set']) & set(data.iloc[validation_idx]['Test Set'])),
                'validation_rows_with_adjacent_training_row_same_case': int(near_adjacent),
                'median_nearest_same_case_training_time_seconds': float(np.median(validation_neighbors)),
                'saved_scaler_sufficient_statistics': stats_checks,
                'power_target_roundtrip_max_error_w': float(np.max(abs(roundtrip - targets))),
                'independent_final_profiling_holdout': False,
            },
        }
        print('Measured data inspected:', mode, len(data), 'rows', flush=True)
    write_csv(output / 'measurement_by_load.csv', levels)
    write_csv(output / 'measurement_by_block.csv', blocks)
    write_csv(output / 'power_inference_stages.csv', stages)
    return mode_results, levels


def grouped_export_checks():
    import numpy as np
    import pandas as pd

    checks = {}
    for mode, prefix in [('dpdk', 'dpdk'), ('usr', 'oai')]:
        processed = pd.read_csv(PROFILING / f'{prefix}_processed.csv')
        raw_path = PROFILING.parent / 'raw' / f'{prefix}_grouped_limited.csv'
        grouped = pd.read_csv(raw_path)
        differences = []
        for _, row in grouped.iterrows():
            matches = processed[(processed['Test Set'] == row['Test_Set'])
                                & np.isclose(processed['Objective Throughput (DL/UL)'], row['Throughput'], atol=0, rtol=1e-12)]
            if len(matches):
                differences.append(float(matches.power_consumption_watts.mean() - row['power_consumption_watt']))
        checks[mode] = {'path': str(raw_path), 'sha256': sha256(raw_path), 'rows': len(grouped),
                        'negative_power_rows': int((grouped.power_consumption_watt < 0).sum()),
                        'matched_load_set_groups': len(differences),
                        'max_abs_difference_from_processed_group_mean_w': max(abs(x) for x in differences)}
    return checks


def workload_census(twin: FittedUPFTwin, output: Path, protocol: Path):
    config = json.loads(protocol.read_text())
    plan = SplitPlan(**{k: datetime.fromisoformat(v) for k, v in config['split'].items()})
    spec = DataSpec(**config['data_spec'])
    raw_main = Timeline.read_csv(ROOT / config['datasets']['netflix'], 'netflix', timedelta(minutes=config['slot_minutes']))
    scale = LoadScale.fit_training(plan.partition(raw_main)['train'], config['load_scale']['reference_peak_mbps'])
    results, details = {}, []
    for dataset, relative in config['datasets'].items():
        raw = Timeline.read_csv(ROOT / relative, dataset, timedelta(minutes=config['slot_minutes']),
                                usage='transfer' if dataset == 'youtube' else 'development')
        parts = plan.partition(scale.transform(raw))
        partition_results = {}
        for role in ('train', 'validation'):
            allowed_partition(role)
            part = parts[role]
            assert part.stop <= parts['final_test'].start
            per_action = {a: [] for a in (0, 1, 2)}
            for action in per_action:
                for index in range(part.start, part.stop):
                    prediction = twin.raw_prediction(action, part.timeline.load_mbps[index])
                    per_action[action].append(prediction)
                    details.append({'dataset': dataset, 'partition': role,
                                    'timestamp': part.timeline.timestamps[index].isoformat(), 'source_index': index,
                                    'scored_by_corrected_protocol': index >= spec.warmup,
                                    **asdict(prediction), 'failures': ';'.join(prediction.failures)})
            reports = {}
            for action, predictions in per_action.items():
                scored_offset = max(part.start, spec.warmup) - part.start
                reports[str(action)] = {'all_source_rows': summarize_failures(predictions),
                                        'scored_rows': summarize_failures(predictions[scored_offset:])}
            partition_results[role] = {
                'source_indices_half_open': [part.start, part.stop],
                'first_timestamp': part.timestamps[0].isoformat(), 'last_timestamp': part.timestamps[-1].isoformat(),
                'any_action_invalid_all_rows': sum(any(per_action[a][i].failures for a in (0, 1, 2))
                                                   for i in range(part.stop - part.start)),
                'actions': reports,
            }
            print('Workload census:', dataset, role, part.stop - part.start, 'rows per action', flush=True)
        results[dataset] = {'source': relative, 'source_sha256': raw.source_sha256,
                            'usage': 'policy_development' if dataset == 'netflix' else 'transfer_context_validity_only_no_fitting',
                            'partitions': partition_results}
    write_csv(output / 'workload_prediction_census.csv', details)
    write_csv(output / 'invalid_workload_predictions.csv', [row for row in details if row['failures']])
    return results


def low_load_grid(twin: FittedUPFTwin, output: Path):
    # Fixed diagnostic resolution, not a proposed operating-domain threshold.
    rows = []
    for j in range(2001):
        raw = twin.raw_prediction(1, j / 100)
        rows.append({'per_instance_load_mbps': j / 100, 'predicted_power_w': raw.power_w,
                     'predicted_qos': raw.qos, 'failures': ';'.join(raw.failures)})
    write_csv(output / 'usr_low_load_grid.csv', rows)
    negative = [r for r in rows if r['predicted_power_w'] < 0]
    return {'range_mbps': [0, 20], 'step_mbps': .01, 'samples': len(rows),
            'negative_samples': len(negative),
            'first_negative_sample_mbps': min(r['per_instance_load_mbps'] for r in negative),
            'last_negative_sample_mbps': max(r['per_instance_load_mbps'] for r in negative),
            'minimum_predicted_power_w': min(r['predicted_power_w'] for r in rows),
            'zero_load_is_extrapolated': True}


def plot_low_load(output: Path, levels: list[dict]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with (output / 'usr_low_load_grid.csv').open() as handle:
        curve = list(csv.DictReader(handle))
    measured = [r for r in levels if r['mode'] == 'usr' and r['offered_load_mbps'] <= 20]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    x = [r['offered_load_mbps'] for r in measured]
    y = [r['measured_mean_w'] for r in measured]
    ax.errorbar(x, y, yerr=([r['measured_mean_w'] - r['measured_min_w'] for r in measured],
                           [r['measured_max_w'] - r['measured_mean_w'] for r in measured]),
                fmt='o', capsize=3, color='#1b7837', label='Measured attributed power: mean and sample range')
    ax.plot([float(r['per_instance_load_mbps']) for r in curve],
            [float(r['predicted_power_w']) for r in curve], color='#2166ac', label='Original saved USR power model')
    ax.axhline(0, color='black', linewidth=.8)
    ax.axhspan(-.07, 0, color='#d6604d', alpha=.12, label='Physically invalid power')
    ax.set(xlabel='Offered load per USR instance (Mbps)', ylabel='UPF-attributed power (W)',
           xlim=(0, 20), ylim=(-.07, .21), title='Original USR model validity at low load; no refitting')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=.2)
    fig.savefig(output / 'usr_power_low_load.png', dpi=180)
    fig.savefig(output / 'usr_power_low_load.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / 'revision'):
        raise ValueError('Diagnostic outputs must be new files under revision/')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    twin = FittedUPFTwin()
    measurements, levels = measurement_analysis(twin, args.output_dir)
    grouped = grouped_export_checks()
    workloads = workload_census(twin, args.output_dir, DEFAULT_PROTOCOL)
    grid = low_load_grid(twin, args.output_dir)
    plot_low_load(args.output_dir, levels)
    sources = [PROFILING / f'{m}_processed.csv' for m in ('dpdk', 'oai')]
    sources += [PROFILING.parent / 'raw' / f'{m}_grouped_limited.csv' for m in ('dpdk', 'oai')]
    sources += [TRAINING_SOURCE / 'train_power_models.py', TRAINING_SOURCE / 'train_performance_models.py',
                Path('/home/ubuntu/xRL_UpfSelection/config.yaml'), ROOT / 'config.yaml']
    report = {'task': '2B', 'analysis_kind': 'read_only_original_model_validity_audit',
              'measurements': measurements, 'grouped_export_checks': grouped, 'workloads': workloads,
              'low_load_grid': grid, 'protocol_sha256': sha256(DEFAULT_PROTOCOL), 'twin_id': twin.identity,
              'source_files': [{'path': str(p), 'sha256': sha256(p)} for p in sources],
              'no_refitting': True, 'no_controller_rollouts': True, 'no_final_test_inference': True}
    with (args.output_dir / 'validity_summary.json').open('x') as handle:
        json.dump(diagnostic_json(report), handle, indent=2, allow_nan=False)
        handle.write('\n')
    with (args.output_dir / 'artifact_manifest.json').open('x') as handle:
        json.dump(twin.manifest, handle, indent=2)
        handle.write('\n')
    twin.verify_artifacts()
    print('Validity audit diagnostics complete; no models fitted or corrected.', flush=True)


if __name__ == '__main__':
    main()
