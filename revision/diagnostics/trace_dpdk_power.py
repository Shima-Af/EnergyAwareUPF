"""Trace DPDK labels to surviving process exports without altering either source."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from revision.evaluation.twin import ROOT, sha256
from revision.evaluation.validate_static import write_csv
from revision.profiling.refit_usr_power import write_json


def main():
    cfg = json.loads((ROOT / 'revision/configs/final_dpdk_power_v1.json').read_text())
    source = Path(cfg['source_csv'])
    assert sha256(source) == cfg['source_sha256']
    frame = pd.read_csv(source)
    base = Path('/home/ubuntu/UpfProfilingCampaign/UpfProfilingCampaign')
    raw_path = base / 'data/raw/scaphandre/dpdk/scaph_process_power_consumption_microwatts.csv'
    pid_path = raw_path.with_name('upf_PIDs.txt')
    pids = [int(line.strip()) for line in pid_path.read_text().splitlines() if line.strip().isdigit()]
    retained = []
    for chunk in pd.read_csv(raw_path, chunksize=250000):
        retained.append(chunk[chunk.pid.isin(pids)])
    raw = pd.concat(retained, ignore_index=True)
    aggregate = raw.groupby('timestamp', as_index=False).value.sum()
    aggregate_times = pd.to_datetime(aggregate.timestamp, format='ISO8601').astype('int64').to_numpy()
    aggregate_watts = aggregate.value.to_numpy() / 1e6
    times = pd.to_datetime(raw.timestamp, format='ISO8601').astype('int64').to_numpy()
    target_times = pd.to_datetime(frame.timestamp, format='ISO8601').astype('int64').to_numpy()
    watts = raw.value.to_numpy() / 1e6
    order = np.argsort(times)
    times, watts = times[order], watts[order]
    ordered = raw.iloc[order].reset_index(drop=True)
    matches = []
    for i, row in frame.iterrows():
        t = target_times[i]
        left, right = np.searchsorted(times, [t - 6000000000, t + 6000000000])
        nearby = np.arange(left, right)
        equal = nearby[np.isclose(watts[nearby], row.power_consumption_watts, rtol=0, atol=1e-12)]
        chosen = int(equal[np.argmin(abs(times[equal] - t))]) if len(equal) else None
        nearby_sums = np.flatnonzero((abs(aggregate_times - t) <= 6000000000) &
                                    np.isclose(aggregate_watts, row.power_consumption_watts, rtol=0, atol=1e-12))
        matches.append({'source_row': int(i), 'timestamp': row.timestamp,
                        'sweep': int(row['Test Set']), 'processed_w': float(row.power_consumption_watts),
                        'matching_process_value_found': chosen is not None,
                        'matching_catalog_pid_sum_found': len(nearby_sums) > 0,
                        'pid': int(ordered.pid.iloc[chosen]) if chosen is not None else None,
                        'exe': ordered.exe.iloc[chosen] if chosen is not None else None,
                        'offset_seconds': float((times[chosen] - t) / 1e9) if chosen is not None else None})
    out = ROOT / 'revision/reports/final_twin'
    write_csv(out / 'dpdk_power_raw_aggregation_matches.csv', matches)
    write_json(out / 'dpdk_power_aggregation_provenance.json', {
        'processed_path': str(source), 'processed_sha256': sha256(source),
        'raw_path': str(raw_path), 'raw_sha256': sha256(raw_path),
        'pid_file': str(pid_path), 'pid_sha256': sha256(pid_path), 'catalog_pids': pids,
        'retained_raw_rows': len(raw), 'raw_executables': sorted(raw.exe.unique().tolist()),
        'labels': len(matches), 'individual_process_value_matches': sum(m['matching_process_value_found'] for m in matches),
        'catalog_pid_sum_matches': sum(m['matching_catalog_pid_sum_found'] for m in matches),
        'aggregation_evidence': 'Surviving campaign src/merge.py _load_scaphandre_csv filters catalog PIDs and sums values at equal timestamps; not authenticated as the historical processed-CSV generator',
        'aggregation_code_path': str(base / 'src/merge.py'),
        'aggregation_code_sha256': sha256(base / 'src/merge.py'),
        'conversion': 'raw microwatts / 1e6', 'value_tolerance_w': 1e-12,
        'time_window_seconds': 6,
        'interpretation': 'Numerical label provenance only; not metering accuracy or authentication of historical join/aggregation',
        'measurement_summary_by_sweep': [
            {'sweep': int(s), 'n': len(g), 'mean_w': float(g.power_consumption_watts.mean()),
             'sd_w': float(g.power_consumption_watts.std()), 'min_w': float(g.power_consumption_watts.min()),
             'max_w': float(g.power_consumption_watts.max())} for s, g in frame.groupby('Test Set')]
    })


if __name__ == '__main__':
    main()
