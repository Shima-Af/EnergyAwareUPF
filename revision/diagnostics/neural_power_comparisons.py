"""Post-selection power-ranking diagnostics on a synthetic load grid, not traffic."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from revision.evaluation.twin import ROOT, FittedUPFTwin, sha256
from revision.evaluation.validate_static import write_csv
from revision.profiling.neural_power import NeuralPowerModel
from revision.profiling.power_model import PowerModel
from revision.profiling.refit_usr_power import write_json


def residual_budget(loads, residuals):
    """Empirical validation error scenario, not a statistical bound or interval."""
    levels = np.asarray(sorted(residuals))
    budgets = []
    for x in loads:
        right = min(np.searchsorted(levels, x, side='left'), len(levels) - 1)
        left = max(right - 1, 0)
        budgets.append(max(residuals[levels[left]], residuals[levels[right]]))
    return np.asarray(budgets)


def run(output):
    original = FittedUPFTwin()
    selected_path = ROOT / 'revision/reports/task2d/selection_frozen.json'
    selected = json.loads(selected_path.read_text())
    manifest = Path(selected['selected_manifest_path'])
    if sha256(manifest) != selected['selected_manifest_sha256']:
        raise ValueError('Frozen neural selection changed')
    nn = NeuralPowerModel(manifest)
    pwl = PowerModel.load(ROOT / 'revision/models/task2c/selected_usr_power.json')
    peer_paths = sorted(manifest.parent.parent.glob(f"{nn.metadata['architecture']['architecture']}_seed_*/manifest.json"))
    peers = [NeuralPowerModel(p) for p in peer_paths]
    x = np.unique(np.r_[np.linspace(.02, 400., 2001), [.02, .1, 1., 2., 4., 8., 20., 40., 100., 200., 400.]])
    # No traffic file is loaded and these are not controller decisions/rollouts.
    dpdk = np.asarray([original.raw_prediction(0, float(v)).power_w for v in x])
    qos = np.asarray([[original.raw_prediction(a, float(v)).qos for a in range(3)] for v in x])
    powers = {}
    for m in (nn, pwl, *peers):
        powers[m.name] = np.column_stack([dpdk, m.predict(x), 2 * m.predict(x / 2)])
    validation = pd.read_csv(ROOT / 'revision/reports/task2d/predictions.csv')
    validation = validation[(validation.model == nn.name) & (validation.role == 'validation')]
    observed = validation.groupby('load_mbps').error_w.apply(lambda r: float(abs(r).max())).to_dict()
    b1, b2 = residual_budget(x, observed), 2 * residual_budget(x / 2, observed)
    ref, other = powers[nn.name], powers[pwl.name]
    usr_margin = abs(ref[:, 1] - ref[:, 2])
    seed_signs = np.stack([powers[p.name][:, 1] < powers[p.name][:, 2] for p in peers])
    ambiguity = usr_margin <= b1 + b2
    rows = []
    for i, load in enumerate(x):
        feasible = np.flatnonzero(qos[i] >= .9)
        nn_feasible = int(feasible[np.argmin(ref[i, feasible])]) if len(feasible) else None
        pwl_feasible = int(feasible[np.argmin(other[i, feasible])]) if len(feasible) else None
        rows.append({'total_load_mbps': float(load), 'nn_usr1_power_w': float(ref[i, 1]),
          'nn_usr2_power_w': float(ref[i, 2]), 'pwl_usr1_power_w': float(other[i, 1]),
          'pwl_usr2_power_w': float(other[i, 2]), 'dpdk_power_w': float(dpdk[i]),
          'nn_lowest_power_action': int(np.argmin(ref[i])), 'pwl_lowest_power_action': int(np.argmin(other[i])),
          'nn_lowest_power_predicted_qos_feasible_action': nn_feasible,
          'pwl_lowest_power_predicted_qos_feasible_action': pwl_feasible,
          'nn_usr1_minus_usr2_w': float(ref[i, 1] - ref[i, 2]),
          'usr_ranking_differs_nn_vs_pwl': bool((ref[i, 1] < ref[i, 2]) != (other[i, 1] < other[i, 2])),
          'usr_ranking_differs_between_nn_seeds': bool(seed_signs[:, i].any() != seed_signs[:, i].all()),
          'empirical_usr1_error_budget_w': float(b1[i]), 'empirical_usr2_error_budget_w': float(b2[i]),
          'usr_ranking_ambiguous_under_error_scenario': bool(ambiguity[i])})
    write_csv(output / 'configuration_power_comparisons.csv', rows)
    summary = {'synthetic_total_load_range_mbps': [.02, 400.], 'grid_points': len(x),
       'traffic_files_read': False, 'controller_rollouts': False, 'model_selection_used_diagnostics': False,
       'nn_usr_ranking_differs_from_pwl_points': int(sum(r['usr_ranking_differs_nn_vs_pwl'] for r in rows)),
       'lowest_power_action_differs_nn_vs_pwl_points': int(sum(r['nn_lowest_power_action'] != r['pwl_lowest_power_action'] for r in rows)),
       'predicted_qos_feasible_power_action_differs_points': int(sum(r['nn_lowest_power_predicted_qos_feasible_action'] != r['pwl_lowest_power_predicted_qos_feasible_action'] for r in rows)),
       'usr_ranking_differs_between_seeds_points': int(sum(r['usr_ranking_differs_between_nn_seeds'] for r in rows)),
       'usr_ranking_ambiguous_under_validation_error_scenario_points': int(ambiguity.sum()),
       'scenario_definition': 'Sum of per-configuration propagated maximum absolute validation residuals at bracketing profiling levels; may vary independently for each configuration',
       'limitations': ['Empirical stress scenario, not a calibrated error bound or probability',
                      'USR errors share a fitted function; worst-case independent signs need not be jointly attainable',
                      'Two seed spread is not a calibrated uncertainty interval and can miss common bias',
                      'DPDK and QoS model uncertainties excluded; predicted feasibility is not physical safety',
                      'Grid frequencies depend on sampling and are not workload frequencies',
                      'No cooldown, reward, state, forecast or energy integration; not controller results'],
       'selected_manifest_sha256': sha256(manifest)}
    write_json(output / 'configuration_comparison_summary.json', summary)
    original.verify_artifacts()
    for m in peers: m.verify_artifacts()
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / 'revision'):
        raise ValueError('Diagnostics must stay under revision/')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run(args.output_dir)


if __name__ == '__main__':
    main()
