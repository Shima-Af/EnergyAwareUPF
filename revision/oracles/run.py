"""Validation-only oracle comparisons; consumes frozen Stage 7 outcomes/results.

Run: .venv/bin/python -m revision.oracles.run
All writers refuse overwrites. No controller is loaded, trained or replayed.
"""
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

from revision.evaluation.contract import (Outcome, State, Step, EvaluationResult,
                                           account, actuate)
from revision.evaluation.twin import ROOT, sha256
from revision.stage7.common import horizons
from revision.oracles.solver import OBJECTIVES, solve

OUT = ROOT / 'revision/reports/oracles'
PRIOR = ROOT / 'revision/reports/stage7'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate_solution(horizon, contract, solution, outcome_rows):
    if not solution.feasible:
        return None, []
    state = State(contract.initial_action, contract.cooldown_slots)
    steps = []; rows = []
    for t, (index, action) in enumerate(zip(horizon.indices, solution.actions)):
        tr = actuate(state, action, contract)
        if tr.blocked:
            raise AssertionError('Oracle emitted inadmissible transition')
        load = horizon.partition.timeline.load_mbps[index]
        out = outcome_rows[t][action]
        a = account(out, load, 900., tr, contract)
        steps.append(Step(index, horizon.context(index), None, load, tr, out, a))
        rows.append(dict(source_index=index, timestamp=horizon.timestamps[t].isoformat(),
                         actual_load_mbps=load, **{k: v for k, v in asdict(tr).items() if k != 'next_state'},
                         **asdict(out), **asdict(a)))
        state = tr.next_state
    result = EvaluationResult(solution.objective, horizon, contract, 'instant', tuple(steps))
    summary = dict(result.summary)
    summary.update(min_predicted_qos=min(s.outcome.qos for s in steps),
                   mean_predicted_qos=math.fsum(s.outcome.qos for s in steps) / len(steps),
                   action_counts=[sum(s.transition.executed == a for s in steps) for a in range(3)],
                   horizon_identity=horizon.identity)
    key = 'discounted_return' if solution.objective == 'reward_optimal_oracle' else 'energy_wh'
    assert summary[key] == solution.objective_value  # Correctly rounded exact sum.
    assert summary['reconfigurations'] == solution.reconfigurations
    return summary, rows


def gaps(metrics, oracles, action_set):
    energy = oracles['energy_optimal_oracle']['metrics']['energy_wh']
    upper = oracles['reward_optimal_oracle']['metrics']['discounted_return']
    feasible = oracles['qos_constrained_energy_oracle']
    # Subtractions use rounded reporting values; bound verification allows only
    # numerical summation precision, never an optimization/tie tolerance.
    e = metrics['energy_wh'] - energy
    r = upper - metrics['discounted_return']
    if e < -1e-10 or r < -1e-10:
        raise AssertionError('Archived controller exceeds its matched oracle bound')
    constrained_gap = None
    if feasible['feasible'] and metrics['qos_violation_rate'] == 0:
        constrained_gap = metrics['energy_wh'] - feasible['metrics']['energy_wh']
        assert constrained_gap >= -1e-10
    return dict(action_set=action_set, energy_gap_wh=e, energy_gap_pct=100 * e / energy,
                discounted_return_gap=r, qos_constrained_energy_gap_wh=constrained_gap,
                qos_constraint_satisfied=metrics['qos_violation_rate'] == 0)


def main():
    meta = json.loads((PRIOR / 'prepared.json').read_text())
    cache_path = PRIOR / 'outcome_cache.csv'
    assert sha256(cache_path) == meta['cache_sha256']
    for spec in meta['twin_manifest']['config']['models'].values():
        for artifact in spec['artifacts']:
            assert sha256(ROOT / artifact['path']) == artifact['sha256']
    with cache_path.open() as f:
        cache = {(int(r['action']), float(r['load_mbps'])):
                 Outcome(float(r['power_w']), float(r['qos'])) for r in csv.DictReader(f)}
    archive = json.loads((PRIOR / 'comparison.json').read_text())
    report = dict(twin_id=meta['twin_id'], workloads={}, final_test_evaluated=False,
                  source_comparison_sha256=sha256(PRIOR / 'comparison.json'),
                  source_cache_sha256=sha256(cache_path),
                  objectives=list(OBJECTIVES), binary_actions=[0, 1], ternary_actions=[0, 1, 2],
                  reward_gamma=.995, slot_hours=.25, policy_replays=0,
                  exact_cost_arithmetic='common-denominator binary64 integers; no optimization tolerance')
    all_gaps = []
    for workload in ('netflix', 'youtube'):
        _, horizon, contract = horizons(workload)
        assert horizon.partition.role == 'validation' and tuple(horizon.indices) == tuple(range(2016, 2520))
        old = archive['workloads'][workload]
        assert old['horizon'] == horizon.identity or all(
            m['horizon_identity'] == horizon.identity for m in old['controllers_selected'].values())
        assert old['timestamps'] == [t.isoformat() for t in horizon.timestamps]
        loads = [horizon.partition.timeline.load_mbps[i] for i in horizon.indices]
        outcomes = [[cache[a, float(load)] for a in range(3)] for load in loads]
        found = dict(horizon=horizon.identity, timestamps=old['timestamps'], slots=len(loads),
                     all_action_outcomes_meet_target=all(o.qos >= .9 for row in outcomes for o in row),
                     action_sets={}, controller_gaps={})
        for label, actions in (('binary', (0, 1)), ('ternary', (0, 1, 2))):
            values = {}
            for objective in OBJECTIVES:
                solution = solve(loads, outcomes, objective=objective, actions=actions, contract=contract)
                summary, rows = evaluate_solution(horizon, contract, solution, outcomes)
                values[objective] = {**asdict(solution), 'metrics': summary}
                if rows:
                    write_csv(OUT / f'{workload}_{label}_{objective}_steps.csv', rows)
            found['action_sets'][label] = values
        b = found['action_sets']['binary']; t = found['action_sets']['ternary']
        found['value_of_usr2'] = dict(
            energy_reduction_wh=b['energy_optimal_oracle']['metrics']['energy_wh'] - t['energy_optimal_oracle']['metrics']['energy_wh'],
            discounted_return_increase=t['reward_optimal_oracle']['metrics']['discounted_return'] - b['reward_optimal_oracle']['metrics']['discounted_return'])
        assert found['value_of_usr2']['energy_reduction_wh'] >= 0
        assert found['value_of_usr2']['discounted_return_increase'] >= 0
        for endpoint in ('selected', 'endpoint'):
            for name, metrics in old[f'controllers_{endpoint}'].items():
                # Historical COMCOM is binary by definition, regardless of occupancy.
                # Every learned policy and A/B rule has three available actions.
                label = 'binary' if name == 'comcom_historical' else 'ternary'
                row = dict(workload=workload, controller=name, endpoint=endpoint,
                           energy_wh=metrics['energy_wh'], discounted_return=metrics['discounted_return'],
                           **gaps(metrics, found['action_sets'][label], label))
                all_gaps.append(row)
                found['controller_gaps'][name + '_' + endpoint] = row
        report['workloads'][workload] = found
    dump(OUT / 'comparison.json', report)
    write_csv(OUT / 'controller_gaps.csv', all_gaps)
    dump(OUT / 'provenance.json', dict(
        source_paths=[str(PRIOR.relative_to(ROOT) / name) for name in ('comparison.json', 'prepared.json', 'outcome_cache.csv')],
        input_source_hashes={str(PRIOR.relative_to(ROOT) / name): sha256(PRIOR / name)
                             for name in ('comparison.json', 'prepared.json', 'outcome_cache.csv')},
        implementation_hashes={str(p.relative_to(ROOT)): sha256(p) for p in
                               (Path(__file__), Path(__file__).with_name('solver.py'),
                                ROOT / 'revision/evaluation/contract.py', ROOT / 'revision/stage7/common.py')},
        contract=asdict(contract), twin_manifest=meta['twin_manifest'],
        input_access='Forecast-service prefix-only reader, rows 0..2519; only 2016..2519 scored',
        inference='Hash-verified frozen Stage7 real-twin outcome cache; no model or controller selection'))
    print(json.dumps({w: {a: {o: v['metrics'] for o, v in d.items()}
                         for a, d in r['action_sets'].items()}
                      for w, r in report['workloads'].items()}, indent=2))


if __name__ == '__main__':
    main()
