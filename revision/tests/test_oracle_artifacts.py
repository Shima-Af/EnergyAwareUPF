"""Verify all validation-only bounds and frozen saved controller trajectories.

No model fitting, policy calls, legacy full-file reader, or final-test horizon.
"""
import csv
import json
import math
from fractions import Fraction

import pytest

from revision.evaluation.contract import Outcome, State, account, actuate
from revision.evaluation.twin import ROOT, sha256
from revision.oracles.solver import OBJECTIVES, solve
from revision.stage7.common import horizons

OUT = ROOT / 'revision/reports/oracles'
PRIOR = ROOT / 'revision/reports/stage7'


def cache():
    meta = json.loads((PRIOR / 'prepared.json').read_text())
    assert sha256(PRIOR / 'outcome_cache.csv') == meta['cache_sha256']
    with (PRIOR / 'outcome_cache.csv').open() as f:
        return {(int(r['action']), float(r['load_mbps'])):
                Outcome(float(r['power_w']), float(r['qos'])) for r in csv.DictReader(f)}


@pytest.mark.parametrize('workload', ['netflix', 'youtube'])
def test_every_saved_controller_matches_horizon_twin_contract_and_bounds(workload):
    _, h, c = horizons(workload)
    outcomes = cache()
    archive = json.loads((PRIOR / 'comparison.json').read_text())['workloads'][workload]
    comparison = json.loads((OUT / 'comparison.json').read_text())['workloads'][workload]
    assert len(h.timestamps) == 504 and h.partition.stop == 2520
    assert h.partition.timeline.slot.total_seconds() == 900
    assert len(h.partition.timeline.timestamps) == 2520
    for endpoint in ('selected', 'endpoint'):
        for name, metrics in archive[f'controllers_{endpoint}'].items():
            with (PRIOR / workload / f'{name}_{endpoint}_steps.csv').open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == len(h.timestamps)
            state = State(0, 4); energies = []; rewards = []; discounts = []
            for t, (i, row) in enumerate(zip(h.indices, rows)):
                assert int(row['source_index']) == i
                assert row['timestamp'] == h.timestamps[t].isoformat()
                load = float(row['actual_load_mbps'])
                assert load == h.partition.timeline.load_mbps[i]
                request = int(row['requested'])
                if name == 'comcom_historical':
                    assert request in (0, 1)
                tr = actuate(state, request, c)
                assert tr.executed == int(row['executed'])
                assert tr.blocked == (row['blocked'] == 'True')
                outcome = outcomes[tr.executed, load]
                assert outcome == Outcome(float(row['power_w']), float(row['qos']))
                a = account(outcome, load, 900., tr, c)
                assert a.reward == float(row['reward'])
                assert a.energy_wh == float(row['energy_wh'])
                energies.append(a.energy_wh); rewards.append(a.reward)
                discounts.append(c.gamma ** t * a.reward)
                state = tr.next_state
            assert math.fsum(energies) == metrics['energy_wh']
            assert math.fsum(discounts) == metrics['discounted_return']
            assert math.fsum(rewards) == metrics['total_reward']
            gap = comparison['controller_gaps'][name + '_' + endpoint]
            assert gap['energy_gap_wh'] >= 0 and gap['discounted_return_gap'] >= 0


@pytest.mark.parametrize('workload', ['netflix', 'youtube'])
def test_all_oracle_artifacts_repeat_exactly_and_constraints_hold(workload):
    _, h, c = horizons(workload)
    table = cache()
    loads = [h.partition.timeline.load_mbps[i] for i in h.indices]
    outcomes = [[table[a, load] for a in range(3)] for load in loads]
    report = json.loads((OUT / 'comparison.json').read_text())
    assert not report['final_test_evaluated'] and report['policy_replays'] == 0
    data = report['workloads'][workload]
    assert data['horizon'] == h.identity
    for label, actions in [('binary', (0, 1)), ('ternary', (0, 1, 2))]:
        for objective in OBJECTIVES:
            saved = data['action_sets'][label][objective]
            solved = solve(loads, outcomes, actions=actions, contract=c, objective=objective)
            assert solved.feasible and list(solved.actions) == saved['actions']
            assert solved.objective_value == saved['objective_value']
            assert Fraction(solved.exact_objective_numerator, solved.exact_objective_denominator) == Fraction(
                saved['exact_objective_numerator'], saved['exact_objective_denominator'])
            assert saved['metrics']['blocked_requests'] == 0
            if objective == 'qos_constrained_energy_oracle':
                assert all(outcomes[t][a].qos >= .9 for t, a in enumerate(solved.actions))
                assert saved['actions'] == data['action_sets'][label]['energy_optimal_oracle']['actions']


def test_oracle_pipeline_does_not_call_full_file_reader(monkeypatch):
    from revision.evaluation.data import Timeline
    def forbidden(*args, **kwargs):
        raise AssertionError('Full-file reader is prohibited')
    monkeypatch.setattr(Timeline, 'read_csv', forbidden)
    for workload in ('netflix', 'youtube'):
        _, h, _ = horizons(workload)
        assert list(h.indices) == list(range(2016, 2520))
