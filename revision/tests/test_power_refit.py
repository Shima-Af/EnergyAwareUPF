"""Grouped model learning, physical constraints and opt-in twin checks."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from revision.profiling.power_model import (
    LOAD, POWER, PowerModel, ProfilingPartition, fit_candidate, linear_basis,
    partition_sweeps, select_candidate,
)
from revision.evaluation.diagnostic_power_twin import DiagnosticPowerTwin
from revision.evaluation.twin import InvalidPrediction, RawPrediction, sha256

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def partitions():
    rows = []
    for sweep in range(1, 6):
        for i, load in enumerate((0.01, 1., 5., 10.)):
            rows.append({'Test Set': sweep, 'Test Case': sweep * 10 + i,
                         'parsed_timestamp': pd.Timestamp('2025-01-01') + pd.Timedelta(days=sweep, seconds=i),
                         LOAD: load, POWER: .02 + .1 * load})
    return partition_sweeps(pd.DataFrame(rows), {'train': [1, 2, 3], 'validation': [4], 'internal_test': [5]})


def test_whole_sweeps_are_disjoint_and_chronological(partitions):
    assert [len(p.frame) for p in partitions.values()] == [12, 4, 4]
    assert set(partitions['train'].frame['Test Set']) == {1, 2, 3}
    assert not set(partitions['train'].frame.index) & set(partitions['validation'].frame.index)


@pytest.mark.parametrize('split', [
    {'train': [1, 2, 3], 'validation': [3, 4], 'internal_test': [5]},
    {'train': [2, 3, 4], 'validation': [1], 'internal_test': [5]},
    {'train': [1, 2], 'validation': [4], 'internal_test': [5]},
])
def test_overlapping_missing_or_nonchronological_sweeps_rejected(partitions, split):
    frame = pd.concat([p.frame for p in partitions.values()])
    with pytest.raises(ValueError):
        partition_sweeps(frame, split)


def test_cross_sweep_case_block_rejected(partitions):
    frame = pd.concat([p.frame for p in partitions.values()])
    frame.loc[frame.index[4], 'Test Case'] = frame.iloc[0]['Test Case']
    with pytest.raises(ValueError, match='crosses sweeps'):
        partition_sweeps(frame, {'train': [1, 2, 3], 'validation': [4], 'internal_test': [5]})


@pytest.mark.parametrize('role', ['validation', 'internal_test'])
def test_fit_cannot_use_selection_or_internal_test_labels(partitions, role):
    with pytest.raises(ValueError, match='TRAIN'):
        fit_candidate(partitions[role], {'name': 'affine', 'family': 'affine'}, (.01, 10.))


@pytest.mark.parametrize('role', ['train', 'internal_test'])
def test_model_selection_rejects_nonvalidation_roles(partitions, role):
    model = PowerModel('a', 'affine', (.01, 10.), (0., .1))
    with pytest.raises(ValueError, match='VALIDATION'):
        select_candidate((model,), partitions[role])


@pytest.mark.parametrize('family', ['affine', 'piecewise_linear'])
def test_recover_known_positive_curve_deterministically_in_watts(partitions, family):
    spec = {'name': family, 'family': family, 'knots_mbps': [.01, 1., 10.]}
    one = fit_candidate(partitions['train'], spec, (.01, 10.))
    two = fit_candidate(partitions['train'], spec, (.01, 10.))
    assert one == two
    np.testing.assert_allclose(one.predict([.01, 2., 10.]), [.021, .22, 1.02], atol=1e-14)
    assert np.all(one.predict(np.linspace(.01, 10., 1001)) >= 0)


def test_future_label_poisoning_does_not_change_fit_or_selection(partitions):
    specs = [{'name': 'a', 'family': 'affine'},
             {'name': 'b', 'family': 'piecewise_linear', 'knots_mbps': [.01, 1., 10.]}]
    before = tuple(fit_candidate(partitions['train'], s, (.01, 10.)) for s in specs)
    chosen = select_candidate(before, partitions['validation'])[0]
    partitions['internal_test'].frame[POWER] = 1e99
    after = tuple(fit_candidate(partitions['train'], s, (.01, 10.)) for s in specs)
    assert before == after
    assert chosen == select_candidate(after, partitions['validation'])[0]
    partitions['validation'].frame[POWER] = 1e80
    assert before == tuple(fit_candidate(partitions['train'], s, (.01, 10.)) for s in specs)


def test_interpolation_weights_are_convex_including_endpoints():
    basis = linear_basis([.01, .5, 1., 9., 10.], [.01, 1., 10.])
    assert np.all(basis >= 0)
    np.testing.assert_allclose(basis.sum(axis=1), 1., atol=1e-15)
    np.testing.assert_array_equal(basis[[0, -1]], [[1, 0, 0], [0, 0, 1]])


@pytest.mark.parametrize('bad_load', [0., -.1, 5001., float('nan'), float('inf')])
def test_saved_model_rejects_unmeasured_idle_and_extrapolation(bad_load):
    model = PowerModel.load(ROOT / 'revision/models/task2c/selected_usr_power.json')
    with pytest.raises(ValueError, match='domain'):
        model.predict([bad_load])


def test_parameters_cannot_hide_negative_or_nonfinite_power():
    for value in (-.01, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='nonnegative'):
            PowerModel('bad', 'affine', (.01, 10.), (value, 1.))


class FakeOriginal:
    identity = 'unchanged-original'
    qos = .95

    def raw_prediction(self, action, load):
        k = 2 if action == 2 else 1
        # Negative original USR power is deliberately replaced, never clipped.
        p = 7. if action == 0 else -1.
        return RawPrediction(action, load, k, load / k, float(np.float32(load / k)), p, k * p, self.qos)

    def verify_artifacts(self):
        pass


@pytest.fixture
def diagnostic(tmp_path):
    artifact = {'schema_version': 1, 'target_units': 'process_attributed_W',
                'model': asdict(PowerModel('simple', 'affine', (.01, 100.), (.01, .1)))}
    model = tmp_path / 'model.json'
    model.write_text(json.dumps(artifact))
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'selected_name': 'simple', 'selected_model_sha256': sha256(model),
                                     'traffic_used_for_selection': False}))
    return DiagnosticPowerTwin(FakeOriginal(), model, selection)


def test_opt_in_replacement_preserves_dpdk_qos_and_balanced_instances(diagnostic):
    assert diagnostic.raw_prediction(0, 8.).power_w == 7.
    assert diagnostic.original.raw_prediction(1, 8.).power_w == -1.
    one = diagnostic.raw_prediction(1, 4.)
    two = diagnostic.raw_prediction(2, 8.)
    assert two.power_w == 2 * one.power_w
    assert one.qos == two.qos == diagnostic.original.qos
    assert two.power_model_input_mbps == 4.
    assert not two.failures
    assert diagnostic.outcome(2, 8.) == diagnostic.outcome(2, 8.)


def test_replacement_does_not_mask_invalid_original_qos(diagnostic):
    diagnostic.original.qos = 1.1
    with pytest.raises(InvalidPrediction):
        diagnostic.outcome(2, 8.)


def test_frozen_selection_hash_and_mutation_are_checked(diagnostic):
    diagnostic.model_path.write_text(diagnostic.model_path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        diagnostic.verify_artifacts()
    with pytest.raises(ValueError, match='frozen'):
        DiagnosticPowerTwin(FakeOriginal(), diagnostic.model_path, diagnostic.selection_path)


def test_saved_artifact_reloads_exactly_and_records_training_only_provenance():
    path = ROOT / 'revision/models/task2c/selected_usr_power.json'
    selection = json.loads((ROOT / 'revision/reports/task2c/selection_frozen.json').read_text())
    artifact = json.loads(path.read_text())
    assert sha256(path) == selection['selected_model_sha256']
    assert artifact['training_sweeps'] == [1, 2, 3]
    assert selection['validation_sweeps'] == [4]
    assert selection['internal_test_scored_before_selection'] is False
    assert selection['traffic_used_for_selection'] is False
    x = np.geomspace(.01, 5000., 2001)
    np.testing.assert_array_equal(PowerModel.load(path).predict(x), PowerModel.load(path).predict(x))
    assert np.all(PowerModel.load(path).predict(x) > 0)
