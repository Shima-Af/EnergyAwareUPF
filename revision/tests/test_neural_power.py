"""Neural artifact, grouped protocol and physical-output contract checks."""

import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest

from revision.evaluation.twin import _runtime, sha256
from revision.profiling.neural_power import NeuralPowerModel, create_network, transformed_input
from revision.profiling.power_model import LOAD, POWER, ProfilingPartition
from revision.profiling.train_neural_power import CONFIG, fit_one

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def selected_nn():
    selection = json.loads((ROOT / 'revision/reports/task2d/selection_frozen.json').read_text())
    return NeuralPowerModel(Path(selection['selected_manifest_path']))


def test_grouped_split_and_budget_match_frozen_task2c():
    config = json.loads(CONFIG.read_text())
    prior = json.loads((ROOT / config['task2c_config']).read_text())
    protocol = json.loads((ROOT / 'revision/reports/task2d/prespecified_protocol.json').read_text())
    assert config['split'] == prior['split'] == {'train': [1, 2, 3], 'validation': [4], 'internal_test': [5]}
    expected = pd.read_csv(ROOT / 'revision/reports/task2c/profiling_split.csv')
    for role, rows in protocol['source_rows_by_role'].items():
        assert rows == list(expected[expected.role == role].source_row)
    assert len(config['candidates']) * len(config['seeds']) == 4
    assert config['max_epochs'] == 600
    assert protocol['config_sha256'] == sha256(CONFIG)


def test_selection_matches_validation_only_and_test_not_scored_for_losers():
    result = json.loads((ROOT / 'revision/reports/task2d/selection_frozen.json').read_text())
    minimum = min(result['validation_scores'], key=lambda n: result['validation_scores'][n]['rmse_w'])
    assert result['selected_name'] == minimum
    assert not result['internal_test_scored_before_selection']
    assert not result['traffic_used_for_selection']
    assert not result['search_expanded']
    scores = json.loads((ROOT / 'revision/reports/task2d/metrics.json').read_text())
    assert set(scores['models']) == {minimum, 'nonnegative_pwl_7', 'original_saved_usr_power_descriptive_only'}


def test_training_only_scaler_matches_measurements(selected_nn):
    config = json.loads((ROOT / 'revision/configs/usr_power_refit_v1.json').read_text())
    frame = pd.read_csv(config['source_csv'])
    x = np.log1p(frame[frame['Test Set'].isin([1, 2, 3])][LOAD].to_numpy())
    scaler = selected_nn.metadata['input_scaler']
    assert scaler['mean'] == x.mean()
    assert scaler['scale'] == x.std()
    assert scaler['fit_sweeps'] == [1, 2, 3]
    assert scaler['fit_rows'] == 2736


@pytest.mark.parametrize('spec,bn,dropout', [
    ({'batch_norm': True, 'dropout': .15}, 2, 1),
    ({'batch_norm': False, 'dropout': 0.}, 0, 0),
])
def test_minimal_architecture_and_physical_output_head(spec, bn, dropout):
    _, tf, _, _ = _runtime()
    model = create_network(tf, spec, 42)
    dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    assert [l.units for l in dense] == [64, 32, 1]
    assert dense[-1].get_config()['activation'] == 'softplus'
    assert sum(isinstance(l, tf.keras.layers.BatchNormalization) for l in model.layers) == bn
    assert sum(isinstance(l, tf.keras.layers.Dropout) for l in model.layers) == dropout
    # Force a negative latent output: the native head still returns nonnegative watts.
    kernel, bias = dense[-1].get_weights()
    dense[-1].set_weights([np.zeros_like(kernel), np.full_like(bias, -50.)])
    y = np.asarray(model(np.asarray([[-2.], [0.], [2.]], dtype=np.float32), training=False))
    assert np.all(y >= 0) and np.all(y < 1e-20)


@pytest.mark.parametrize('load', [0., -1., 5001., np.nan, np.inf])
def test_invalid_neural_domain_is_not_repaired(selected_nn, load):
    with pytest.raises(ValueError, match='domain'):
        selected_nn.predict([load])


def test_reload_repeated_outputs_and_between_measurement_loads(selected_nn):
    x = np.geomspace(.01, 5000., 1003)
    first = selected_nn.predict(x)
    np.testing.assert_array_equal(first, selected_nn.predict(x))
    np.testing.assert_array_equal(first, NeuralPowerModel(selected_nn.manifest_path).predict(x))
    assert np.isfinite(first).all() and (first >= 0).all()
    assert selected_nn.metadata['target_transform'] == 'identity_watts'
    assert np.isfinite(selected_nn.predict([2.3, 4.36, 11.17, 113.2])).all()


def test_neural_artifact_hash_tampering_rejected(selected_nn, tmp_path):
    manifest = json.loads(selected_nn.manifest_path.read_text())
    shutil.copy(selected_nn.model_path, tmp_path / 'model.keras')
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with (tmp_path / 'model.keras').open('ab') as f:
        f.write(b'tamper')
    with pytest.raises(ValueError, match='hash mismatch'):
        NeuralPowerModel(path)


def synthetic_part(role, sweep):
    x = np.asarray([.01, 1., 10., 100.])
    return ProfilingPartition(role, (sweep,), pd.DataFrame({'Test Set': sweep, LOAD: x, POWER: .01 + x * .01}))


def test_training_rejects_internal_test_as_learning_or_selection(tmp_path):
    _, tf, _, _ = _runtime()
    config = json.loads(CONFIG.read_text())
    spec = config['candidates'][0]
    test = synthetic_part('internal_test', 5)
    train, val = synthetic_part('train', 1), synthetic_part('validation', 4)
    for learning, selection in [(test, val), (train, test)]:
        with pytest.raises(ValueError, match='TRAIN'):
            fit_one(learning, selection, spec, 42, config, tmp_path / 'forbidden', tf)


def test_small_same_seed_training_is_reproducible(tmp_path):
    _, tf, _, _ = _runtime()
    config = json.loads(CONFIG.read_text())
    config.update({'max_epochs': 2, 'batch_size': 2})
    train, val = synthetic_part('train', 1), synthetic_part('validation', 4)
    spec = config['candidates'][0]
    a = fit_one(train, val, spec, 42, config, tmp_path / 'a', tf)
    b = fit_one(train, val, spec, 42, config, tmp_path / 'b', tf)
    np.testing.assert_array_equal(a.predict([.01, 3., 100.]), b.predict([.01, 3., 100.]))


def test_error_scenario_is_explicit_and_does_not_use_traffic():
    from revision.diagnostics.neural_power_comparisons import residual_budget
    np.testing.assert_array_equal(residual_budget([1., 3., 10.], {1.: .1, 10.: .3}), [.1, .3, .3])
    report = json.loads((ROOT / 'revision/reports/task2d/comparisons/configuration_comparison_summary.json').read_text())
    assert not report['traffic_files_read']
    assert not report['controller_rollouts']
    assert not report['model_selection_used_diagnostics']
