"""Frozen four-run profiling budget. No traffic or controller outcomes are read."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .neural_power import NeuralPowerModel, create_network, transformed_input
from .power_model import LOAD, POWER, PowerModel, metrics, partition_sweeps
from .refit_usr_power import write_json
from revision.evaluation.twin import ROOT, _runtime, sha256
from revision.evaluation.validate_static import write_csv

CONFIG = ROOT / 'revision/configs/usr_power_nn_v1.json'


def fit_one(training, validation, spec, seed, config, folder, tf):
    if training.role != 'train' or validation.role != 'validation':
        raise ValueError('Neural fitting requires TRAIN learning and VALIDATION selection groups')
    folder.mkdir(exist_ok=False)
    log_x = np.log1p(training.frame[LOAD].to_numpy(dtype=np.float64))
    scaler = {'mean': float(log_x.mean()), 'scale': float(log_x.std()),
              'fit_sweeps': list(training.sweeps), 'fit_rows': len(log_x)}
    x = transformed_input(training.frame[LOAD], scaler, config['domain_mbps'])
    xv = transformed_input(validation.frame[LOAD], scaler, config['domain_mbps'])
    y = training.frame[POWER].to_numpy(dtype=np.float32).reshape(-1, 1)
    yv = validation.frame[POWER].to_numpy(dtype=np.float64)
    net = create_network(tf, spec, seed)
    optimizer = tf.keras.optimizers.Adam(config['initial_learning_rate'])

    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([None, 1], tf.float32)])
    def step(bx, by):
        with tf.GradientTape() as tape:
            pred = net(bx, training=True)
            loss = tf.reduce_mean(tf.square(pred - by))
        optimizer.apply_gradients(zip(tape.gradient(loss, net.trainable_weights), net.trainable_weights))
        return loss

    rng = np.random.default_rng(seed)
    best, best_epoch, weights = float('inf'), 0, None
    stale, lr_stale = 0, 0
    history = []
    for epoch in range(1, config['max_epochs'] + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(x), config['batch_size']):
            indices = order[start:start + config['batch_size']]
            loss = float(step(x[indices], y[indices]))
            if not np.isfinite(loss):
                raise ValueError('Nonfinite training loss; no silent candidate repair')
        prediction = np.asarray(net(xv, training=False)).reshape(-1).astype(np.float64)
        score = float(np.mean((prediction - yv) ** 2))
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation loss')
        improved = score < best
        if improved:
            best, best_epoch, weights = score, epoch, net.get_weights()
            stale, lr_stale = 0, 0
        else:
            stale += 1
            lr_stale += 1
        history.append({'epoch': epoch, 'validation_mse_w2': score,
                        'learning_rate': float(optimizer.learning_rate.numpy()), 'selected_so_far': improved})
        if lr_stale >= config['lr_patience']:
            rate = max(config['minimum_learning_rate'], float(optimizer.learning_rate.numpy()) * config['lr_factor'])
            optimizer.learning_rate.assign(rate)
            lr_stale = 0
        if epoch % 100 == 0:
            print(spec['architecture'], seed, 'epoch', epoch, 'best validation RMSE', best ** .5, flush=True)
        if stale >= config['early_stopping_patience']:
            break
    net.set_weights(weights)
    net.save(folder / 'model.keras')
    name = f"{spec['architecture']}_seed_{seed}"
    metadata = {'schema_version': 1, 'name': name, 'architecture': spec, 'seed': seed,
                'model_file': 'model.keras', 'model_sha256': sha256(folder / 'model.keras'),
                'domain_mbps': config['domain_mbps'], 'input_scaler': scaler,
                'target_units': 'process_attributed_W', 'target_transform': 'identity_watts',
                'output_activation': 'softplus', 'best_epoch': best_epoch, 'epochs_run': epoch,
                'parameter_count': net.count_params(), 'config_sha256': sha256(CONFIG),
                'training_sweeps': list(training.sweeps), 'validation_sweeps': list(validation.sweeps),
                'training_source_rows': [int(i) for i in training.frame.index],
                'source_sha256': config['source_sha256'],
                'implementation_sha256': sha256(Path(__file__)),
                'inference_implementation_sha256': sha256(Path(__file__).with_name('neural_power.py'))}
    write_json(folder / 'manifest.json', metadata)
    write_csv(folder / 'history.csv', history)
    print('Completed', name, 'epochs', epoch, 'selected', best_epoch, 'RMSE', best ** .5, flush=True)
    return NeuralPowerModel(folder / 'manifest.json')


def run(output, model_dir):
    config = json.loads(CONFIG.read_text())
    prior_config_path = ROOT / config['task2c_config']
    prior = json.loads(prior_config_path.read_text())
    if config['split'] != prior['split'] or config['domain_mbps'] != prior['domain_mbps']:
        raise ValueError('Task 2C split/domain must remain identical')
    source = Path(prior['source_csv'])
    if sha256(source) != config['source_sha256']:
        raise ValueError('Profiling source changed')
    frame = pd.read_csv(source)
    frame['parsed_timestamp'] = pd.to_datetime(frame.timestamp, format='ISO8601')
    parts = partition_sweeps(frame, config['split'])
    _, tf, _, runtime = _runtime()
    write_json(output / 'prespecified_protocol.json', {'config': config, 'config_sha256': sha256(CONFIG),
               'task2c_split_sha256': sha256(ROOT / 'revision/reports/task2c/profiling_split.csv'),
               'source_path': str(source), 'source_sha256': sha256(source), 'runtime': runtime,
               'source_rows_by_role': {r: [int(i) for i in p.frame.index] for r, p in parts.items()},
               'new_independent_campaign': False})
    models = []
    for spec in config['candidates']:
        for seed in config['seeds']:
            folder = model_dir / f"{spec['architecture']}_seed_{seed}"
            models.append(fit_one(parts['train'], parts['validation'], spec, seed, config, folder, tf))
    scores = {m.name: metrics(m, parts['validation'].frame) for m in models}
    selected = min(models, key=lambda m: (scores[m.name]['rmse_w'], m.metadata['parameter_count'], m.name))
    selection = {'selected_name': selected.name, 'selected_manifest_path': str(selected.manifest_path),
                 'selected_manifest_sha256': sha256(selected.manifest_path), 'model_sha256': selected.model_hash,
                 'validation_scores': scores, 'config_sha256': sha256(CONFIG),
                 'internal_test_scored_before_selection': False, 'traffic_used_for_selection': False,
                 'search_expanded': False, 'refit_on_train_plus_validation': False}
    write_json(output / 'selection_frozen.json', selection)
    print('FROZEN SELECTION:', selected.name, flush=True)
    # Internal test is accessed for outcome scoring only after the selection above.
    pwl_path = ROOT / 'revision/models/task2c/selected_usr_power.json'
    pwl_selection = json.loads((ROOT / 'revision/reports/task2c/selection_frozen.json').read_text())
    if sha256(pwl_path) != pwl_selection['selected_model_sha256']:
        raise ValueError('Frozen Task 2C candidate changed')
    pwl = PowerModel.load(pwl_path)
    summary, predictions, by_load, by_sweep = {}, [], [], []
    # Unselected NN internal-test scores intentionally not computed.
    for model in (selected, pwl):
        summary[model.name] = {}
        for role, part in parts.items():
            summary[model.name][role] = metrics(model, part.frame)
            for (index, row), power in zip(part.frame.iterrows(), model.predict(part.frame[LOAD])):
                predictions.append({'model': model.name, 'role': role, 'source_row': int(index),
                    'sweep': int(row['Test Set']), 'load_mbps': float(row[LOAD]),
                    'measured_power_w': float(row[POWER]), 'predicted_power_w': float(power),
                    'error_w': float(power - row[POWER])})
            for load, group in part.frame.groupby(LOAD):
                by_load.append({'model': model.name, 'role': role, 'load_mbps': float(load), **metrics(model, group)})
            for sweep, group in part.frame.groupby('Test Set'):
                by_sweep.append({'model': model.name, 'role': role, 'sweep': int(sweep), **metrics(model, group)})
    old_path = ROOT / 'revision/reports/task2c/profiling_predictions.csv'
    old = pd.read_csv(old_path)
    old = old[old.model == 'original_saved_usr_power_descriptive_only']
    for _, row in old.iterrows():
        index = int(row.source_row)
        if frame.loc[index, LOAD] != row.load_mbps or not np.isclose(frame.loc[index, POWER], row.measured_power_w, atol=1e-14, rtol=0):
            raise ValueError('Original reference measurement mismatch')
    original_metrics = json.loads((ROOT / 'revision/reports/task2c/metrics.json').read_text())['candidates']['original_saved_usr_power_descriptive_only']
    summary['original_saved_usr_power_descriptive_only'] = original_metrics
    write_csv(output / 'predictions.csv', predictions)
    write_csv(output / 'errors_by_load.csv', by_load)
    write_csv(output / 'errors_by_sweep.csv', by_sweep)
    grid = np.unique(np.r_[np.geomspace(.01, 5000., 10001), np.linspace(.01, 5000., 10001), np.linspace(.01, 20., 2001)])
    physical = []
    for m in models:
        p = m.predict(grid)
        physical.append({'model': m.name, 'grid_samples': len(grid), 'minimum_w': float(p.min()),
                         'maximum_w': float(p.max()), 'nonnegative_and_finite': bool(np.isfinite(p).all() and (p >= 0).all()),
                         'guarantee': 'softplus(z) in watts; no centered inverse target transform'})
    peers = [m for m in models if m.metadata['architecture']['architecture'] == selected.metadata['architecture']['architecture']]
    seed_predictions = np.stack([m.predict(grid) for m in peers])
    selected_grid, pwl_grid = selected.predict(grid), pwl.predict(grid)
    write_csv(output / 'domain_predictions.csv', [{'load_mbps': float(x), 'selected_nn_w': float(n),
            'task2c_pwl_w': float(p), 'seed_min_w': float(lo), 'seed_max_w': float(hi)}
            for x,n,p,lo,hi in zip(grid,selected_grid,pwl_grid,seed_predictions.min(0),seed_predictions.max(0))])
    write_json(output / 'metrics.json', {'selected': selected.name, 'models': summary,
                'physical_checks': physical, 'pwl_sha256': sha256(pwl_path),
                'original_reference_predictions_sha256': sha256(old_path),
                'original_comparison': 'descriptive only, historical split contains all five sweeps',
                'uncertainty': 'two prespecified seeds, not a calibrated interval; bias may dominate'})
    for m in models: m.verify_artifacts()
    print('Selected-network internal-test scoring complete; no traffic or controller outcomes used.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    args = p.parse_args()
    for folder in (args.output_dir, args.model_dir):
        if not folder.resolve().is_relative_to(ROOT / 'revision'):
            raise ValueError('New artifacts must remain under revision/')
        folder.mkdir(parents=True, exist_ok=False)
    run(args.output_dir, args.model_dir)


if __name__ == '__main__':
    main()
