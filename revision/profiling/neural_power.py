"""Opt-in, constrained neural power inference; no default twin integration."""

import json
from pathlib import Path
import numpy as np
from revision.evaluation.twin import _runtime, sha256


def transformed_input(loads, scaler, domain):
    x = np.asarray(loads, dtype=np.float64).reshape(-1)
    if (not np.isfinite(x).all() or np.any(x < domain[0]) or np.any(x > domain[1])):
        raise ValueError('Outside declared neural power interpolation domain')
    if not np.isfinite([scaler['mean'], scaler['scale']]).all() or scaler['scale'] <= 0:
        raise ValueError('Invalid training-only input scaler')
    return ((np.log1p(x) - scaler['mean']) / scaler['scale']).astype(np.float32).reshape(-1, 1)


def create_network(tf, specification, seed):
    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.layers.Input(shape=(1,))]
    for index, width in enumerate((64, 32)):
        layers.append(tf.keras.layers.Dense(width, activation='relu'))
        if specification['batch_norm']:
            layers.append(tf.keras.layers.BatchNormalization())
        if index == 0 and specification['dropout']:
            layers.append(tf.keras.layers.Dropout(specification['dropout'], seed=seed))
    # The output is physical watts; there is no centered inverse target transform.
    layers.append(tf.keras.layers.Dense(1, activation='softplus'))
    return tf.keras.Sequential(layers)


class NeuralPowerModel:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.resolve()
        self.metadata = json.loads(manifest_path.read_text())
        if (self.metadata['schema_version'] != 1 or self.metadata['target_units'] != 'process_attributed_W'
                or self.metadata['target_transform'] != 'identity_watts'
                or self.metadata['output_activation'] != 'softplus'):
            raise ValueError('Unsupported constrained neural artifact')
        self.model_path = manifest_path.parent / self.metadata['model_file']
        self.model_hash = sha256(self.model_path)
        if self.model_hash != self.metadata['model_sha256']:
            raise ValueError('Neural model artifact hash mismatch')
        _, tf, _, _ = _runtime()
        self.network = tf.keras.models.load_model(self.model_path, compile=False)
        if self.network.layers[-1].get_config()['activation'] != 'softplus':
            raise ValueError('Neural power head must be nonnegative by construction')
        if any(not np.isfinite(w).all() for w in self.network.get_weights()):
            raise ValueError('Nonfinite neural parameters')
        self.name = self.metadata['name']

    def predict(self, loads):
        x = transformed_input(loads, self.metadata['input_scaler'], self.metadata['domain_mbps'])
        if len(x) == 0:
            return np.empty(0, dtype=np.float64)
        values = np.asarray(self.network(x, training=False)).reshape(-1).astype(np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid neural power output; no clipping or repair')
        return values

    def verify_artifacts(self):
        if sha256(self.model_path) != self.model_hash:
            raise ValueError('Neural model artifact changed')
