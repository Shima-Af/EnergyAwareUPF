"""Nonnegative power in original watts, with an explicit interpolation domain."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits

LOAD = 'Objective Throughput (DL/UL)'
POWER = 'power_consumption_watts'


@dataclass(frozen=True)
class ProfilingPartition:
    role: str
    sweeps: tuple[int, ...]
    frame: object  # pandas DataFrame, retaining original source row identifiers.

    def __post_init__(self):
        if self.role not in ('train', 'validation', 'internal_test'):
            raise ValueError('Unknown profiling partition')
        if not self.sweeps or set(self.frame['Test Set'].unique()) != set(self.sweeps):
            raise ValueError('Partition sweep identities do not match its data')


def partition_sweeps(frame, split: dict) -> dict[str, ProfilingPartition]:
    if set(split) != {'train', 'validation', 'internal_test'}:
        raise ValueError('Three explicit profiling roles are required')
    groups = [int(s) for values in split.values() for s in values]
    if len(groups) != len(set(groups)) or set(groups) != set(frame['Test Set'].unique()):
        raise ValueError('Sweep groups must be disjoint and cover the entire dataset')
    parts = {r: ProfilingPartition(r, tuple(groups), frame[frame['Test Set'].isin(groups)].copy())
             for r, groups in split.items()}
    for case, rows in frame.groupby('Test Case'):
        if rows['Test Set'].nunique() != 1:
            raise ValueError(f'Case block crosses sweeps: {case}')
    # Group-level chronological order, not merely disjoint row positions.
    ordered = [parts[r].frame['parsed_timestamp'] for r in ('train', 'validation', 'internal_test')]
    if not ordered[0].max() < ordered[1].min() or not ordered[1].max() < ordered[2].min():
        raise ValueError('Profiling sweep partitions must be chronological')
    return parts


def linear_basis(loads, knots) -> np.ndarray:
    x, nodes = np.asarray(loads, dtype=np.float64).reshape(-1), np.asarray(knots, dtype=np.float64)
    if len(nodes) < 2 or not np.all(np.diff(nodes) > 0):
        raise ValueError('Knots must be strictly increasing')
    if not np.isfinite(x).all() or np.any(x < nodes[0]) or np.any(x > nodes[-1]):
        raise ValueError('Load is outside the declared interpolation domain')
    right = np.searchsorted(nodes, x, side='right')
    right[right == len(nodes)] = len(nodes) - 1
    left = right - 1
    weight = (x - nodes[left]) / (nodes[right] - nodes[left])
    matrix = np.zeros((len(x), len(nodes)), dtype=np.float64)
    matrix[np.arange(len(x)), left] = 1 - weight
    matrix[np.arange(len(x)), right] = weight
    return matrix


@dataclass(frozen=True)
class PowerModel:
    name: str
    family: str
    domain_mbps: tuple[float, float]
    coefficients_w: tuple[float, ...]
    knots_mbps: tuple[float, ...] = ()
    input_scale_mbps: float = 1.0

    def __post_init__(self):
        low, high = self.domain_mbps
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
            raise ValueError('Invalid declared power-model domain')
        if not self.coefficients_w or any(not math.isfinite(c) or c < 0 for c in self.coefficients_w):
            raise ValueError('Power parameters must be finite and nonnegative in watts')
        if self.family == 'affine':
            if len(self.coefficients_w) != 2 or not math.isfinite(self.input_scale_mbps) or self.input_scale_mbps <= 0:
                raise ValueError('Affine model needs two coefficients and a positive training input scale')
        elif self.family == 'piecewise_linear':
            if (len(self.knots_mbps) != len(self.coefficients_w)
                    or any(not math.isfinite(k) for k in self.knots_mbps)
                    or self.knots_mbps[0] != low or self.knots_mbps[-1] != high
                    or any(b <= a for a, b in zip(self.knots_mbps, self.knots_mbps[1:]))):
                raise ValueError('PWL model needs ordered knots covering its declared domain')
        else:
            raise ValueError('Unsupported power-model family')

    def predict(self, loads) -> np.ndarray:
        x = np.asarray(loads, dtype=np.float64).reshape(-1)
        if not np.isfinite(x).all() or np.any(x < self.domain_mbps[0]) or np.any(x > self.domain_mbps[1]):
            raise ValueError('Load is outside the declared power-model domain; no extrapolation or clamping')
        if self.family == 'affine':
            result = self.coefficients_w[0] + self.coefficients_w[1] * (x / self.input_scale_mbps)
        else:
            result = linear_basis(x, self.knots_mbps) @ np.asarray(self.coefficients_w)
        if not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError('Constrained power model returned an invalid prediction')
        return result  # No output projection/clipping and no target inverse scaler.

    @classmethod
    def from_dict(cls, values: dict):
        return cls(values['name'], values['family'], tuple(values['domain_mbps']),
                   tuple(values['coefficients_w']), tuple(values.get('knots_mbps', ())),
                   values.get('input_scale_mbps', 1.0))

    @classmethod
    def load(cls, path: Path):
        artifact = json.loads(path.read_text())
        if artifact['schema_version'] != 1 or artifact['target_units'] != 'process_attributed_W':
            raise ValueError('Unsupported power artifact schema or units')
        return cls.from_dict(artifact['model'])


def fit_candidate(training: ProfilingPartition, spec: dict, domain: tuple[float, float]) -> PowerModel:
    if training.role != 'train':
        raise ValueError('Power parameter learning requires profiling TRAIN sweeps')
    x = training.frame[LOAD].to_numpy(dtype=np.float64)
    y = training.frame[POWER].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('Invalid training power labels; labels must not be repaired implicitly')
    if x.min() != domain[0] or x.max() != domain[1]:
        raise ValueError('Training observations must cover the declared domain endpoints')
    scale = float(np.max(x))
    if spec['family'] == 'affine':
        matrix, knots = np.column_stack([np.ones(len(x)), x / scale]), ()
    elif spec['family'] == 'piecewise_linear':
        knots = (tuple(float(v) for v in np.unique(x)) if spec['knots_mbps'] == 'unique_training_load_levels'
                 else tuple(spec['knots_mbps']))
        matrix = linear_basis(x, knots)
    else:
        raise ValueError('Unsupported power family')
    with threadpool_limits(limits=1):
        coefficients, _ = nnls(matrix, y)
    return PowerModel(spec['name'], spec['family'], domain, tuple(float(c) for c in coefficients), knots, scale)


def metrics(model, frame, low_limit=20.0) -> dict:
    y = frame[POWER].to_numpy(dtype=np.float64)
    predicted = model.predict(frame[LOAD].to_numpy(dtype=np.float64))
    error = predicted - y
    low = frame[LOAD].to_numpy() <= low_limit
    sst = float(np.sum((y - y.mean()) ** 2))
    return {'n': len(y), 'rmse_w': float(np.sqrt(np.mean(error ** 2))),
            'mae_w': float(np.mean(abs(error))), 'bias_w': float(np.mean(error)),
            'r2': 1 - float(np.sum(error ** 2)) / sst if sst else None,
            'low_load_n': int(low.sum()), 'low_load_rmse_w': float(np.sqrt(np.mean(error[low] ** 2))) if low.any() else None,
            'low_load_mae_w': float(np.mean(abs(error[low]))) if low.any() else None,
            'low_load_bias_w': float(np.mean(error[low])) if low.any() else None,
            'negative_predictions': int(np.sum(predicted < 0))}


def select_candidate(models: tuple[PowerModel, ...], validation: ProfilingPartition):
    if validation.role != 'validation':
        raise ValueError('Complexity selection requires profiling VALIDATION sweeps only')
    scores = {m.name: metrics(m, validation.frame) for m in models}
    winner = min(models, key=lambda m: (scores[m.name]['rmse_w'], len(m.coefficients_w), m.name))
    return winner, scores
