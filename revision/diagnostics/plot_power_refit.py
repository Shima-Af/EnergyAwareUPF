"""Render frozen profiling diagnostics; no fitting or traffic access."""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision.profiling.power_model import PowerModel


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / 'revision/reports/task2c'
    rows = pd.read_csv(output / 'profiling_predictions.csv')
    measured = rows[rows.model == 'nonnegative_pwl_7']
    low = pd.read_csv(output / 'low_load_comparison.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for index, axis in enumerate(axes):
        for role, marker in [('train', 'o'), ('validation', 's'), ('internal_test', '^')]:
            means = measured[measured.role == role].groupby('load_mbps').measured_power_w.mean()
            if index == 0:
                means = means[means.index <= 20]
            axis.scatter(means.index, means.values, s=22, marker=marker, label=f'Measured {role.replace("_", " ")} mean')
        axis.set_xlabel('Recorded per-instance DL/UL objective (Mbps)')
        axis.set_ylabel('Process-attributed power (W)')
        axis.axhline(0, color='gray', linewidth=.7)
        axis.grid(alpha=.2)
    axes[0].plot(low.load_mbps, low.original_power_w, color='tab:red', label='Original saved MLP')
    axes[0].plot(low.load_mbps, low.selected_power_w, color='black', label='Selected seven-knot model')
    axes[0].fill_between(low.load_mbps, low.jackknife_min_w, low.jackknife_max_w,
                         color='gray', alpha=.35, label='Training jackknife range (not CI)')
    axes[0].set_xlim(.01, 20)
    axes[0].set_title('Low-load behavior, without clipping')
    x = np.geomspace(.01, 5000., 2001)
    for name, style in [('nonnegative_affine', ':'), ('nonnegative_pwl_7', '-'),
                        ('nonnegative_pwl_training_levels', '--')]:
        model = PowerModel.load(root / f'revision/models/task2c/{name}.json')
        axes[1].plot(x, model.predict(x), style, linewidth=1.3,
                     label={'nonnegative_affine': 'Constrained affine', 'nonnegative_pwl_7': 'Selected seven knots',
                            'nonnegative_pwl_training_levels': '22 knots'}[name])
    original = rows[rows.model == 'original_saved_usr_power_descriptive_only'].groupby('load_mbps').predicted_power_w.mean()
    axes[1].plot(original.index, original.values, color='tab:red', linewidth=.9, label='Original at measured loads')
    axes[1].set_xscale('log')
    axes[1].set_title('Declared interpolation domain')
    for axis in axes:
        axis.legend(fontsize=6.7)
    for extension in ('png', 'pdf'):
        path = output / f'power_refit_comparison.{extension}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
