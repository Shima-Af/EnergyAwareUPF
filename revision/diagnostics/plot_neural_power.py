"""Render frozen Task 2D results; no fitting or model selection."""

from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / 'revision/reports/task2d'
    selected = json.loads((folder / 'selection_frozen.json').read_text())['selected_name']
    grid = pd.read_csv(folder / 'domain_predictions.csv')
    rows = pd.read_csv(folder / 'predictions.csv')
    old = pd.read_csv(root / 'revision/reports/task2c/low_load_comparison.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for index, axis in enumerate(axes):
        g = grid[grid.load_mbps <= 20] if index == 0 else grid
        axis.plot(g.load_mbps, g.selected_nn_w, color='tab:blue', label='Selected constrained NN')
        axis.plot(g.load_mbps, g.task2c_pwl_w, color='black', label='Frozen Task 2C seven knots')
        axis.fill_between(g.load_mbps, g.seed_min_w, g.seed_max_w, alpha=.2, color='tab:blue',
                          label='Two NN seeds (not a CI)')
        for role, marker in [('train', 'o'), ('validation', 's'), ('internal_test', '^')]:
            observed = rows[(rows.model == selected) & (rows.role == role)]
            means = observed.groupby('load_mbps').measured_power_w.mean()
            if index == 0:
                means = means[means.index <= 20]
            axis.scatter(means.index, means.values, s=20, marker=marker, label=f'Measured {role.replace("_", " ")} mean')
        axis.set_xlabel('Recorded per-instance DL/UL objective (Mbps)')
        axis.set_ylabel('Process-attributed power (W)')
        axis.axhline(0, color='gray', linewidth=.7)
        axis.grid(alpha=.2)
    axes[0].plot(old.load_mbps, old.original_power_w, color='tab:red', linewidth=.9,
                 label='Original NN (descriptive)')
    axes[0].set_xlim(.01, 20)
    axes[0].set_title('Low-load bias and positivity')
    axes[1].set_xscale('log')
    axes[1].set_title('Declared interpolation domain')
    for axis in axes:
        axis.legend(fontsize=6.8)
    for extension in ('png', 'pdf'):
        path = folder / f'neural_power_comparison.{extension}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
