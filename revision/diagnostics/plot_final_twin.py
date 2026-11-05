"""Render final profiling evidence; no fitting or model selection occurs here."""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision.evaluation.twin import ROOT


def main():
    base = ROOT / 'revision/reports/final_twin'
    out = base / 'figures'
    out.mkdir(exist_ok=False)
    data = pd.read_csv(base / 'integration/profiling_predictions.csv')
    components = ['dpdk_power', 'usr_power', 'dpdk_qos', 'usr_qos']
    labels = ['DPDK power (W)', 'USR power per instance (W)',
              'DPDK composite QoS', 'USR composite QoS']
    colors = {'validation': '#126caa', 'internal_test': '#c1561c'}
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 180})

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(out / f'{name}.{extension}', bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, component, title in zip(axes.flat, components, labels):
        for role, color in colors.items():
            g = data[(data.component == component) & (data.role == role)]
            ax.scatter(g.measured, g.predicted, s=7, alpha=.24,
                       color=color, label=role.replace('_', ' '))
        g = data[data.component == component]
        lo = min(g.measured.min(), g.predicted.min())
        hi = max(g.measured.max(), g.predicted.max())
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=.8)
        ax.set(title=title, xlabel='Measured', ylabel='Predicted')
        ax.legend(fontsize=8)
    save(fig, 'prediction_vs_measurement')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, component, title in zip(axes.flat, components, labels):
        for role, color in colors.items():
            g = data[(data.component == component) & (data.role == role)]
            grouped = g.groupby('load_mbps').error
            x = grouped.mean().index.to_numpy()
            ax.plot(x, grouped.mean(), '.-', color=color, label=role.replace('_', ' '))
            ax.fill_between(x, grouped.quantile(.05), grouped.quantile(.95),
                            color=color, alpha=.14)
        ax.axhline(0, color='black', linewidth=.8)
        ax.set(xscale='log', title=title, xlabel='Per-instance offered load (Mbps)',
               ylabel='Prediction − measurement')
        ax.legend(fontsize=8)
    fig.suptitle('Mean residual and empirical 5th–95th percentiles; not confidence intervals')
    save(fig, 'residuals_by_load')

    tails = pd.read_csv(base / 'integration/conditional_qos_tails.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, mode in enumerate(['DPDK', 'USR']):
        for sweep, g in tails[tails['mode'] == mode].groupby('sweep'):
            axes[0, column].plot(g.load_mbps, g.empirical_unsafe_fraction,
                                 '.-', alpha=.75, label=f'Sweep {sweep}')
            axes[1, column].plot(g.load_mbps, g.measured_q05, '.-', alpha=.75)
        g = tails[(tails['mode'] == mode) & (tails.sweep == 4)]
        axes[1, column].plot(g.load_mbps, g.prediction, 'k--', label='Frozen point predictor')
        for row in (0, 1):
            axes[row, column].set_xscale('log')
            axes[row, column].set_xlabel('Per-instance offered load (Mbps)')
            axes[row, column].set_ylim(-.03, 1.03)
        axes[0, column].set(title=mode, ylabel='Measured fraction with QoS < .90')
        axes[1, column].set_ylabel('Empirical QoS 5th percentile')
        axes[1, column].axhline(.9, color='grey', linewidth=.8)
        axes[0, column].legend(fontsize=8)
        axes[1, column].legend(fontsize=8)
    fig.suptitle('Exposed same-campaign sweeps; descriptive tails, not calibrated risk')
    save(fig, 'qos_conditional_tails')

    sensitivity = pd.read_csv(base / 'integration/ranking_sensitivity.csv')
    g = sensitivity[sensitivity.scope == 'synthetic']
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for action, name in enumerate(['dpdk', 'usr1', 'usr2']):
        axes[0].plot(g.total_load_mbps, g[f'power_{name}_w'], label=name.upper())
    axes[0].set_ylabel('Predicted total attributed power (W)')
    axes[0].legend()
    axes[1].scatter(g.total_load_mbps, g.lowest_power_action, s=5, label='Frozen twin')
    axes[1].scatter(g.total_load_mbps, g.usr_nn_alternative_lowest_power_action,
                    marker='x', s=6, label='Retained USR neural alternative')
    axes[1].set(xlabel='Synthetic total load (Mbps)', ylabel='Lowest-power action', yticks=[0, 1, 2])
    axes[1].legend()
    save(fig, 'configuration_sensitivity')


if __name__ == '__main__':
    main()
