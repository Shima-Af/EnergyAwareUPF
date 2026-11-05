"""Standalone plots and descriptive convergence/forecast diagnostics, no selection changes."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from revision.evaluation.validate_static import write_csv
from revision.evaluation.twin import ROOT
from .common import CFG,OUT,MODELS,dump


def main():
    figures=OUT/'figures';figures.mkdir()
    plt.rcParams.update({'font.size':9,'savefig.dpi':170})
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(figures/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig)
    base=json.loads((OUT/'controller_comparison.json').read_text())['controllers']
    convergence=json.loads((OUT/'convergence.json').read_text());colors=plt.get_cmap('tab10')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for row,alg in enumerate(CFG['algorithms']):
        for j,seed in enumerate(CFG['seeds']):
            name=f'{alg}_{seed}';d=convergence[name]['all_scores'];x=[s['timesteps'] for s in d]
            axes[row,0].plot(x,[s['total_reward'] for s in d],'.-',color=colors(j),label=f'Seed {seed}')
            axes[row,1].plot(x,[s['metrics']['energy_wh'] for s in d],'.-',color=colors(j))
        axes[row,0].axhline(base['rule_three_action']['total_reward'],color='black',linestyle='--',label='Unchanged rule')
        axes[row,1].axhline(base['rule_three_action']['energy_wh'],color='black',linestyle='--')
        axes[row,0].set(title=alg,ylabel='Validation total reward',xlabel='Training interactions');axes[row,0].legend(fontsize=7)
        axes[row,1].set(title=alg,ylabel='Validation attributed energy (Wh)',xlabel='Training interactions')
    save(fig,'validation_convergence')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    keys=['train/entropy_loss','train/approx_kl','train/clip_fraction','train/value_loss','train/policy_gradient_loss','train/explained_variance']
    diagnostic={}
    for alg in CFG['algorithms']:
        for j,seed in enumerate(CFG['seeds']):
            path=MODELS/'controllers'/alg/f'seed_{seed}'
            updates=pd.read_json(path/'training_updates.jsonl',lines=True)
            style='-' if alg=='ppo' else '--'
            for ax,k in zip(axes.flat,keys):
                ax.plot(updates.timesteps,updates[k],style,color=colors(j),alpha=.7,linewidth=.8)
                ax.set(xlabel='Training interactions',ylabel=k)
            last=updates.iloc[-1]
            diagnostic[f'{alg}_{seed}']={'last_optimizer_metrics':{k:float(last[k]) for k in keys},
               'maximum_approx_kl':float(updates['train/approx_kl'].max()),'maximum_clip_fraction':float(updates['train/clip_fraction'].max()),
               'optimizer_updates_logged':len(updates)}
    fig.suptitle('Optimizer diagnostics: solid FF PPO, dashed recurrent PPO; colors identify seeds')
    save(fig,'optimizer_diagnostics')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,alg in zip(axes,CFG['algorithms']):
        for j,seed in enumerate(CFG['seeds']):
            d=pd.read_json(MODELS/'controllers'/alg/f'seed_{seed}'/'episodes.jsonl',lines=True)
            d=d.groupby('timesteps').reward.mean()
            ax.plot(d.index,d.values,color=colors(j),label=f'Seed {seed}')
        ax.set(title=alg,xlabel='Interactions at episode completion',ylabel='Mean training episode reward across four collectors');ax.legend(fontsize=7)
    save(fig,'training_episode_rewards')
    f=pd.read_csv(OUT/'forecasts.csv');f=f[f['index']>=2016];old=pd.read_csv(ROOT/'revision/reports/stage3/forecasts.csv');old=old[old['index']>=2016]
    p=pd.read_csv(OUT/'persistence_forecasts.csv');p=p[p['index']>=2016]
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
    axes[0].plot(f['index'],f.actual_mbps,color='black',label='Actual')
    axes[0].plot(f['index'],f.forecast_mbps,label='Stage 4 selected LSTM')
    axes[0].plot(old['index'],old.forecast_mbps,alpha=.6,label='Stage 3 LSTM')
    axes[0].plot(p['index'],p.forecast_mbps,alpha=.6,label='Persistence');axes[0].legend();axes[0].set_ylabel('Mbps')
    axes[1].plot(f['index'],f.forecast_mbps-f.actual_mbps,label='Stage 4 residual')
    axes[1].plot(p['index'],p.forecast_mbps-p.actual_mbps,alpha=.6,label='Persistence residual')
    axes[1].axhline(0,color='black',linewidth=.7);axes[1].set(xlabel='Common validation source interval',ylabel='Prediction − actual (Mbps)');axes[1].legend()
    save(fig,'forecast_comparison')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,name in zip(axes,CFG['forecast']['candidates']):
        path=MODELS/'forecast'/name;h=pd.read_csv(path/'history.csv');m=json.loads((path/'manifest.json').read_text());scale=m['scale_training_max_mbps']
        ax.plot(h.epoch,np.sqrt(h.training_mse_scaled)*scale,label='Training loss RMSE (dropout active)')
        ax.plot(h.epoch,np.sqrt(h.validation_mse_scaled)*scale,label='Validation RMSE (inference)')
        ax.axvline(m['best_epoch'],color='black',linestyle=':');ax.set(title=name,xlabel='Epoch',ylabel='Mbps');ax.legend(fontsize=7)
    save(fig,'forecast_training')
    names=list(base);fig,axes=plt.subplots(1,2,figsize=(15,6));x=np.arange(len(names))
    axes[0].bar(x,[base[n]['energy_wh'] for n in names]);axes[0].set_ylabel('Attributed validation energy (Wh)')
    bottom=np.zeros(len(names))
    for a,label in enumerate(['DPDK','USR(1)','USR(2)']):
        counts=np.asarray([base[n]['action_counts'][a] for n in names]);axes[1].bar(x,counts,bottom=bottom,label=label);bottom+=counts
    axes[1].set_ylabel('Executed slots');axes[1].legend()
    for ax in axes:ax.set_xticks(x,names,rotation=70,ha='right')
    save(fig,'controller_comparison')
    rows=[{'controller':n,**r,'action_counts':'/'.join(map(str,r['action_counts'])),'requested_action_counts':'/'.join(map(str,r['requested_action_counts']))} for n,r in base.items()]
    write_csv(OUT/'controller_comparison.csv',rows)
    dump(OUT/'optimizer_diagnostics.json',diagnostic)
    print('Plots and optimizer diagnostics completed',flush=True)

if __name__=='__main__':main()
