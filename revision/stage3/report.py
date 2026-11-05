"""Plot completed validation evidence and write machine-readable claim diagnostics."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import OUT,dump


def main():
    data=json.loads((OUT/'controller_comparison.json').read_text())['controllers']
    out=OUT/'figures';out.mkdir()
    names=list(data);x=np.arange(len(names))
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,key,label in zip(axes,['energy_wh','mean_sec_j_per_mbit','qos_violation_rate'],['Attributed energy (Wh)','Mean SEC (J/Mbit)','Predicted-score shortfall fraction']):
        ax.bar(x,[data[n][key] for n in names]);ax.set_xticks(x,names,rotation=65,ha='right');ax.set_ylabel(label)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'controller_comparison.{ext}',bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(len(names),1,figsize=(12,12),sharex=True)
    for ax,name in zip(axes,names):
        d=pd.read_csv(OUT/f'{name}_steps.csv');ax.step(np.arange(len(d)),d.executed,where='post');ax.set_ylabel(name,rotation=0,ha='right',fontsize=8);ax.set_yticks([0,1,2])
    axes[-1].set_xlabel('Common validation slot');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'action_timelines.{ext}',bbox_inches='tight')
    plt.close(fig)
    f=pd.read_csv(OUT/'forecasts.csv');f=f[f['index']>=2016]
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(f['index'],f.actual_mbps,label='Actual interval load');ax.plot(f['index'],f.forecast_mbps,label='Causal selected LSTM');ax.set(xlabel='Source interval index (validation only)',ylabel='Load (Mbps)');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'forecast_validation.{ext}',bbox_inches='tight')
    plt.close(fig)
    baseline=data['rule_three_action'];summary={}
    for family in ['ppo','recurrent_ppo']:
        runs=[data[n] for n in names if n.startswith(family+'_') and (family!='ppo' or not n.startswith('recurrent'))]
        energy=np.asarray([r['energy_wh'] for r in runs]);sec=np.asarray([r['mean_sec_j_per_mbit'] for r in runs])
        summary[family]={'n_seeds':len(runs),'energy_mean_wh':float(energy.mean()),'energy_sample_sd_wh':float(energy.std(ddof=1)),
          'energy_delta_from_rule_mean_wh':float(energy.mean()-baseline['energy_wh']),
          'energy_relative_to_rule_percent':float(100*(energy.mean()/baseline['energy_wh']-1)),
          'mean_sec':float(sec.mean()),'sec_relative_to_rule_percent':float(100*(sec.mean()/baseline['mean_sec_j_per_mbit']-1)),
          'seeds_lower_energy_than_rule':int((energy<baseline['energy_wh']).sum()),
          'seeds_with_any_scaling':sum(r['scale_up_events']+r['scale_down_events']>0 for r in runs),
          'all_seed_metrics':runs}
    dump(OUT/'claim_diagnostics.json',{'validation_exploratory_only':True,'not_inferential_evidence':True,
       'rule_reference':baseline,'families':summary,'risk_threshold_approved':False})

if __name__=='__main__':main()
