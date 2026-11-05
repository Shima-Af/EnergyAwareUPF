"""Plots only: no parameter selection or numerical metric changes."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from revision.evaluation.twin import ROOT
from revision.stage3.common import dump
from revision.evaluation.validate_static import write_csv

OUT=ROOT/'revision/reports/stage5'

def main():
    frozen=json.loads((OUT/'baseline_frozen.json').read_text());metrics=json.loads((OUT/'comparison.json').read_text())['controllers']
    figs=OUT/'figures';figs.mkdir()
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(figs/f'{name}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    names=list(metrics);fig,axes=plt.subplots(len(names)+1,1,figsize=(14,22),sharex=True)
    df=pd.read_csv(OUT/'slices_extended_manual_steps.csv');x=pd.to_datetime(df.timestamp)
    axes[0].plot(x,df.actual_load_mbps,label='Realized load',color='black',linewidth=.7)
    axes[0].plot(x,df.forecast_load_mbps,label='Common Stage 4 forecast',alpha=.7,linewidth=.7)
    axes[0].set_ylabel('Mbps');axes[0].legend(fontsize=7,ncol=2)
    for ax,name in zip(axes[1:],names):
        d=pd.read_csv(OUT/f'{name}_steps.csv')
        ax.step(x,d.executed,where='post',linewidth=.8,label='Executed')
        blocked=d.blocked.astype(str).str.lower().eq('true')
        ax.scatter(x[blocked],d.requested[blocked],color='red',s=5,label='Blocked request')
        ax.set_yticks([0,1,2],['DPDK','USR1','USR2'],fontsize=7);ax.set_ylim(-.2,2.2)
        ax.text(1.002,.5,name,transform=ax.transAxes,fontsize=7,va='center');ax.grid(alpha=.2)
    axes[-1].set_xlabel('Common validation timestamps (timezone unspecified in source)');save(fig,'configuration_timelines')
    fig,ax=plt.subplots(figsize=(12,5));ax.bar(names,[metrics[n]['energy_wh'] for n in names])
    ax.set_ylabel('Attributed validation energy (Wh)');ax.tick_params(axis='x',rotation=75);save(fig,'energy_comparison')
    curves=pd.read_csv(OUT/'profiling_curves.csv');fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for a,label in enumerate(['DPDK','USR(1)','USR(2)']):
        d=curves[curves.action==a];axes[0].plot(d.load_mbps,d.slices_utility,label=label)
        if a<2:axes[1].plot(d.load_mbps,d.comcom_utility,label=label)
    for r in frozen['slices_regions'][:-1]:
        t=r['right'];axes[0].axvline(t,color='black',linestyle='--',linewidth=.7)
        axes[0].axvspan(t-26,t+26,color='gray',alpha=.15)
    axes[0].set_ylabel('SLICES extension utility');axes[1].set_ylabel('Historical COMCOM utility');axes[1].set_xlabel('Synthetic total offered load (Mbps)')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save(fig,'threshold_hysteresis')
    regions=[]
    for i,r in enumerate(frozen['slices_regions']):
        lo=r['left']+26 if i else r['left'];hi=r['right']-26 if i<len(frozen['slices_regions'])-1 else r['right']
        regions.append({**r,'request_interior_lower_mbps':lo,'request_interior_upper_mbps':hi,
                        'empty_request_interior':lo>=hi,'boundary_policy':'Strict beyond crossover margin; domain endpoints inclusive'})
    write_csv(OUT/'hysteresis_regions.csv',regions)
    dump(OUT/'threshold_audit.json',{'slices_regions_with_hysteresis':regions,'historical_utility_regions':frozen['comcom_utility_regions'],
      'slices_old_95_6_reused':False,'comcom_old_thresholds_reused':False,
      'historical_margin_executed':False,'strong_rule_unchanged':True,
      'strong_rule_traffic_thresholds':'Not a fixed-threshold algorithm; feasibility and current-relative power deadband evaluated at every forecast',
      'control_parameter_tuning':False})
    print('Comparison plots and threshold audit written',flush=True)

if __name__=='__main__':main()
