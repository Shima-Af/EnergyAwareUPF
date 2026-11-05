"""Prespecified paired effects, convergence, costs and figures; no model changes."""
import itertools,json,math
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision.evaluation.validate_static import write_csv
from .common import CFG,ROOT,OUT,directory,run_name,dump


def paired(values):
    d=np.asarray(values,float);mean=float(d.mean());sd=float(d.std(ddof=1));half=float(t.ppf(.975,len(d)-1)*sd/math.sqrt(len(d)))
    null=np.array([np.mean(d*np.array(s)) for s in itertools.product([-1,1],repeat=len(d))])
    p=float(np.mean(abs(null)>=abs(mean)))
    return {'n':len(d),'differences':d.tolist(),'mean_difference':mean,'sample_sd':sd,
      'mean_difference_t95_lower':mean-half,'mean_difference_t95_upper':mean+half,
      'energy_wins_negative':int(sum(d<0)),'ties':int(sum(d==0)),'exact_two_sided_sign_flip_p':p}


def holm(items):
    order=sorted(items,key=lambda k:items[k]['exact_two_sided_sign_flip_p']);previous=0.
    for j,k in enumerate(order):
        previous=max(previous,min(1.,(len(order)-j)*items[k]['exact_two_sided_sign_flip_p']))
        items[k]['holm_adjusted_p']=previous


def main():
    source=json.loads((OUT/'comparison.json').read_text());workloads=source['workloads'];summaries={};effects={};convergence={};costs={};family_rows=[];pairrows=[]
    for workload,data in workloads.items():
        families={};rows=data['controllers_selected'];endpoints=data['controllers_endpoint'];effects[workload]={}
        for algorithm in CFG['algorithms']:
            for schema in CFG['schemas']:
                key=f'{algorithm}_{schema}';names=[run_name(algorithm,schema,s) for s in CFG['seeds']]
                family={'algorithm':algorithm,'schema':schema}
                for metric in ['energy_wh','mean_sec_j_per_mbit','aggregate_sec_j_per_mbit','total_reward','type_switches','scale_up_events','scale_down_events','blocked_requests','mean_predicted_qos','min_predicted_qos','qos_violation_rate','inference_median_ms','inference_p95_ms']:
                    a=np.array([rows[n][metric] for n in names]);family[metric+'_mean']=float(a.mean());family[metric+'_sample_sd']=float(a.std(ddof=1))
                family['endpoint_energy_mean_wh']=float(np.mean([endpoints[n]['energy_wh'] for n in names]));family['endpoint_energy_sample_sd_wh']=float(np.std([endpoints[n]['energy_wh'] for n in names],ddof=1))
                family['endpoint_reward_mean']=float(np.mean([endpoints[n]['total_reward'] for n in names]));family['occupancy_mean']=[float(np.mean([rows[n]['action_counts'][a] for n in names])) for a in range(3)]
                family['energy_wins_vs_strong']=sum(rows[n]['energy_wh']<rows['strong_rule_stage4']['energy_wh'] for n in names)
                family['reward_wins_vs_strong']=sum(rows[n]['total_reward']>rows['strong_rule_stage4']['total_reward'] for n in names)
                families[key]=family;family_rows.append({'workload':workload,'configuration':key,**{k:json.dumps(v) if isinstance(v,list) else v for k,v in family.items()}})
        summaries[workload]=families
        pairs=list(CFG['primary_pairs'])+[['ppo_forecast','ppo_instant'],['recurrent_ppo_forecast','recurrent_ppo_instant']]
        for left,right in pairs:
            key=f'{left} minus {right}';d=[rows[f'{left}_{s}']['energy_wh']-rows[f'{right}_{s}']['energy_wh'] for s in CFG['seeds']]
            item=paired(d);item['left']=left;item['right']=right
            item['reward_mean_difference']=float(np.mean([rows[f'{left}_{s}']['total_reward']-rows[f'{right}_{s}']['total_reward'] for s in CFG['seeds']]))
            item['sec_mean_difference']=float(np.mean([rows[f'{left}_{s}']['mean_sec_j_per_mbit']-rows[f'{right}_{s}']['mean_sec_j_per_mbit'] for s in CFG['seeds']]))
            item['endpoint_energy_mean_difference']=float(np.mean([endpoints[f'{left}_{s}']['energy_wh']-endpoints[f'{right}_{s}']['energy_wh'] for s in CFG['seeds']]))
            item['primary_test']=workload=='netflix' and [left,right] in CFG['primary_pairs']
            effects[workload][key]=item
            for seed,difference in zip(CFG['seeds'],d):pairrows.append({'workload':workload,'contrast':key,'seed':seed,'energy_difference_wh':difference,'primary':item['primary_test']})
        d=[(rows[f'recurrent_ppo_hybrid_{s}']['energy_wh']-rows[f'ppo_hybrid_{s}']['energy_wh'])-(rows[f'recurrent_ppo_history_{s}']['energy_wh']-rows[f'ppo_history_{s}']['energy_wh']) for s in CFG['seeds']]
        effects[workload]['forecast_by_recurrence_interaction']={**paired(d),'primary_test':workload=='netflix'}
        if workload=='netflix':holm({k:v for k,v in effects[workload].items() if v['primary_test']})
    for algorithm in CFG['algorithms']:
        for schema in CFG['schemas']:
            for seed in CFG['seeds']:
                name=run_name(algorithm,schema,seed);path=directory(algorithm,schema,seed);selection=json.loads((path/'selection.json').read_text());r=json.loads((path/'result.json').read_text())
                scores=selection['validation_scores'];late=scores[-3:];x=np.array([p['timesteps'] for p in late]);y=np.array([p['total_reward'] for p in late]);diff=np.diff(y)
                direction='increasing' if (diff>0).all() else ('decreasing' if (diff<0).all() else 'mixed')
                end=scores[-1];at32=next(p for p in scores if p['timesteps']==32768)
                convergence[name]={'selected_step':r['selected_timesteps'],'late_direction':direction,'late_reward_slope_per100k':float(np.polyfit((x-x[0])/1e5,y,1)[0]),
                  'endpoint_minus_selected_energy_wh':end['metrics']['energy_wh']-r['energy_wh'],
                  'endpoint_minus_32k_reward':end['total_reward']-at32['total_reward'],
                  'endpoint_minus_32k_energy_wh':end['metrics']['energy_wh']-at32['metrics']['energy_wh'],
                  'late_reward_range':float(np.ptp(y)),'all_scores':scores}
                costs[name]={k:r[k] for k in ['algorithm','schema','seed','training_wall_seconds_including_checkpoint_validation','training_cpu_seconds_including_checkpoint_validation','process_peak_rss_mib','pre_model_process_peak_rss_mib','policy_parameter_bytes','inference_median_ms','inference_p95_ms','inference_calls_timed']}
    for name,c in costs.items():
        c['worker_inference_median_ms']=c['inference_median_ms'];c['worker_inference_p95_ms']=c['inference_p95_ms']
        c['inference_median_ms']=workloads['netflix']['controllers_selected'][name]['inference_median_ms']
        c['inference_p95_ms']=workloads['netflix']['controllers_selected'][name]['inference_p95_ms']
        c['latency_scope']='Post-training sequential CPU1 benchmark; raw worker timings also retained'
    dump(OUT/'family_summary.json',summaries);dump(OUT/'paired_effects.json',effects);dump(OUT/'convergence.json',convergence);dump(OUT/'computational_cost.json',costs)
    write_csv(OUT/'family_summary.csv',family_rows);write_csv(OUT/'paired_seed_differences.csv',pairrows)
    figures=OUT/'figures';figures.mkdir()
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(figures/f'{name}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for row,workload in enumerate(['netflix','youtube']):
        for col,kind in enumerate(['selected','endpoint']):
            ax=axes[row,col]
            for algorithm,offset in [('ppo',-.18),('recurrent_ppo',.18)]:
                means=[];sds=[]
                for schema in CFG['schemas']:
                    m=summaries[workload][f'{algorithm}_{schema}'];means.append(m['energy_wh_mean'] if kind=='selected' else m['endpoint_energy_mean_wh']);sds.append(m['energy_wh_sample_sd'] if kind=='selected' else m['endpoint_energy_sample_sd_wh'])
                ax.bar(np.arange(4)+offset,means,width=.35,yerr=sds,label=algorithm)
            for key,style in [('strong_rule_stage4','--'),('slices_recalibrated_v2',':'),('comcom_historical','-.')]:ax.axhline(workloads[workload]['controllers_selected'][key]['energy_wh'],linestyle=style,color='black',alpha=.65,label=key)
            ax.set_xticks(np.arange(4),CFG['schemas']);ax.set(title=f'{workload} / {kind}',ylabel='Attributed validation energy Wh');ax.legend(fontsize=6)
    save(fig,'energy_observation_comparison')
    fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=True)
    for row,algorithm in enumerate(CFG['algorithms']):
        for col,schema in enumerate(CFG['schemas']):
            ax=axes[row,col]
            for seed in CFG['seeds']:
                d=convergence[run_name(algorithm,schema,seed)]['all_scores'];ax.plot([r['timesteps'] for r in d],[r['total_reward'] for r in d],'.-',linewidth=.7,label=str(seed))
            ax.axhline(workloads['netflix']['controllers_selected']['strong_rule_stage4']['total_reward'],color='black',linestyle='--',linewidth=.8)
            ax.set(title=f'{algorithm}/{schema}',ylabel='Validation total reward',xlabel='Interactions');ax.legend(fontsize=6)
    save(fig,'validation_learning_curves')
    for workload in ['netflix','youtube']:
        primary=[run_name(a,s,s0) for a in CFG['algorithms'] for s in ['history','hybrid'] for s0 in CFG['seeds']]
        names=['slices_extended_manual_26','slices_recalibrated_v2','strong_rule_stage4','comcom_historical']+primary
        fig,axes=plt.subplots(len(names),1,figsize=(14,25),sharex=True)
        for ax,name in zip(axes,names):
            d=pd.read_csv(OUT/workload/f'{name}_selected_steps.csv');ax.step(pd.to_datetime(d.timestamp),d.executed,where='post',linewidth=.65)
            ax.set_yticks([0,1,2],['D','U1','U2'],fontsize=6);ax.text(1.002,.5,name,transform=ax.transAxes,fontsize=6);ax.grid(alpha=.15)
        save(fig,workload+'_configuration_timelines')
    fig,axes=plt.subplots(2,2,figsize=(13,8));optimizer={}
    for row,algorithm in enumerate(CFG['algorithms']):
        for schema in CFG['schemas']:
            curves=[];episodes=[]
            for seed in CFG['seeds']:
                name=run_name(algorithm,schema,seed);path=directory(algorithm,schema,seed)
                d=pd.read_json(path/'training_updates.jsonl',lines=True);entropy=-pd.to_numeric(d['train/entropy_loss'],errors='coerce')
                curves.append(entropy.to_numpy())
                e=pd.read_json(path/'episodes.jsonl',lines=True).groupby('timesteps').reward.mean();episodes.append(e)
                optimizer[name]={'final_entropy_estimate':float(entropy.iloc[-1]),
                  'maximum_mean_approx_kl':float(pd.to_numeric(d['train/approx_kl'],errors='coerce').max()),
                  'final_value_loss':float(pd.to_numeric(d['train/value_loss'],errors='coerce').iloc[-1]),
                  'optimizer_updates':len(d)}
            a=np.asarray(curves);axes[row,0].plot(d.timesteps,a.mean(0),label=schema)
            axes[row,0].fill_between(d.timesteps,a.mean(0)-a.std(0,ddof=1),a.mean(0)+a.std(0,ddof=1),alpha=.1)
            e=pd.concat(episodes,axis=1).mean(axis=1);axes[row,1].plot(e.index,e,label=schema)
        axes[row,0].set(title=algorithm,ylabel='Policy entropy estimate',xlabel='Interactions');axes[row,0].legend()
        axes[row,1].set(title=algorithm,ylabel='Mean completed training episode reward',xlabel='Interactions');axes[row,1].legend()
    save(fig,'training_entropy_and_episode_rewards');dump(OUT/'optimizer_summary.json',optimizer)
    print('All paired comparisons, convergence and figures written',flush=True)

if __name__=='__main__':main()
