"""Common validation comparisons, paired budget contrasts and fixed-policy forecast replay."""
import json
from pathlib import Path
import numpy as np
import torch
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.static import StaticController
from revision.evaluation.contract import require_comparable,account,actuate
from revision.stage3.pipeline import CachedTwin,ForecastLedger,RuleController,run_controller,summarize
from revision.stage3.pilot import export_steps,load_bundle as load_stage3
from .common import CFG,OUT,MODELS,inputs,dump
from .train import load_bundle


def scaling_accounting(result,twin):
    """Local same-slot alternative only; no claim of a feasible reoptimized trajectory."""
    records=[]
    for step in result.steps:
        if not(step.transition.scale_up or step.transition.scale_down):continue
        old=step.transition.previous;new=step.transition.executed;load=step.load_mbps
        hold=twin.outcome(old,load)
        records.append({'index':step.index,'timestamp':step.context.decision_at.isoformat(),'before':old,'after':new,
           'executed_power_w':step.outcome.power_w,'hold_power_w':hold.power_w,
           'local_energy_saved_wh':(hold.power_w-step.outcome.power_w)*.25,
           'executed_qos':step.outcome.qos,'hold_qos':hold.qos})
    return {'events':len(records),'local_energy_difference_sum_wh':float(sum(r['local_energy_saved_wh'] for r in records)),
       'events_lower_executed_power':sum(r['local_energy_saved_wh']>0 for r in records),
       'interpretation':'One-slot accounting comparison holding previous USR level at the same realized load; excludes later state, cooldown, policy response and reoptimization; not a causal policy ablation',
       'records':records}


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    _,val,contract=inputs();ledger=ForecastLedger(OUT/'forecasts.csv');twin=CachedTwin()
    comparisons={};results=[];convergence={};scaling={}
    for name,controller in [(f'static_{a}',StaticController(a)) for a in range(3)]+[('rule_three_action',RuleController(twin.real))]:
        r=run_controller(val,controller,twin,ledger,contract,name,sha256(ROOT/'revision/stage4/protocol.json'))
        comparisons[name]=summarize(r);results.append(r);export_steps(name,r,OUT)
    for algorithm in CFG['algorithms']:
        for seed in CFG['seeds']:
            name=f'{algorithm}_{seed}';directory=MODELS/'controllers'/algorithm/f'seed_{seed}'
            selection=json.loads((directory/'selection.json').read_text());path=ROOT/selection['selected_checkpoint']
            controller=load_bundle(path,algorithm,val,twin,ledger,contract)
            r=run_controller(val,controller,twin,ledger,contract,name,sha256(path/'bundle.json'))
            comparisons[name]={**summarize(r),'selected_checkpoint':selection['selected_checkpoint'],'actual_training_timesteps':300000}
            results.append(r);scaling[name]=scaling_accounting(r,twin)
            scores=selection['validation_scores'];late=scores[-CFG['interpretation']['late_curve_points']:]
            xs=np.asarray([s['timesteps'] for s in late],float);ys=np.asarray([s['total_reward'] for s in late],float)
            slope=float(np.polyfit((xs-xs[0])/100000,ys,1)[0]);deltas=np.diff(ys)
            trajectory='increasing' if (deltas>0).all() else ('decreasing' if (deltas<0).all() else 'mixed')
            at32=next(s['metrics'] for s in scores if s['timesteps']==32768);end=scores[-1]['metrics']
            convergence[name]={'all_scores':scores,'matched_32768':at32,'endpoint_300000':end,
                'late_points':[{'timesteps':s['timesteps'],'total_reward':s['total_reward'],'energy_wh':s['metrics']['energy_wh']} for s in late],
                'late_reward_slope_per_100k':slope,'late_observed_direction':trajectory,
                'endpoint_minus_32768_reward':end['total_reward']-at32['total_reward'],
                'endpoint_minus_32768_wh':end['energy_wh']-at32['energy_wh'],
                'selected_minus_32768_wh':r.summary['energy_wh']-at32['energy_wh'],
                'selected_minus_endpoint_reward':r.summary['total_reward']-end['total_reward'],
                'not_a_statistical_convergence_test':True}
    require_comparable(tuple(results))
    dump(OUT/'controller_comparison.json',{'validation_only':True,'horizon':val.identity,'controllers':comparisons})
    dump(OUT/'convergence.json',convergence);dump(OUT/'scaling_accounting.json',scaling)
    alternatives={'stage3_lstm':ForecastLedger(ROOT/'revision/reports/stage3/forecasts.csv'),
                  'persistence':ForecastLedger(OUT/'persistence_forecasts.csv')}
    replay={}
    for scenario,alternative in alternatives.items():
        metrics={}
        for name,entry in comparisons.items():
            if name.startswith('static'):controller=StaticController(int(name[-1]))
            elif name=='rule_three_action':controller=RuleController(twin.real)
            else:
                alg='recurrent_ppo' if name.startswith('recurrent_ppo') else 'ppo'
                controller=load_bundle(ROOT/entry['selected_checkpoint'],alg,val,twin,ledger,contract)
            r=run_controller(val,controller,twin,alternative,contract,name,sha256(ROOT/'revision/stage4/protocol.json'))
            metrics[name]=summarize(r)
            primary=__import__('pandas').read_csv(OUT/f'{name}_steps.csv').requested.to_numpy()
            metrics[name]['changed_requests']=int((primary!=np.asarray([s.transition.requested for s in r.steps])).sum())
        replay[scenario]=metrics
    pilot=json.loads((ROOT/'revision/reports/stage3/controller_comparison.json').read_text())['controllers']
    forward={}
    for name,entry in pilot.items():
        if 'selected_checkpoint' not in entry:continue
        alg='recurrent_ppo' if name.startswith('recurrent_ppo') else 'ppo'
        controller=load_stage3(ROOT/entry['selected_checkpoint'],alg,val,twin,alternatives['stage3_lstm'],contract)
        r=run_controller(val,controller,twin,ledger,contract,name,sha256(ROOT/'revision/stage4/protocol.json'))
        forward[name]=summarize(r)
    dump(OUT/'forecast_sensitivity.json',{'fixed_stage4_policies':replay,'fixed_stage3_policies_new_forecasts':forward,
       'no_retraining_reselection':True,'interpretation':'Changing forecasts at inference with fixed weights/normalizers; not a fully trained forecast-factorial ablation'})
    dump(OUT/'comparison_summary.json',summaries(comparisons,convergence))
    print('Validation and forecast sensitivity complete',flush=True)


def summaries(comparisons,convergence):
    result={};rule=comparisons['rule_three_action']
    keys=['energy_wh','mean_sec_j_per_mbit','aggregate_sec_j_per_mbit','total_reward','qos_violation_rate','mean_qos_shortfall',
        'type_switches','scale_up_events','scale_down_events','blocked_requests','min_predicted_qos']
    for alg in CFG['algorithms']:
        rows=[comparisons[f'{alg}_{s}'] for s in CFG['seeds']]
        result[alg]={'n':len(rows),'metrics':{k:{'mean':float(np.mean([r[k] for r in rows])),
                'sample_sd':float(np.std([r[k] for r in rows],ddof=1)),'minimum':float(min(r[k] for r in rows)),
                'maximum':float(max(r[k] for r in rows))} for k in keys},
            'lower_energy_than_rule':sum(r['energy_wh']<rule['energy_wh'] for r in rows),
            'higher_reward_than_rule':sum(r['total_reward']>rule['total_reward'] for r in rows),
            'mean_energy_percent_over_rule':100*(np.mean([r['energy_wh'] for r in rows])/rule['energy_wh']-1),
            'endpoint_improves_reward_over_32k':sum(convergence[f'{alg}_{s}']['endpoint_minus_32768_reward']>0 for s in CFG['seeds'])}
    return {'validation_exploratory':True,'not_sufficient_reviewer_replication':True,'algorithms':result}

if __name__=='__main__':main()
