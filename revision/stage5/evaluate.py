"""Validation-only frozen baseline/policy replay. No fitting or checkpoint selection."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.contract import require_comparable
from revision.evaluation.static import StaticController
from revision.evaluation.validate_static import write_csv
from revision.stage3.common import inputs,dump
from revision.stage3.pipeline import CachedTwin,ForecastLedger,RuleController,run_controller,summarize
from revision.stage3.pilot import export_steps
from revision.stage4.train import load_bundle
from revision.stage4.rng_isolation import isolated_rng
from .baselines import SlicesExtendedManual,ComcomHistorical

OUT=ROOT/'revision/reports/stage5'
PRIMARY=ROOT/'revision/reports/stage4_rng_isolated'


def verify_frozen(frozen):
    for path,digest in frozen['source_hashes'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Frozen baseline source/input changed: '+path)
    for item in frozen['existing_policy_selections'].values():
        if sha256(ROOT/item['selection_path'])!=item['selection_sha256']:raise ValueError('Policy selection changed')
        for p,h in item['artifacts'].items():
            if sha256(ROOT/p)!=h:raise ValueError('Selected policy/normalizer changed')


def main():
    frozen=json.loads((OUT/'baseline_frozen.json').read_text());verify_frozen(frozen)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    _,val,contract=inputs();ledger=ForecastLedger(ROOT/frozen['configuration']['forecasts']);twin=CachedTwin()
    if twin.identity!=frozen['twin_id']:raise ValueError('Twin identity mismatch')
    controllers=[('slices_extended_manual',SlicesExtendedManual(twin.real,frozen['slices_regions'],
                   frozen['normalization_min_power_w'],frozen['normalization_max_power_w'])),
                 ('strong_rule_stage4',RuleController(twin.real)),('comcom_historical',ComcomHistorical(twin.real))]
    controllers.extend((f'static_{a}',StaticController(a)) for a in range(3))
    comparison={};results=[]
    for name,controller in controllers:
        r=run_controller(val,controller,twin,ledger,contract,name,sha256(OUT/'baseline_frozen.json'))
        repeated=run_controller(val,controller,twin,ledger,contract,name,sha256(OUT/'baseline_frozen.json'))
        if r!=repeated:raise AssertionError('Non-deterministic baseline '+name)
        comparison[name]={**summarize(r),'deterministic_repeat_equal':True};results.append(r);export_steps(name,r,OUT)
        print(name,comparison[name]['energy_wh'],flush=True)
    original=json.loads((PRIMARY/'controller_comparison.json').read_text())['controllers']
    for name,item in frozen['existing_policy_selections'].items():
        algorithm=name.rsplit('_',1)[0];path=ROOT/item['selected_checkpoint']
        with isolated_rng():
            controller=load_bundle(path,algorithm,val,twin,ledger,contract)
            before=pickle.dumps(controller.normalizer.obs_rms)
            r=run_controller(val,controller,twin,ledger,contract,name,sha256(path/'bundle.json'))
            repeated=run_controller(val,controller,twin,ledger,contract,name,sha256(path/'bundle.json'))
            if r!=repeated or pickle.dumps(controller.normalizer.obs_rms)!=before:raise AssertionError('Replay or normalizer failure')
            metrics=summarize(r)
            for k,v in metrics.items():
                if original[name][k]!=v:raise AssertionError('Stage 4 replay differs: '+name+' '+k)
            comparison[name]={**metrics,'selected_checkpoint':item['selected_checkpoint'],
                              'deterministic_repeat_equal':True,'stage4_metrics_exactly_reproduced':True}
            results.append(r);export_steps(name,r,OUT);controller.normalizer.close()
        print(name,comparison[name]['energy_wh'],'exact Stage 4 agreement',flush=True)
    for k,v in summarize(results[1]).items():
        if original['rule_three_action'][k]!=v:raise AssertionError('Immutable strong rule differs: '+k)
    require_comparable(tuple(results))
    dump(OUT/'comparison.json',{'controllers':comparison,'same_horizon':True,'final_test_accessed':False,
         'frozen_sha256':sha256(OUT/'baseline_frozen.json'),'all_static_actions_predicted_safe':all(comparison[f'static_{a}']['qos_violation_rate']==0 for a in range(3))})
    rows=[]
    for name,m in comparison.items():
        rows.append({'controller':name,**{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in m.items()}})
    keys=list(dict.fromkeys(k for row in rows for k in row));write_csv(OUT/'comparison.csv',[{k:r.get(k) for k in keys} for r in rows])
    summaries={}
    for algorithm in frozen['configuration']['algorithms']:
        runs=[comparison[f'{algorithm}_{s}'] for s in frozen['configuration']['seeds']]
        energies=np.asarray([r['energy_wh'] for r in runs]);summaries[algorithm]={'energy_mean_wh':float(energies.mean()),'energy_sample_sd_wh':float(energies.std(ddof=1)),
          'comparisons':{name:{'mean_energy_difference_wh':float(energies.mean()-comparison[name]['energy_wh']),
             'mean_energy_difference_pct':float(100*(energies.mean()/comparison[name]['energy_wh']-1)),
             'energy_wins':int(sum(r['energy_wh']<comparison[name]['energy_wh'] for r in runs)),
             'reward_wins':int(sum(r['total_reward']>comparison[name]['total_reward'] for r in runs))}
             for name in ('slices_extended_manual','strong_rule_stage4','comcom_historical')}}
    dump(OUT/'family_summary.json',summaries);verify_frozen(frozen);twin.real.verify_artifacts()
    print('All 16 controllers: common horizon, exact repeats and unchanged Stage 4 selections.',flush=True)

if __name__=='__main__':main()
