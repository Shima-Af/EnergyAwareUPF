"""Freeze40 Netflix selections, then score unchanged baselines and policy transfer."""
import json,pickle
import numpy as np
import torch
from revision.evaluation.contract import require_comparable
from revision.evaluation.validate_static import write_csv
from revision.stage4.rng_isolation import isolated_rng
from .common import CFG,ROOT,OUT,dump,sha256,horizons,directory,run_name
from .pipeline import CachedTwin,Ledger,run_controller,summarize,export_steps
from .train import load_bundle,verify_inputs
from .baselines import definitions,Timed


def main():
    verify_inputs();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    selections={}
    for algorithm in CFG['algorithms']:
        for schema in CFG['schemas']:
            for seed in CFG['seeds']:
                name=run_name(algorithm,schema,seed);path=directory(algorithm,schema,seed)
                selection=json.loads((path/'selection.json').read_text())
                if selection['youtube_used'] or selection['final_test_used'] or selection['actual_timesteps']!=300000:raise ValueError('Selection scope failure')
                expected=min(selection['validation_scores'],key=lambda r:(-r['total_reward'],r['timesteps']))
                if selection['selected_checkpoint']!=expected['checkpoint']:raise ValueError('Selection changed')
                selected=ROOT/selection['selected_checkpoint'];endpoint=path/'checkpoint_300000'
                selections[name]={'algorithm':algorithm,'schema':schema,'seed':seed,'selection_sha256':sha256(path/'selection.json'),
                  'selected':str(selected.relative_to(ROOT)),'endpoint':str(endpoint.relative_to(ROOT)),
                  'selected_bundle_sha256':sha256(selected/'bundle.json'),'endpoint_bundle_sha256':sha256(endpoint/'bundle.json')}
    dump(OUT/'all_netflix_selections_frozen.json',{'selections':selections,'count':40,'youtube_outcomes_used':False,'protocol_sha256':sha256(ROOT/'revision/stage7/protocol.json')})
    twin=CachedTwin();allrows={}
    for workload in ['netflix','youtube']:
        _,val,contract=horizons(workload);ledger=Ledger(ROOT/CFG['forecast_ledgers'][workload]);transfer=workload=='youtube'
        out=OUT/workload;out.mkdir(exist_ok=True);selected_rows={};endpoints={};results=[]
        for name,raw in definitions(twin,workload):
            controller=Timed(raw);r=run_controller(val,controller,twin,ledger,contract,name,sha256(ROOT/'revision/stage7/protocol.json'),'hybrid',transfer=transfer)
            latency=list(controller.latency_ns)
            repeated=run_controller(val,controller,twin,ledger,contract,name,sha256(ROOT/'revision/stage7/protocol.json'),'hybrid',transfer=transfer)
            assert repeated==r
            selected_rows[name]={**summarize(r),'family':'baseline','deterministic_repeat_equal':True,
              'inference_median_ms':float(np.median(latency)/1e6),'inference_p95_ms':float(np.percentile(latency,95)/1e6)}
            export_steps(name,r,out/f'{name}_selected_steps.csv',ledger);results.append(r)
        for name,item in selections.items():
            algorithm,schema,seed=item['algorithm'],item['schema'],item['seed']
            for kind,dest in [('selected',selected_rows),('endpoint',endpoints)]:
                path=ROOT/item[kind]
                stored=json.loads((directory(algorithm,schema,seed)/'result.json').read_text()) if workload=='netflix' and kind=='selected' else {}
                with isolated_rng():
                    controller=load_bundle(path,algorithm,schema,twin,contract);before=pickle.dumps(controller.normalizer.obs_rms)
                    r=run_controller(val,controller,twin,ledger,contract,name,sha256(path/'bundle.json'),schema,transfer=transfer)
                    latency=list(controller.latency_ns);states=list(controller.state_norms)
                    repeated=run_controller(val,controller,twin,ledger,contract,name,sha256(path/'bundle.json'),schema,transfer=transfer)
                    assert r==repeated and before==pickle.dumps(controller.normalizer.obs_rms)
                    metrics=summarize(r)
                    if workload=='netflix':
                        archived=json.loads((path/'validation.json').read_text())['metrics']
                        assert metrics==archived
                    if stored:
                        for key,value in metrics.items():assert stored[key]==value
                    dest[name]={**stored,**metrics,'post_training_sequential_cpu_benchmark':True,'algorithm':algorithm,'schema':schema,'seed':seed,'checkpoint':str(path.relative_to(ROOT)),
                      'deterministic_repeat_equal':True,'recurrent_calls':controller.calls if algorithm=='recurrent_ppo' else 0,
                      'recurrent_state_norm_min':min(states) if states else None,
                      'inference_median_ms':float(np.median(latency)/1e6),'inference_p95_ms':float(np.percentile(latency,95)/1e6)}
                    suffix='selected_replay' if stored else kind
                    export_steps(name,r,out/f'{name}_{suffix}_steps.csv',ledger);results.append(r);controller.normalizer.close()
            print('Scored',workload,name,'selected',selected_rows[name]['energy_wh'],'endpoint',endpoints[name]['energy_wh'],flush=True)
        require_comparable(tuple(results))
        # Include all worker-generated selected trajectories in the common horizon check.
        for name in selections:
            import csv
            rows=list(csv.DictReader((out/f'{name}_selected_steps.csv').open()))
            assert [int(r['source_index']) for r in rows]==list(val.indices)
            assert [r['timestamp'] for r in rows]==[t.isoformat() for t in val.timestamps]
        allrows[workload]={'controllers_selected':selected_rows,'controllers_endpoint':endpoints,'horizon':val.identity,
          'timestamps':[t.isoformat() for t in val.timestamps],'transfer_without_policy_training':transfer}
        flat=[{'workload':workload,'endpoint_kind':kind,'controller':name,**{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in m.items()}}
              for kind,group in [('selected',selected_rows),('endpoint',endpoints)] for name,m in group.items()]
        keys=list(dict.fromkeys(k for r in flat for k in r));write_csv(out/'comparison.csv',[{k:r.get(k) for k in keys} for r in flat])
    dump(OUT/'comparison.json',{'workloads':allrows,'policy_fits':40,'final_test_used':False,'forecast_refits':0,'youtube_used_for_selection':False})
    print('Completed both validation workloads; no checkpoint reselection',flush=True)

if __name__=='__main__':main()
