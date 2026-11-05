"""Freeze four models, validate integration, and measure conditional sensitivity."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from revision.evaluation.twin import ROOT,FittedUPFTwin,sha256
from revision.evaluation.revised_twin import RevisionTwin,CONFIG
from revision.evaluation.validate_static import validation_inputs,run_static_validation,write_csv,DEFAULT_PROTOCOL
from revision.diagnostics.profiling_validity import workload_census
from revision.diagnostics.qos_reliability import threshold_statistics
from revision.diagnostics.neural_power_comparisons import residual_budget
from revision.profiling.neural_power import NeuralPowerModel
from revision.profiling.neural_qos import SimpleQoSModel
from revision.profiling.power_model import LOAD
from revision.profiling.refit_usr_power import write_json


def stats(y,p):
    y,p=np.asarray(y),np.asarray(p);e=p-y;sst=float(np.sum((y-y.mean())**2))
    return {'n':len(y),'mae':float(abs(e).mean()),'rmse':float(np.sqrt(np.mean(e**2))),
            'r2':1-float(np.sum(e**2))/sst if sst else None,'bias':float(e.mean()),
            'residual_q05':float(np.quantile(e,.05)),'residual_q95':float(np.quantile(e,.95))}


def freeze():
    d=json.loads((ROOT/'revision/reports/final_twin/dpdk_power/selection_frozen.json').read_text())
    q=json.loads((ROOT/'revision/reports/task2h/selection_frozen.json').read_text())['modes']
    models={}
    def add(key,path,kind,reason):
        path=Path(path);a=[{'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}]
        if kind!='simple_power':
            m=json.loads(path.read_text());weight=path.parent/m['model_file']
            a.append({'path':str(weight.relative_to(ROOT)),'sha256':sha256(weight)})
        models[key]={'path':str(path.relative_to(ROOT)),'kind':kind,'artifacts':a,'choice_reason':reason}
    add('dpdk_power',d['artifact_path'],d['kind'],'Minimum grouped profiling validation RMSE across prespecified constant/affine/PWL/two constrained neural candidates')
    add('usr_power',ROOT/'revision/models/task2c/selected_usr_power.json','simple_power',
        'Frozen seven-knot fit: continuous nonnegativity and much better low-load accuracy than the Task 2D validation-selected NN; no controller-based selection')
    for mode in ('DPDK','USR'):
        add(mode.lower()+'_qos',q[mode]['selected_manifest'],'bounded_qos','Task 2H validation-selected bounded conditional-mean predictor, not a QoS safety guarantee')
    write_json(CONFIG,{'version':'comcom.revised_twin.v1','models':models,'per_instance_domain_mbps':[.01,5000.],
        'joint_three_action_total_domain_mbps':[.02,5000.],'historical_default_changed':False,
        'qos_specification':{'path':'revision/specifications/qos_score_v1.json','sha256':sha256(ROOT/'revision/specifications/qos_score_v1.json')},
        'implementations':[{'path':p,'sha256':sha256(ROOT/p)} for p in [
            'revision/evaluation/revised_twin.py','revision/evaluation/twin.py',
            'revision/profiling/neural_qos.py','revision/profiling/neural_power.py','revision/profiling/power_model.py']],
        'selection_precedes_workload_diagnostics':True,'controller_results_used_for_selection':False,
        'intended_scope':'exploratory offline model-conditional comparison; no physical/5QI certification'})


def main():
    out=ROOT/'revision/reports/final_twin/integration';out.mkdir(parents=True,exist_ok=False)
    freeze();twin=RevisionTwin();original=FittedUPFTwin()
    write_json(out/'frozen_manifest.json',twin.manifest)
    # Fresh pointwise inference at identical profiling load levels, matching the active adapter.
    rows=[];table=[];byload=[];bysweep=[];tails=[];decomp=[]
    for mode,prefix,action in [('DPDK','dpdk',0),('USR','oai',1)]:
        source=Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')/f'{prefix}_processed.csv'
        f=pd.read_csv(source)
        cache={float(x):twin.raw_prediction(action,float(x)) for x in f[LOAD].unique()}
        f['power_prediction']=[cache[float(x)].power_w for x in f[LOAD]]
        f['qos_prediction']=[cache[float(x)].qos for x in f[LOAD]]
        for target,pcol in [('power_consumption_watts','power_prediction'),('QoS_w','qos_prediction')]:
            component=mode.lower()+('_power' if target.startswith('power') else '_qos')
            for role,groups in [('train',[1,2,3]),('validation',[4]),('internal_test',[5])]:
                g=f[f['Test Set'].isin(groups)];v=stats(g[target],g[pcol])
                if target=='QoS_w':v.update(threshold_statistics(g[target],g[pcol]));v['predicted_safe_coverage']=float((g[pcol]>=.9).mean())
                table.append({'component':component,'role':role,**v})
                for i,r in g.iterrows():rows.append({'component':component,'role':role,'source_row':int(i),'sweep':int(r['Test Set']),
                    'load_mbps':float(r[LOAD]),'measured':float(r[target]),'predicted':float(r[pcol]),'error':float(r[pcol]-r[target])})
                for load,h in g.groupby(LOAD):byload.append({'component':component,'role':role,'load_mbps':float(load),**stats(h[target],h[pcol])})
                for sweep,h in g.groupby('Test Set'):bysweep.append({'component':component,'role':role,'sweep':int(sweep),**stats(h[target],h[pcol])})
                means=g.groupby(LOAD)[target].transform('mean');sst=float(((g[target]-g[target].mean())**2).sum())
                within=float(((g[target]-means)**2).sum())
                decomp.append({'component':component,'role':role,'within_load_fraction_target_variation':within/sst,
                               'in_sample_load_mean_r2':1-within/sst,'measured_sd':float(g[target].std(ddof=0)),
                               'model_prediction_range':float(g[pcol].max()-g[pcol].min())})
        for (sweep,load),g in f.groupby(['Test Set',LOAD]):
            e=g.qos_prediction-g.QoS_w
            tails.append({'mode':mode,'sweep':int(sweep),'role':'train' if sweep<=3 else ('validation' if sweep==4 else 'internal_test'),
              'load_mbps':float(load),'n':len(g),'measured_q05':float(g.QoS_w.quantile(.05)),
              'measured_q10':float(g.QoS_w.quantile(.10)),'measured_min':float(g.QoS_w.min()),
              'measured_mean':float(g.QoS_w.mean()),'prediction':float(g.qos_prediction.iloc[0]),
              'unsafe_count':int((g.QoS_w<.9).sum()),'empirical_unsafe_fraction':float((g.QoS_w<.9).mean()),
              'optimism_q95':float(e.quantile(.95)),'interpretation':'descriptive correlated observations, not calibrated risk'})
    write_json(out/'four_model_metrics.json',table);write_json(out/'variance_decomposition.json',decomp)
    write_csv(out/'profiling_predictions.csv',rows);write_csv(out/'errors_by_load.csv',byload);write_csv(out/'errors_by_sweep.csv',bysweep)
    write_csv(out/'conditional_qos_tails.csv',tails)
    print('Frozen profiling evidence exported; beginning train/validation input census',flush=True)
    census=workload_census(twin,out,DEFAULT_PROTOCOL);write_json(out/'input_census.json',census)
    horizon,contract,metadata=validation_inputs()
    summary,predictions,records=run_static_validation(twin,horizon,contract,metadata['protocol_sha256'])
    summary['evaluation']=metadata;summary['twin_id']=twin.identity
    write_json(out/'static_validation.json',summary);write_csv(out/'static_predictions.csv',predictions);write_csv(out/'static_steps.csv',records)
    nn_selection=json.loads((ROOT/'revision/reports/task2d/selection_frozen.json').read_text())
    alternative=NeuralPowerModel(Path(nn_selection['selected_manifest_path']))
    qrefs={m:SimpleQoSModel(json.loads((ROOT/f'revision/models/task2h/{m}/training_load_mean_pwl.json').read_text())) for m in ['dpdk','usr']}
    residuals={}
    for component in ['dpdk_power','usr_power']:
        z=pd.DataFrame(rows);z=z[(z.component==component)&(z.role=='validation')]
        residuals[component]=z.groupby('load_mbps').error.apply(lambda x:float(abs(x).quantile(.95))).to_dict()
    def sensitivity(loads,scope,timestamps=None):
        results=[]
        for i,total in enumerate(loads):
            raw=[twin.raw_prediction(a,float(total)) for a in range(3)]
            powers=np.asarray([r.power_w for r in raw]);qos=np.asarray([r.qos for r in raw])
            alt=np.asarray([powers[0],alternative.predict([total])[0],2*alternative.predict([total/2])[0]])
            budgets=np.asarray([residual_budget([total],residuals['dpdk_power'])[0],residual_budget([total],residuals['usr_power'])[0],2*residual_budget([total/2],residuals['usr_power'])[0]])
            low=powers-budgets;high=powers+budgets;winner=int(np.argmin(powers))
            ambiguous=any(low[a]<=high[winner] for a in range(3) if a!=winner)
            qalt=np.asarray([qrefs['dpdk' if a==0 else 'usr'].predict([total/(2 if a==2 else 1)])[0] for a in range(3)])
            historical=[original.raw_prediction(a,float(total)) for a in range(3)]
            results.append({'scope':scope,'timestamp':timestamps[i].isoformat() if timestamps else '',
              'total_load_mbps':float(total),'power_dpdk_w':float(powers[0]),'power_usr1_w':float(powers[1]),'power_usr2_w':float(powers[2]),
              'lowest_power_action':winner,'usr_nn_alternative_lowest_power_action':int(np.argmin(alt)),
              'usr_order_differs_with_nn':bool((powers[1]<powers[2])!=(alt[1]<alt[2])),
              'ranking_ambiguous_under_error_scenario':bool(ambiguous),
              'qos_dpdk':float(qos[0]),'qos_usr1':float(qos[1]),'qos_usr2':float(qos[2]),
              'qos_safe_classification_disagreements_with_load_reference':int(((qos>=.9)!=(qalt>=.9)).sum()),
              'historical_invalid_actions':sum(bool(r.failures) for r in historical),
              'all_selected_outputs_valid':not any(r.failures for r in raw)})
        return results
    grid=np.unique(np.r_[np.linspace(.02,400.,1001),[.02,1.,2.,4.,8.,20.,40.,100.,200.,400.]])
    validation_loads=[horizon.partition.timeline.load_mbps[i] for i in horizon.indices]
    sensitivity_rows=sensitivity(grid,'synthetic')+sensitivity(validation_loads,'netflix_validation',horizon.timestamps)
    write_csv(out/'ranking_sensitivity.csv',sensitivity_rows)
    sensitivities={}
    for scope in ['synthetic','netflix_validation']:
        r=[x for x in sensitivity_rows if x['scope']==scope]
        sensitivities[scope]={'points':len(r),'lowest_power_action_changes_with_usr_nn':sum(x['lowest_power_action']!=x['usr_nn_alternative_lowest_power_action'] for x in r),
           'usr_order_changes_with_nn':sum(x['usr_order_differs_with_nn'] for x in r),
           'ambiguous_under_error_scenario':sum(x['ranking_ambiguous_under_error_scenario'] for x in r),
           'qos_reference_classification_disagreements':sum(x['qos_safe_classification_disagreements_with_load_reference'] for x in r),
           'invalid_selected_outputs':sum(not x['all_selected_outputs_valid'] for x in r)}
    write_json(out/'sensitivity_summary.json',{'scopes':sensitivities,
      'error_scenario':'95th absolute validation residual at bracketing load levels, multiplied by instances; independent adversarial signs; not a calibrated interval or operational probability',
      'limitations':['errors shared across actions can be correlated; worst-case signs may not be jointly attainable',
                     'profiling measurements and workload/synthetic surrogate outcomes are separate evidence',
                     'energy rankings are pointwise power/SEC rankings, not controller-policy energy results',
                     'QoS alternatives and conditional tails are diagnostics, not new reward targets or safety guards']})
    twin.verify_artifacts();original.verify_artifacts()
    print('Integration complete; no controller training or final-test traffic outcomes',flush=True)


if __name__=='__main__':main()
