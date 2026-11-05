"""Fixed Task 2G eight-run budget; no traffic, controllers, or default promotion."""

import argparse
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

from .neural_qos import BoundedQoSModel,SimpleQoSModel,TARGET,create_qos_network
from .neural_power import transformed_input
from .power_model import LOAD,partition_sweeps
from .refit_usr_power import write_json
from revision.diagnostics.qos_reliability import audit_metrics,variance_decomposition
from revision.evaluation.twin import ROOT,_runtime,sha256
from revision.evaluation.validate_static import write_csv

CONFIG=ROOT/'revision/configs/qos_validation_v1.json'


def score_predictions(frame,predicted):
    scored=frame.copy();scored['predicted_qos']=predicted
    result=audit_metrics(scored)
    if not len(frame):return result
    shortfall=np.maximum(0,.9-frame[TARGET].to_numpy())
    result.update({'predicted_safe_coverage':float(np.mean(predicted>=.9)),
        'mean_measured_shortfall':float(shortfall.mean()),'p95_measured_shortfall':float(np.quantile(shortfall,.95)),
        'max_measured_shortfall':float(shortfall.max())})
    return result


def fit_one(training,validation,spec,seed,config,folder,tf,source):
    if training.role!='train' or validation.role!='validation':raise ValueError('TRAIN learning and VALIDATION checkpoint selection required')
    folder.mkdir(exist_ok=False)
    log_x=np.log1p(training.frame[LOAD].to_numpy(dtype=np.float64))
    scaler={'mean':float(log_x.mean()),'scale':float(log_x.std()),'fit_sweeps':list(training.sweeps),'fit_rows':len(log_x)}
    domain=config['input']['domain_mbps']
    x=transformed_input(training.frame[LOAD],scaler,domain)
    xv=transformed_input(validation.frame[LOAD],scaler,domain)
    y=training.frame[TARGET].to_numpy(dtype=np.float32).reshape(-1,1)
    yv=validation.frame[TARGET].to_numpy(dtype=np.float64)
    network=create_qos_network(tf,spec,seed)
    optimizer=tf.keras.optimizers.Adam(config['learning_rate'])

    @tf.function(input_signature=[tf.TensorSpec([None,1],tf.float32),tf.TensorSpec([None,1],tf.float32)])
    def step(bx,by):
        with tf.GradientTape() as tape:
            prediction=network(bx,training=True)
            loss=tf.reduce_mean(tf.square(prediction-by))
        optimizer.apply_gradients(zip(tape.gradient(loss,network.trainable_weights),network.trainable_weights))
        return loss

    best,best_epoch,weights=float('inf'),0,None
    stale,lr_stale=0,0
    rng=np.random.default_rng(seed);history=[]
    for epoch in range(1,config['max_epochs_per_fit']+1):
        order=rng.permutation(len(x))
        for start in range(0,len(x),config['batch_size']):
            indices=order[start:start+config['batch_size']]
            if not np.isfinite(float(step(x[indices],y[indices]))):raise ValueError('Nonfinite training loss')
        p=np.asarray(network(xv,training=False)).reshape(-1).astype(np.float64)
        val_mse=float(np.mean((p-yv)**2))
        if not np.isfinite(val_mse):raise ValueError('Nonfinite validation loss')
        improved=val_mse<best
        if improved:best,best_epoch,weights=val_mse,epoch,network.get_weights();stale,lr_stale=0,0
        else:stale+=1;lr_stale+=1
        history.append({'epoch':epoch,'validation_mse':val_mse,'learning_rate':float(optimizer.learning_rate.numpy()),'best_so_far':improved})
        schedule=config['learning_rate_schedule']
        if lr_stale>=schedule['patience']:
            optimizer.learning_rate.assign(max(schedule['minimum'],float(optimizer.learning_rate.numpy())*schedule['factor']))
            lr_stale=0
        if epoch%100==0:print(source['mode'],spec['name'],seed,'epoch',epoch,'best val RMSE',best**.5,flush=True)
        if stale>=config['early_stopping']['patience']:break
    network.set_weights(weights)
    network.save(folder/'model.keras')
    metadata={'schema_version':1,'name':f"{spec['name']}_seed_{seed}",'mode':source['mode'],'architecture':spec,
        'seed':seed,'model_file':'model.keras','model_sha256':sha256(folder/'model.keras'),
        'target':'QoS_w','target_transform':'identity_score','output_activation':'sigmoid',
        'domain_mbps':domain,'input_scaler':scaler,'best_epoch':best_epoch,'epochs_run':epoch,
        'parameter_count':network.count_params(),'training_sweeps':list(training.sweeps),'validation_sweeps':list(validation.sweeps),
        'training_source_rows':[int(i) for i in training.frame.index],'source_sha256':source['sha256'],
        'protocol_sha256':sha256(CONFIG),'score_specification_sha256':sha256(ROOT/config['score_specification']),
        'training_implementation_sha256':sha256(Path(__file__)),
        'inference_implementation_sha256':sha256(Path(__file__).with_name('neural_qos.py')),
        'shared_input_implementation_sha256':sha256(Path(__file__).with_name('neural_power.py'))}
    write_json(folder/'manifest.json',metadata);write_csv(folder/'history.csv',history)
    print('Completed',source['mode'],metadata['name'],'epochs',epoch,'best',best_epoch,'RMSE',best**.5,flush=True)
    return BoundedQoSModel(folder/'manifest.json')


def run(output,model_dir):
    config=json.loads(CONFIG.read_text())
    old_integrity=json.loads((ROOT/'revision/reports/task2g/execution_and_integrity.json').read_text())
    if sha256(CONFIG)!=old_integrity['qos_validation_protocol_sha256']:raise ValueError('Approved protocol changed')
    if sha256(ROOT/config['score_specification'])!=old_integrity['qos_specification_sha256']:raise ValueError('Approved score changed')
    _,tf,_,runtime=_runtime()
    partitions={};sources={}
    for source in config['datasets']:
        path=Path(config['source_directory'])/source['file']
        if sha256(path)!=source['sha256']:raise ValueError('Historical source changed')
        frame=pd.read_csv(path);frame['parsed_timestamp']=pd.to_datetime(frame.timestamp,format='ISO8601')
        if not np.isfinite(frame[[LOAD,TARGET]].to_numpy()).all() or not frame[TARGET].between(0,1).all():raise ValueError('Invalid historical targets')
        parts=partition_sweeps(frame,config['split'])
        if {r:len(p.frame) for r,p in parts.items()}!=config['expected_partition_rows'][source['mode']]:raise ValueError('Unexpected grouped rows')
        partitions[source['mode']]=parts;sources[source['mode']]=source
    write_json(output/'prespecified_protocol.json',{'protocol':config,'protocol_sha256':sha256(CONFIG),
        'score_specification_sha256':sha256(ROOT/config['score_specification']),'runtime':runtime,
        'source_rows':{m:{r:[int(i) for i in p.frame.index] for r,p in parts.items()} for m,parts in partitions.items()},
        'risk_tolerance_approved':False,'user_authorized_profiling_training_without_invented_risk_gate':True,
        'traffic_data_used':False})
    ledger=output/'candidate_runs.jsonl';models={m:[] for m in partitions};summaries={};selection={}
    def event(value):
        with ledger.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    for mode,parts in partitions.items():
        for spec in config['candidates_per_mode']:
            for seed in config['seeds']:
                identity={'mode':mode,'architecture':spec['name'],'seed':seed}
                event({**identity,'status':'started'})
                try:
                    folder=model_dir/mode.lower()/f"{spec['name']}_seed_{seed}"
                    folder.parent.mkdir(exist_ok=True)
                    model=fit_one(parts['train'],parts['validation'],spec,seed,config,folder,tf,sources[mode])
                    models[mode].append(model)
                    event({**identity,'status':'completed','manifest':str(model.manifest_path),'epochs':model.metadata['epochs_run']})
                except Exception as exc:
                    event({**identity,'status':'failed','error':str(exc),'traceback':traceback.format_exc(),'retry':False})
                    print('Candidate failed, no retry:',identity,exc,flush=True)
        val={model.name:score_predictions(parts['validation'].frame,model.predict(parts['validation'].frame[LOAD])) for model in models[mode]}
        winner=min(models[mode],key=lambda m:(val[m.name]['rmse'],m.metadata['parameter_count'],m.name)) if models[mode] else None
        selection[mode]={'selected_name':winner.name if winner else None,'validation_scores':val,
            'selected_manifest':str(winner.manifest_path) if winner else None,
            'selected_manifest_sha256':sha256(winner.manifest_path) if winner else None}
    # BOTH modes are frozen before ANY internal-test outcome scoring.
    write_json(output/'selection_frozen.json',{'modes':selection,'protocol_sha256':sha256(CONFIG),
        'internal_test_scored_before_selection':False,'traffic_used_for_selection':False,
        'search_expanded':False,'default_twin_replaced':False})
    print('FROZEN BOTH NEURAL SELECTIONS',json.dumps({m:s['selected_name'] for m,s in selection.items()}),flush=True)
    original_path=ROOT/'revision/reports/task2e/predictions.csv'
    original=pd.read_csv(original_path)
    predictions,by_load,by_sweep,bands,domains,variance,physical,macro=[],[],[],[],[],{},{},[]
    grid=np.unique(np.r_[np.geomspace(.01,5000.,10001),np.linspace(.01,5000.,10001),np.linspace(.01,20.,2001)])
    grid_rows=[]
    for mode,parts in partitions.items():
        simple=[SimpleQoSModel.fit_training(parts['train'],s['name'],config['input']['domain_mbps']) for s in config['baselines']]
        for model in simple:
            write_json(model_dir/mode.lower()/f'{model.name}.json',{'schema_version':1,**model.metadata,
                'protocol_sha256':sha256(CONFIG),'source_sha256':sources[mode]['sha256']})
        original_mode=original[original['mode']==('dpdk' if mode=='DPDK' else 'usr')].set_index('source_row')
        summaries[mode]={};variance[mode]={};physical[mode]=[]
        def consume(name,role,frame,p):
            if not np.allclose(original_mode.loc[frame.index,'measured_qos'],frame[TARGET],rtol=0,atol=1e-14):raise ValueError('Historical reference rows mismatch')
            scored=frame.copy();scored['predicted_qos']=p
            summaries[mode].setdefault(name,{})[role]=score_predictions(frame,p)
            for i,r in scored.iterrows():
                predictions.append({'mode':mode,'model':name,'role':role,'source_row':int(i),'sweep':int(r['Test Set']),
                    'load_mbps':float(r[LOAD]),'measured_qos':float(r[TARGET]),'predicted_qos':float(r.predicted_qos),
                    'error':float(r.predicted_qos-r[TARGET])})
            load_metrics=[]
            for load,g in scored.groupby(LOAD):
                v=score_predictions(g,g.predicted_qos.to_numpy());load_metrics.append(v)
                by_load.append({'mode':mode,'model':name,'role':role,'load_mbps':float(load),**v})
            macro.append({'mode':mode,'model':name,'role':role,'load_levels':len(load_metrics),
                'macro_mae':float(np.mean([v['mae'] for v in load_metrics])),
                'macro_rmse':float(np.mean([v['rmse'] for v in load_metrics]))})
            for sweep,g in scored.groupby('Test Set'):
                by_sweep.append({'mode':mode,'model':name,'role':role,'sweep':int(sweep),**score_predictions(g,g.predicted_qos.to_numpy())})
            for bound in (200.,400.):
                g=scored[scored[LOAD]<=bound]
                domains.append({'mode':mode,'model':name,'role':role,'max_load_mbps':bound,**score_predictions(g,g.predicted_qos.to_numpy())})
            for anchor,column in [('measured',TARGET),('predicted','predicted_qos')]:
                g=scored[abs(scored[column]-.9)<=.05]
                bands.append({'mode':mode,'model':name,'role':role,'anchor':anchor,'half_width':.05,**score_predictions(g,g.predicted_qos.to_numpy())})
        for model in [*models[mode],*simple]:
            for role,part in parts.items():
                if role=='internal_test' and model in models[mode] and model.name!=selection[mode]['selected_name']:continue
                consume(model.name,role,part.frame,model.predict(part.frame[LOAD]))
            gp=model.predict(grid)
            physical[mode].append({'model':model.name,'grid_points':len(grid),'minimum':float(gp.min()),'maximum':float(gp.max()),
               'all_finite_bounded':bool(np.isfinite(gp).all() and ((gp>=0)&(gp<=1)).all()),
               'continuous_domain_construction':'sigmoid head in score units' if isinstance(model,BoundedQoSModel) else 'training mean or convex interpolation of bounded training load means'})
            for x,q in zip(grid,gp):grid_rows.append({'mode':mode,'model':model.name,'load_mbps':float(x),'predicted_qos':float(q)})
        for role,part in parts.items():
            old=original_mode.loc[part.frame.index,'predicted_qos'].to_numpy()
            consume('original_saved_nn_nonindependent',role,part.frame,old)
            probe=part.frame.copy();probe['predicted_qos']=old
            variance[mode][role]=variance_decomposition(probe)
        for model in models[mode]:model.verify_artifacts()
    write_json(output/'metrics.json',{'modes':summaries,'original_reference_sha256':sha256(original_path),
        'historical_comparison':'nonindependent, original networks trained on rows from all sweeps',
        'no_test_scores_for_unselected_neural_candidates':True})
    write_json(output/'variance_decomposition.json',variance);write_json(output/'bounded_output_checks.json',physical)
    write_json(output/'study_domain_metrics.json',domains);write_json(output/'near_threshold_metrics.json',bands)
    write_csv(output/'predictions.csv',predictions);write_csv(output/'errors_by_load.csv',by_load)
    write_csv(output/'errors_by_sweep.csv',by_sweep);write_csv(output/'macro_load_metrics.csv',macro)
    write_csv(output/'domain_predictions.csv',grid_rows)
    print('Internal scoring complete. No traffic data, controller fits, or automatic promotion.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir=a.output_dir.resolve();a.model_dir=a.model_dir.resolve()
    for folder in (a.output_dir,a.model_dir):
        if not folder.is_relative_to(ROOT/'revision'):raise ValueError('Artifacts must remain under revision/')
        folder.mkdir(parents=True,exist_ok=False)
    run(a.output_dir,a.model_dir)


if __name__=='__main__':main()
