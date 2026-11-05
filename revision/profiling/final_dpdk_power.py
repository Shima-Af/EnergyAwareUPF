"""Fixed grouped DPDK-power comparison for the final revision twin."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .power_model import LOAD,POWER,PowerModel,fit_candidate,partition_sweeps,metrics
from .neural_power import NeuralPowerModel,transformed_input
from .refit_usr_power import artifact,write_json
from revision.evaluation.twin import ROOT,FittedUPFTwin,_runtime,sha256
from revision.evaluation.validate_static import write_csv

CONFIG=ROOT/'revision/configs/final_dpdk_power_v1.json'


def train_nn(train,val,seed,cfg,directory,tf):
    directory.mkdir()
    tf.keras.utils.set_random_seed(seed)
    log=np.log1p(train.frame[LOAD].to_numpy())
    sx={'mean':float(log.mean()),'scale':float(log.std()),'fit_sweeps':[1,2,3],'fit_rows':len(log)}
    x=transformed_input(train.frame[LOAD],sx,cfg['domain_mbps']);v=transformed_input(val.frame[LOAD],sx,cfg['domain_mbps'])
    y=train.frame[POWER].to_numpy(dtype=np.float32).reshape(-1,1);yv=val.frame[POWER].to_numpy()
    net=tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)),tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(1,activation='softplus')])
    opt=tf.keras.optimizers.Adam(cfg['learning_rate'])
    @tf.function(input_signature=[tf.TensorSpec([None,1],tf.float32),tf.TensorSpec([None,1],tf.float32)])
    def step(a,b):
        with tf.GradientTape() as tape:loss=tf.reduce_mean(tf.square(net(a,training=True)-b))
        opt.apply_gradients(zip(tape.gradient(loss,net.trainable_weights),net.trainable_weights));return loss
    rng=np.random.default_rng(seed);best=float('inf');best_epoch=0;weights=None;stale=0;lr_stale=0;history=[]
    for epoch in range(1,cfg['max_epochs']+1):
        order=rng.permutation(len(x))
        for start in range(0,len(x),cfg['batch_size']):
            idx=order[start:start+cfg['batch_size']]
            if not np.isfinite(float(step(x[idx],y[idx]))):raise ValueError('Invalid DPDK training loss')
        pred=np.asarray(net(v,training=False)).ravel();loss=float(np.mean((pred-yv)**2))
        if not np.isfinite(loss):raise ValueError('Invalid validation loss')
        if loss<best:best,best_epoch,weights=loss,epoch,net.get_weights();stale=0;lr_stale=0
        else:stale+=1;lr_stale+=1
        history.append({'epoch':epoch,'validation_mse_w2':loss,'learning_rate':float(opt.learning_rate.numpy())})
        if lr_stale>=cfg['lr_patience']:
            opt.learning_rate.assign(max(cfg['minimum_learning_rate'],float(opt.learning_rate.numpy())*cfg['lr_factor']));lr_stale=0
        if epoch%100==0:print('DPDK power',seed,'epoch',epoch,'best',best**.5,flush=True)
        if stale>=cfg['early_stopping_patience']:break
    net.set_weights(weights);net.save(directory/'model.keras')
    meta={'schema_version':1,'name':f'dpdk_softplus_seed_{seed}','target_units':'process_attributed_W',
          'target_transform':'identity_watts','output_activation':'softplus','model_file':'model.keras',
          'model_sha256':sha256(directory/'model.keras'),'domain_mbps':cfg['domain_mbps'],'input_scaler':sx,
          'seed':seed,'best_epoch':best_epoch,'epochs_run':epoch,'parameter_count':net.count_params(),
          'training_sweeps':[1,2,3],'validation_sweeps':[4],'source_sha256':cfg['source_sha256'],
          'training_source_rows':[int(i) for i in train.frame.index],'config_sha256':sha256(CONFIG),
          'training_code_sha256':sha256(Path(__file__))}
    write_json(directory/'manifest.json',meta);write_csv(directory/'history.csv',history)
    return NeuralPowerModel(directory/'manifest.json')


def main():
    output=ROOT/'revision/reports/final_twin/dpdk_power';directory=ROOT/'revision/models/final_dpdk_power'
    output.mkdir(parents=True,exist_ok=False);directory.mkdir(exist_ok=False)
    cfg=json.loads(CONFIG.read_text());source=Path(cfg['source_csv'])
    if sha256(source)!=cfg['source_sha256']:raise ValueError('DPDK source identity changed')
    f=pd.read_csv(source);f['parsed_timestamp']=pd.to_datetime(f.timestamp,format='ISO8601')
    parts=partition_sweeps(f,cfg['split']);_,tf,_,runtime=_runtime()
    write_json(output/'protocol.json',{'config':cfg,'config_sha256':sha256(CONFIG),'runtime':runtime,
         'source_rows':{r:[int(i) for i in p.frame.index] for r,p in parts.items()}})
    train=parts['train'];domain=tuple(cfg['domain_mbps'])
    candidates=[PowerModel('nonnegative_training_constant','affine',domain,(float(train.frame[POWER].mean()),0.),(),5000.)]
    for name in ['nonnegative_affine','nonnegative_pwl_7']:
        candidates.append(fit_candidate(train,{'name':name,'family':'affine' if name.endswith('affine') else 'piecewise_linear','knots_mbps':cfg['knots_mbps']},domain))
    paths={};counts={}
    for m in candidates:
        path=directory/f'{m.name}.json';write_json(path,artifact(m,train,cfg,CONFIG));paths[m.name]=path
        counts[m.name]=1 if m.name=='nonnegative_training_constant' else len(m.coefficients_w)
    ledger=[]
    for seed in cfg['seeds']:
        try:
            m=train_nn(train,parts['validation'],seed,cfg,directory/f'neural_seed_{seed}',tf)
            candidates.append(m);paths[m.name]=directory/f'neural_seed_{seed}/manifest.json';counts[m.name]=m.metadata['parameter_count']
            ledger.append({'seed':seed,'status':'completed','epochs':m.metadata['epochs_run']})
        except Exception as exc:
            ledger.append({'seed':seed,'status':'failed','reason':str(exc),'retried':False})
    write_json(output/'neural_runs.json',ledger)
    scores={m.name:metrics(m,parts['validation'].frame) for m in candidates}
    selected=min(candidates,key=lambda m:(scores[m.name]['rmse_w'],counts[m.name],m.name))
    selection={'selected_name':selected.name,'artifact_path':str(paths[selected.name]),'artifact_sha256':sha256(paths[selected.name]),
               'kind':'neural_power' if isinstance(selected,NeuralPowerModel) else 'simple_power','validation_scores':scores,
               'candidate_artifacts':{n:{'path':str(p),'sha256':sha256(p)} for n,p in paths.items()},
               'internal_test_scored_before_selection':False,'controller_results_used':False,'search_expanded':False}
    write_json(output/'selection_frozen.json',selection);print('FROZEN DPDK POWER',selected.name,flush=True)
    original=FittedUPFTwin()
    class Old:
        name='historical_nonindependent'
        def predict(self,x):return np.asarray([original.raw_prediction(0,float(v)).power_w for v in x])
    all_metrics={};rows=[];byload=[];bysweep=[]
    for m in [*candidates,Old()]:
        all_metrics[m.name]={}
        for role,part in parts.items():
            if role=='internal_test' and m.name not in (selected.name,'historical_nonindependent'):continue
            all_metrics[m.name][role]=metrics(m,part.frame)
            pred=m.predict(part.frame[LOAD])
            for (i,r),p in zip(part.frame.iterrows(),pred):rows.append({'model':m.name,'role':role,'source_row':int(i),'sweep':int(r['Test Set']),
                'load_mbps':float(r[LOAD]),'measured_power_w':float(r[POWER]),'predicted_power_w':float(p),'error_w':float(p-r[POWER])})
            for load,g in part.frame.groupby(LOAD):byload.append({'model':m.name,'role':role,'load_mbps':float(load),**metrics(m,g)})
            for sweep,g in part.frame.groupby('Test Set'):bysweep.append({'model':m.name,'role':role,'sweep':int(sweep),**metrics(m,g)})
    grid=np.unique(np.r_[np.geomspace(.01,5000.,10001),np.linspace(.01,5000.,10001)])
    checks={m.name:{'minimum_w':float(m.predict(grid).min()),'maximum_w':float(m.predict(grid).max()),
                   'finite_nonnegative':bool(np.isfinite(m.predict(grid)).all() and (m.predict(grid)>=0).all())} for m in [*candidates,Old()]}
    write_json(output/'metrics.json',all_metrics);write_json(output/'domain_checks.json',checks)
    write_csv(output/'predictions.csv',rows);write_csv(output/'errors_by_load.csv',byload);write_csv(output/'errors_by_sweep.csv',bysweep)
    original.verify_artifacts()


if __name__=='__main__':main()
