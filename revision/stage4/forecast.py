"""Prespecified LSTM early stopping and causal expanding-prefix forecasting."""
import json
from pathlib import Path
import numpy as np
from revision.evaluation.twin import ROOT,sha256,_runtime
from revision.evaluation.validate_static import write_csv
from revision.stage3.forecast import arrays,predictions
from .common import CFG,OUT,MODELS,inputs,dump,forecast_metrics


def fit(loads,times,fit_stop,validation_targets,calendar,path,selection_stop):
    if fit_stop>2016 or max(validation_targets)>=selection_stop or selection_stop>2520:raise ValueError('Unauthorized forecast roles')
    _,tf,_,runtime=_runtime();cfg=CFG['forecast'];tf.keras.utils.set_random_seed(cfg['seed'])
    scale=float(max(loads[:fit_stop]));indices=np.arange(96,fit_stop)
    x=arrays(loads,times,indices,scale,calendar);y=np.asarray(loads)[indices,None]/scale
    vx=arrays(loads,times,validation_targets,scale,calendar);vy=np.asarray(loads)[validation_targets,None]/scale
    net=tf.keras.Sequential([tf.keras.layers.Input((96,x.shape[-1])),tf.keras.layers.SpatialDropout1D(.1),
        tf.keras.layers.LSTM(128,return_sequences=True,dropout=.2,recurrent_dropout=0.),tf.keras.layers.LSTM(64,dropout=.2,recurrent_dropout=0.),
        tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(1,activation='softplus')])
    opt=tf.keras.optimizers.Adam(cfg['learning_rate'])
    @tf.function(input_signature=[tf.TensorSpec([None,96,x.shape[-1]],tf.float32),tf.TensorSpec([None,1],tf.float32)])
    def step(a,b):
        with tf.GradientTape() as tape:loss=tf.reduce_mean((net(a,training=True)-b)**2)
        opt.apply_gradients(zip(tape.gradient(loss,net.trainable_weights),net.trainable_weights));return loss
    rng=np.random.default_rng(cfg['seed']);best=float('inf');weights=None;stale=0;lr_stale=0;history=[];best_epoch=0
    for epoch in range(1,cfg['max_epochs']+1):
        order=rng.permutation(len(x));losses=[]
        for start in range(0,len(x),cfg['batch_size']):
            ii=order[start:start+cfg['batch_size']];losses.append(float(step(x[ii],y[ii].astype(np.float32))))
        vp=np.asarray(net(vx,training=False));loss=float(np.mean((vp-vy)**2))
        if not np.isfinite(loss) or not np.isfinite(losses).all():raise ValueError('Nonfinite forecast training')
        if loss<best-cfg['min_delta_scaled_mse']:
            best=loss;weights=net.get_weights();best_epoch=epoch;stale=0;lr_stale=0
        else:stale+=1;lr_stale+=1
        history.append({'epoch':epoch,'training_mse_scaled':float(np.mean(losses)),'validation_mse_scaled':loss,'learning_rate':float(opt.learning_rate.numpy())})
        if lr_stale>=cfg['lr_patience']:
            opt.learning_rate.assign(max(cfg['min_lr'],float(opt.learning_rate.numpy())*cfg['lr_factor']));lr_stale=0
        if stale>=cfg['patience']:break
    net.set_weights(weights);path.mkdir(parents=True);net.save(path/'model.keras');write_csv(path/'history.csv',history)
    dump(path/'manifest.json',{'fit_source_rows_half_open':[0,fit_stop],'parameter_targets_half_open':[96,fit_stop],
       'early_stopping_targets_half_open':[int(min(validation_targets)),int(max(validation_targets))+1],
       'selection_stop_exclusive':selection_stop,'scale_fit_stop_exclusive':fit_stop,'scale_training_max_mbps':scale,
       'calendar':calendar,'seed':cfg['seed'],'epochs_run':epoch,'best_epoch':best_epoch,'best_scaled_validation_mse':best,
       'model_sha256':sha256(path/'model.keras'),'config_sha256':sha256(ROOT/'revision/stage4/protocol.json'),
       'implementation_sha256':sha256(Path(__file__)),'runtime':runtime})
    print('Forecast fit',path.name,'epochs',epoch,'selected',best_epoch,'validation RMSE',best**.5*scale,flush=True)
    return net,scale


def main():
    OUT.mkdir(parents=True,exist_ok=True);train,val,_=inputs();t=train.partition.timeline
    loads=np.asarray(t.load_mbps[:2520]);times=t.timestamps[:2520]
    dump(OUT/'forecast_protocol_frozen.json',{'config':CFG,'implementation_sha256':sha256(Path(__file__)),
       'stage3_forecast_code_sha256':sha256(ROOT/'revision/stage3/forecast.py'),
       'data_sha256':t.source_sha256,'train_horizon':train.identity,'validation_horizon':val.identity})
    models={};scores={}
    for name in CFG['forecast']['candidates']:
        calendar=name=='lstm_calendar';model,scale=fit(loads,times,2016,np.arange(2016,2520),calendar,MODELS/'forecast'/name,2520)
        p=predictions(model,scale,loads,times,list(val.indices),calendar)
        scores[name]=forecast_metrics(loads[list(val.indices)],p);models[name]=(model,scale,p)
    selected=min(scores,key=lambda name:(scores[name]['rmse_mbps'],name));calendar=selected=='lstm_calendar'
    dump(OUT/'forecast_selection.json',{'selected':selected,'validation_metrics':scores,'selection_uses_validation_only':True,
       'manifest_path':str((MODELS/'forecast'/selected/'manifest.json').relative_to(ROOT)),
       'manifest_sha256':sha256(MODELS/'forecast'/selected/'manifest.json'),'no_controller_selection':True,'no_search_expansion':True})
    rows=[]
    def add(indices,p,fit_stop,select_stop,kind):
        for i,v in zip(indices,p):
            if max(fit_stop,select_stop)>i:raise ValueError('Forecast looks ahead')
            rows.append({'index':int(i),'target_timestamp':times[i].isoformat(),'observed_through':times[i-1].isoformat(),
               'fit_stop_exclusive':int(fit_stop),'selection_stop_exclusive':int(select_stop),'forecast_mbps':float(v),'actual_mbps':float(loads[i]),'kind':kind})
    add(range(96,192),loads[95:191],0,0,'explicit_cold_start_persistence')
    bounds=CFG['forecast']['prefix_boundaries']
    for stop,end in zip(bounds,bounds[1:]):
        # Inner holdout ends before every emitted training forecast, including checkpoint selection.
        inner_stop=96+int((stop-96)*CFG['forecast']['prefix_internal_training_fraction'])
        model,scale=fit(loads,times,inner_stop,np.arange(inner_stop,stop),calendar,MODELS/'forecast'/f'prefix_{stop}',stop)
        add(range(stop,end),predictions(model,scale,loads,times,list(range(stop,end)),calendar),inner_stop,stop,'causal_prefix_lstm')
    # Full model selection uses validation as explicitly permitted. Selection cutoff is not a claim of online retraining.
    # Validation forecasts use parameters trained only through 2015; the rows' fit cutoff records that separation.
    for i,v in zip(val.indices,models[selected][2]):
        rows.append({'index':i,'target_timestamp':times[i].isoformat(),'observed_through':times[i-1].isoformat(),
          'fit_stop_exclusive':2016,'selection_stop_exclusive':2520,'forecast_mbps':float(v),'actual_mbps':float(loads[i]),'kind':'validation_selected_lstm'})
    write_csv(OUT/'forecasts.csv',rows)
    persistence=[{**r,'forecast_mbps':float(loads[r['index']-1]),'fit_stop_exclusive':0,'selection_stop_exclusive':0,'kind':'persistence_alternative'} for r in rows]
    write_csv(OUT/'persistence_forecasts.csv',persistence)
    accuracy={}
    for role,indices in [('train',train.indices),('validation',val.indices)]:
        for name,records in [('lstm',rows),('persistence',persistence)]:
            g=[r for r in records if r['index'] in indices]
            accuracy[f'{role}_{name}']=forecast_metrics([r['actual_mbps'] for r in g],[r['forecast_mbps'] for r in g])
    dump(OUT/'forecast_accuracy.json',accuracy);print('Selected',selected,accuracy,flush=True)

if __name__=='__main__':main()
