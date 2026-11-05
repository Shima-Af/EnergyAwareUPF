"""Expanding-prefix LSTM forecasts. No archived forecast column is consumed."""
import json
from pathlib import Path
import numpy as np
from revision.evaluation.twin import _runtime,sha256,ROOT
from revision.evaluation.validate_static import write_csv
from .common import CFG,OUT,MODELS,inputs,dump,forecast_metrics


def arrays(loads,times,targets,scale,calendar):
    x=np.asarray([loads[i-96:i] for i in targets],np.float32)[...,None]/scale
    if calendar:
        c=[]
        for i in targets:
            t=times[i];h=(t.hour*60+t.minute)/1440
            c.append([np.sin(2*np.pi*h),np.cos(2*np.pi*h),np.sin(2*np.pi*t.weekday()/7),np.cos(2*np.pi*t.weekday()/7),float(t.weekday()>4),float(8<=t.hour<18)])
        x=np.concatenate([x,np.repeat(np.asarray(c,np.float32)[:,None,:],96,axis=1)],axis=2)
    return x


def fit(loads,times,stop,calendar,path):
    _,tf,_,runtime=_runtime();cfg=CFG['forecast'];tf.keras.utils.set_random_seed(cfg['seed'])
    scale=float(max(loads[:stop]));targets=np.arange(96,stop)
    x=arrays(loads,times,targets,scale,calendar);y=np.asarray(loads)[targets,None]/scale
    model=tf.keras.Sequential([tf.keras.layers.Input((96,x.shape[-1])),
       tf.keras.layers.SpatialDropout1D(.1),tf.keras.layers.LSTM(128,return_sequences=True,dropout=.2,recurrent_dropout=0.),
       tf.keras.layers.LSTM(64,dropout=.2,recurrent_dropout=0.),tf.keras.layers.Dropout(.2),
       tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(1,activation='softplus')])
    optimizer=tf.keras.optimizers.Adam(cfg['learning_rate'])
    @tf.function(input_signature=[tf.TensorSpec([None,96,x.shape[-1]],tf.float32),tf.TensorSpec([None,1],tf.float32)])
    def step(a,b):
        with tf.GradientTape() as tape:loss=tf.reduce_mean((model(a,training=True)-b)**2)
        optimizer.apply_gradients(zip(tape.gradient(loss,model.trainable_weights),model.trainable_weights));return loss
    rng=np.random.default_rng(cfg['seed']);history=[]
    for epoch in range(cfg['epochs']):
        order=rng.permutation(len(x));losses=[]
        for start in range(0,len(x),cfg['batch_size']):
            idx=order[start:start+cfg['batch_size']];losses.append(float(step(x[idx],y[idx].astype(np.float32))))
        history.append({'epoch':epoch+1,'training_mse_scaled':float(np.mean(losses))})
    path.mkdir(parents=True);model.save(path/'model.keras')
    dump(path/'manifest.json',{'fit_source_rows_half_open':[0,stop],'target_rows_half_open':[96,stop],
       'last_training_timestamp':times[stop-1].isoformat(),'scale_training_max_mbps':scale,
       'calendar':calendar,'history':history,'model_sha256':sha256(path/'model.keras'),
       'protocol_sha256':sha256(ROOT/'revision/stage3/protocol.json'),'code_sha256':sha256(Path(__file__)),
       'runtime':runtime,'seed':cfg['seed'],'uses_validation_labels':False})
    return model,scale


def predictions(model,scale,loads,times,targets,calendar):
    x=arrays(loads,times,targets,scale,calendar)
    p=np.concatenate([np.asarray(model(x[i:i+128],training=False)).ravel() for i in range(0,len(x),128)])*scale
    if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid LSTM forecasts; no repair')
    return p


def main():
    train,val,_=inputs();timeline=train.partition.timeline;loads=np.asarray(timeline.load_mbps[:val.partition.stop]);times=timeline.timestamps[:val.partition.stop]
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    dump(OUT/'forecast_protocol_frozen.json',{'config':CFG['forecast'],'train_horizon':train.identity,'validation_horizon':val.identity,
       'implementation_sha256':sha256(Path(__file__)),'historical_script_sha256':sha256(ROOT/'scripts/traffic/train_traffic_predictor.py'),
       'historical_model_sha256':sha256(ROOT/'saved_models/prediction_models/traffic_predictor.keras')})
    candidates={};models={}
    for name in CFG['forecast']['candidates']:
        cal=name=='lstm_calendar';model,scale=fit(loads,times,2016,cal,MODELS/'forecast'/name)
        p=predictions(model,scale,loads,times,list(val.indices),cal)
        candidates[name]=forecast_metrics(loads[list(val.indices)],p);models[name]=(model,scale,p)
        print('Forecast candidate',name,candidates[name],flush=True)
    selected=min(candidates,key=lambda n:(candidates[n]['rmse_mbps'],n));cal=selected=='lstm_calendar'
    dump(OUT/'forecast_selection.json',{'selected':selected,'validation_metrics':candidates,'selection_uses_validation_only':True,
       'no_final_test':True,'no_search_expansion':True})
    rows=[]
    def add(indices,p,stop,kind):
        for i,v in zip(indices,p):
            if stop>i:raise ValueError('Forecast parameters from future')
            rows.append({'index':int(i),'target_timestamp':times[i].isoformat(),'observed_through':times[i-1].isoformat(),
                 'fit_stop_exclusive':int(stop),'forecast_mbps':float(v),'actual_mbps':float(loads[i]),'kind':kind})
    add(range(96,192),loads[95:191],0,'explicit_cold_start_persistence')
    bounds=CFG['forecast']['prefix_boundaries']
    for stop,end in zip(bounds,bounds[1:]):
        model,scale=fit(loads,times,stop,cal,MODELS/'forecast'/f'walkforward_{stop}')
        add(range(stop,end),predictions(model,scale,loads,times,list(range(stop,end)),cal),stop,'walkforward_lstm')
        print('Walk-forward forecast complete',stop,end,flush=True)
    add(val.indices,models[selected][2],2016,'frozen_validation_lstm')
    write_csv(OUT/'forecasts.csv',rows)
    metrics={}
    for role,indices in [('train',train.indices),('validation',val.indices)]:
        g=[r for r in rows if r['index'] in indices]
        metrics[role]=forecast_metrics([r['actual_mbps'] for r in g],[r['forecast_mbps'] for r in g])
        metrics[role+'_persistence_reference']=forecast_metrics(loads[list(indices)],loads[np.asarray(list(indices))-1])
    dump(OUT/'forecast_accuracy.json',metrics)
    print('Forecast pipeline completed',selected,metrics,flush=True)

if __name__=='__main__':main()
