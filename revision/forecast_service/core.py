"""Versioned causal providers. No controller imports, fitting on validation or test."""
from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path
from itertools import islice
import csv,hashlib,json,math,time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from revision.evaluation.twin import ROOT,sha256,_runtime
from revision.evaluation.contract import Forecast
from revision.evaluation.data import FitProvenance
from revision.stage3.common import dump,forecast_metrics
from revision.evaluation.validate_static import write_csv

BASE=ROOT/'revision/forecast_service';OUT=ROOT/'revision/reports/forecast_service';MODELS=ROOT/'revision/models/forecast_service'
CFG=json.loads((BASE/'protocol.json').read_text())


def raw_prefix(name,stop):
    if name not in CFG['workloads'] or not 0<stop<=2520:raise ValueError('Unauthorized source prefix')
    rows=[]
    with (ROOT/CFG['datasets'][name]).open() as f:
        for r in islice(csv.DictReader(f),stop):rows.append((datetime.fromisoformat(r['timestamp']),float(r['Traffic_bps'])/1e6))
    if len(rows)!=stop:raise ValueError('Insufficient authorized source rows')
    times=tuple(r[0] for r in rows);loads=np.array([r[1] for r in rows],float)
    if any(b-a!=timedelta(minutes=15) for a,b in zip(times,times[1:])) or not np.isfinite(loads).all() or (loads<0).any():raise ValueError('Invalid source timeline')
    return times,loads


def data(name):
    _,reference=raw_prefix('netflix',2016);multiplier=400./float(reference.max())
    times,loads=raw_prefix(name,2520)
    return times,loads*multiplier,multiplier


def data_hash(loads,times,stop):
    return hashlib.sha256(json.dumps([[t.isoformat() for t in times[:stop]],np.asarray(loads[:stop]).tolist()],separators=(',',':')).encode()).hexdigest()


def calendar(t):
    h=(t.hour*60+t.minute)/1440
    return np.array([np.sin(2*np.pi*h),np.cos(2*np.pi*h),np.sin(2*np.pi*t.weekday()/7),np.cos(2*np.pi*t.weekday()/7),float(t.weekday()>4),float(8<=t.hour<18)],np.float32)


def windows(loads,times,indices,scale):
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid training scaler')
    indices=list(indices)
    if any(type(i) not in (int,np.int64,np.int32) or i<96 or i>=len(times) for i in indices):raise ValueError('Invalid forecast target')
    x=np.asarray([loads[i-96:i] for i in indices],np.float32)[...,None]/scale
    c=np.asarray([calendar(times[i]) for i in indices],np.float32)
    return np.concatenate([x,np.repeat(c[:,None,:],96,axis=1)],axis=2)


def tabular(x):
    h=x[:,:,0]
    return np.c_[h,x[:,0,1:],h[:,-4:].mean(1),h[:,-16:].mean(1),h.mean(1),h[:,-1]-h[:,-2]]


def neural(family):
    _,tf,_,_=_runtime();layers=tf.keras.layers
    if family=='lstm':
        return tf.keras.Sequential([layers.Input((96,7)),layers.SpatialDropout1D(.1),
          layers.LSTM(128,return_sequences=True,dropout=.2,recurrent_dropout=0.),layers.LSTM(64,dropout=.2,recurrent_dropout=0.),
          layers.Dropout(.2),layers.Dense(64,activation='relu'),layers.Dropout(.2),layers.Dense(1,activation='softplus')])
    if family=='gru':return tf.keras.Sequential([layers.Input((96,7)),layers.GRU(32),layers.Dense(16,activation='relu'),layers.Dense(1,activation='softplus')])
    raise ValueError('Not a neural family')


def neural_stage(family,loads,times,fit_stop,selection_stop,epochs,select):
    _,tf,_,_=_runtime();cfg=CFG['neural'];tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(cfg['seed'])
    scale=float(max(loads[:fit_stop]));ix=np.arange(96,fit_stop);x=windows(loads,times,ix,scale);y=np.asarray(loads)[ix,None].astype(np.float32)/scale
    if select:
        vi=np.arange(fit_stop,selection_stop);vx=windows(loads,times,vi,scale);vy=np.asarray(loads)[vi,None]/scale
    net=neural(family);opt=tf.keras.optimizers.Adam(cfg['learning_rate'])
    @tf.function(input_signature=[tf.TensorSpec([None,96,7],tf.float32),tf.TensorSpec([None,1],tf.float32)])
    def step(a,b):
        with tf.GradientTape() as tape:loss=tf.reduce_mean((net(a,training=True)-b)**2)
        opt.apply_gradients(zip(tape.gradient(loss,net.trainable_weights),net.trainable_weights));return loss
    rng=np.random.default_rng(cfg['seed']);best=float('inf');best_epoch=0;stale=0;records=[];weights=None
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(x));losses=[]
        for start in range(0,len(x),cfg['batch_size']):
            ii=order[start:start+cfg['batch_size']];losses.append(float(step(x[ii],y[ii])))
        row={'epoch':epoch,'training_mse_scaled':float(np.mean(losses))}
        if select:
            vp=np.asarray(net(vx,training=False));loss=float(np.mean((vp-vy)**2));row['inner_stopping_mse_scaled']=loss
            if not np.isfinite(loss):raise ValueError('Invalid stopping loss')
            if loss<best-cfg['min_delta_scaled_mse']:best=loss;best_epoch=epoch;stale=0;weights=net.get_weights()
            else:stale+=1
        if not np.isfinite(losses).all():raise ValueError('Invalid training loss')
        records.append(row)
        if select and stale>=cfg['patience']:break
    if select:net.set_weights(weights)
    return net,scale,records,best_epoch if select else epochs


def fit(family,workload,loads,times,stop,directory):
    if family not in ('lstm','gru','boosted_poisson') or not 672<=stop<=2016:raise ValueError('Fitting restricted to adequate training prefix')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    cfg=CFG['neural'];inner=stop-cfg['inner_holdout_slots'];history=[]
    if family in ('lstm','gru'):
        model,_,records,best_epoch=neural_stage(family,loads,times,inner,stop,cfg['max_epochs'],True)
        history += [{'phase':'inner_stopping',**r} for r in records]
        del model
        model,scale,records,_=neural_stage(family,loads,times,stop,stop,best_epoch,False)
        history += [{'phase':'past_only_refit',**r} for r in records]
        file='model.keras';model.save(directory/file)
    else:
        scale=float(max(loads[:stop]));indices=np.arange(96,stop)
        kwargs={k:v for k,v in CFG['boosted'].items() if k!='features'}
        model=HistGradientBoostingRegressor(**kwargs).fit(tabular(windows(loads,times,indices,scale)),np.asarray(loads)[indices])
        file='model.joblib';joblib.dump(model,directory/file);best_epoch=None
    if history:write_csv(directory/'history.csv',[{k:r.get(k) for k in ['phase','epoch','training_mse_scaled','inner_stopping_mse_scaled']} for r in history])
    _,_,_,runtime=_runtime()
    manifest={'version':CFG['id'],'workload':workload,'family':family,'fit_stop_exclusive':stop,'parameter_targets_half_open':[96,stop],
      'inner_stopping_targets_half_open':[inner,stop] if family!='boosted_poisson' else None,
      'best_epoch':best_epoch,'full_refit_after_epoch_selection':family!='boosted_poisson','scale':scale,'scaler_fit_stop_exclusive':stop,
      'training_prefix_sha256':data_hash(loads,times,stop),'last_training_timestamp':times[stop-1].isoformat(),
      'model_file':file,'model_sha256':sha256(directory/file),'training_seconds':time.perf_counter()-start,
      'seed':42,'history_slots':96,'selection_stop_exclusive':stop,'selection_cutoff_timestamp':times[stop].isoformat(),'protocol_sha256':sha256(BASE/'protocol.json'),'implementation_sha256':sha256(Path(__file__)),
      'uses_validation_or_final_test_for_fit':False,'runtime':runtime,
      'versions':{'numpy':np.__version__,'sklearn':__import__('sklearn').__version__,'joblib':joblib.__version__},
      'nonnegative_output':'softplus in scaled Mbps' if family!='boosted_poisson' else 'Poisson loss with exponential inverse link'}
    dump(directory/'manifest.json',manifest)
    print(workload,family,'prefix',stop,'epoch',best_epoch,'seconds',round(manifest['training_seconds'],2),flush=True)
    return Provider(directory)


class Provider:
    """Fit artifact or frozen parameter-free provider. Common Forecast interface."""
    def __init__(self,directory):
        self.directory=Path(directory);self.meta=json.loads((self.directory/'manifest.json').read_text())
        self.identity=sha256(self.directory/'manifest.json');m=self.meta;self.family=m['family']
        if m['protocol_sha256']!=sha256(BASE/'protocol.json') or m['implementation_sha256']!=sha256(Path(__file__)):raise ValueError('Provider protocol/implementation mismatch')
        self.model=None
        if m.get('model_file'):
            path=self.directory/m['model_file']
            if sha256(path)!=m['model_sha256']:raise ValueError('Forecast artifact hash mismatch')
            if self.family in ('lstm','gru'):
                _,tf,_,_=_runtime();self.model=tf.keras.models.load_model(path,compile=False)
            else:self.model=joblib.load(path)
        self.fit=FitProvenance(m['training_prefix_sha256'],datetime.fromisoformat(m['last_training_timestamp'])) if m['fit_stop_exclusive'] else None
    def batch(self,loads,times,indices):
        indices=list(indices);m=self.meta
        if not indices or any(i<96 or i>=len(times) or max(m['fit_stop_exclusive'],m.get('selection_stop_exclusive',0))>i for i in indices):raise ValueError('Model fit is after decision or invalid target')
        return self._values(loads,times,indices)
    def diagnostic_in_sample(self,loads,times,indices):
        indices=list(indices)
        if any(i<96 or i>=2016 for i in indices):raise ValueError('In-sample diagnostics restricted to training labels')
        return self._values(loads,times,indices)
    def _values(self,loads,times,indices):
        m=self.meta
        if self.family=='persistence':p=np.asarray(loads)[np.asarray(indices)-1]
        elif self.family=='seasonal_daily':p=np.asarray(loads)[np.asarray(indices)-96]
        else:
            x=windows(loads,times,indices,m['scale'])
            if self.family=='boosted_poisson':p=self.model.predict(tabular(x))
            else:p=np.concatenate([np.asarray(self.model(x[i:i+128],training=False)).ravel() for i in range(0,len(x),128)])*m['scale']
        p=np.asarray(p,float)
        if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid forecast; no clipping/repair')
        return p
    def predict(self,context):
        if len(context.forecast_input_mbps)!=96 or context.decision_at-context.completed_interval_at!=timedelta(minutes=15):raise ValueError('Forecast needs96 completed contiguous intervals')
        if self.fit and self.fit.last_training_label>context.completed_interval_at:raise ValueError('Future-fit model')
        if context.decision_at<datetime.fromisoformat(self.meta['selection_cutoff_timestamp']):raise ValueError('Provider selection from future')
        history=np.asarray(context.forecast_input_mbps,float)
        if not np.isfinite(history).all() or (history<0).any():raise ValueError('Invalid completed history')
        ts=[context.decision_at-timedelta(minutes=15*(96-i)) for i in range(97)]
        # Local index96 refers to this window; batch checks use the actual global fit index separately above.
        if self.family=='persistence':v=float(history[-1])
        elif self.family=='seasonal_daily':v=float(history[0])
        else:
            x=windows(np.r_[history,0.],ts,[96],self.meta['scale'])
            v=float(self.model.predict(tabular(x))[0]) if self.family=='boosted_poisson' else float(np.asarray(self.model(x,training=False)).ravel()[0]*self.meta['scale'])
        if not math.isfinite(v) or v<0:raise ValueError('Invalid service output')
        return Forecast(context.decision_at,context.completed_interval_at,context.decision_at,v)


def simple(family,workload,directory,selection_stop=0,times=None):
    if family not in ('persistence','seasonal_daily'):raise ValueError('Not parameter-free')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    dump(directory/'manifest.json',{'family':family,'workload':workload,'version':CFG['id'],'fit_stop_exclusive':0,
       'selection_stop_exclusive':selection_stop,'selection_cutoff_timestamp':times[selection_stop].isoformat() if selection_stop else '2019-04-01T00:00:00',
       'history_slots':96,'protocol_sha256':sha256(BASE/'protocol.json'),'implementation_sha256':sha256(Path(__file__))})
    return Provider(directory)


def select(scores):
    ranking=['persistence','seasonal_daily','boosted_poisson','gru','lstm']
    return min(scores,key=lambda n:(scores[n]['rmse_mbps'],ranking.index(n)))


def load_service(workload):
    if workload not in CFG['workloads']:raise ValueError('Unknown service workload')
    record=json.loads((OUT/workload/'service.json').read_text())
    manifest=ROOT/record['provider_manifest'];selection=ROOT/record['selection_manifest']
    if sha256(manifest)!=record['provider_manifest_sha256'] or sha256(selection)!=record['selection_sha256']:raise ValueError('Frozen service identity mismatch')
    chosen=json.loads(selection.read_text())
    provider=Provider(manifest.parent)
    if provider.family!=chosen['selected_family'] or provider.family!=record['selected_family'] or provider.meta['workload']!=workload:raise ValueError('Service selection mismatch')
    return provider
