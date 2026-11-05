"""Single-family prespecified fits: only authenticated delay/jitter and forecasts.

No loss-label fabrication, proxy throughput fitting, model-family comparison,
controller replay, checkpoint selection, or traffic final-test target access.
"""
from pathlib import Path
import csv,json,platform,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from revision.evaluation.twin import ROOT,sha256
from revision.forecast_service.core import data,windows,tabular
from revision.stage9.core import segments

BASE=ROOT/'revision/stage10/completion';OUT=ROOT/'revision/reports/stage10/completion'
MODELS=ROOT/'revision/models/stage10';CFG=json.loads((BASE/'protocol.json').read_text())


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)


def csvout(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metrics(y,p,budget=None):
    err=p-y;variance=float(np.sum((y-y.mean())**2))
    out=dict(n=len(y),mae=float(abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))),
             bias=float(err.mean()),r2=float(1-np.sum(err**2)/variance) if variance else None)
    if budget is not None:
        safe=p<=budget;unsafe=y>budget
        out.update(predicted_safe=int(safe.sum()),false_safe=int((safe&unsafe).sum()),
           false_unsafe=int((~safe&~unsafe).sum()),measured_violations=int(unsafe.sum()),
           max_false_safe_exceedance=float((y[safe&unsafe]-budget).max()) if (safe&unsafe).any() else None)
    return out


def main():
    dump(OUT/'independent_protocol_frozen.json',dict(protocol=CFG,protocol_sha256=sha256(BASE/'protocol.json'),
          implementation_sha256=sha256(Path(__file__)),runtime={n:importlib.metadata.version(n) for n in ['numpy','scikit-learn','joblib']},
          python=platform.python_version(),controller_fits=0))
    summaries=[];predictions=[]
    for mode,prefix in [('dpdk','dpdk'),('usr','oai')]:
        source=Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')/f'{prefix}_processed.csv'
        frame=pd.read_csv(source);load=frame['Objective Throughput (DL/UL)'].to_numpy()
        train=frame['Test Set'].isin([1,2,3]).to_numpy()
        for name,spec in CFG['kpi_targets_now'].items():
            y=frame[spec['column']].to_numpy();assert np.isfinite(y).all() and (y>=0).all()
            args={k:v for k,v in CFG['kpi_family'].items() if k!='name'}
            model=RandomForestRegressor(**args);model.fit(np.log1p(load[train,None]),y[train])
            folder=MODELS/'kpi'/mode/name;folder.mkdir(parents=True,exist_ok=False)
            joblib.dump(model,folder/'model.joblib')
            restored=joblib.load(folder/'model.joblib');p=model.predict(np.log1p(load[:,None]))
            assert np.array_equal(p,restored.predict(np.log1p(load[:,None]))) and (p>=0).all()
            dump(folder/'manifest.json',dict(target=name,physical_column=spec['column'],budget=spec['budget'],
                 source=str(source),source_sha256=sha256(source),train_source_rows=np.flatnonzero(train).tolist(),
                 input_transform='log1p(per-instance Mbps); no learned scaler',domain_mbps=[.01,5000],
                 output_constraint='Nonnegative convex averages of nonnegative measured targets; no clipping',
                 family=CFG['kpi_family'],model_sha256=sha256(folder/'model.joblib'),
                 protocol_sha256=sha256(BASE/'protocol.json'),selection='none; one frozen model per mode/target',
                 scientific_scope='Recorded delay/jitter only; not a complete QoS provider'))
            for role,sweeps in CFG['group_split'].items():
                mask=frame['Test Set'].isin(sweeps).to_numpy()
                summaries.append(dict(mode=mode,target=name,role=role,**metrics(y[mask],p[mask],spec['budget'])))
            for i in range(len(frame)):
                predictions.append(dict(mode=mode,target=name,source_row=i,sweep=int(frame['Test Set'].iloc[i]),
                    load_mbps=float(load[i]),measured=float(y[i]),predicted=float(p[i])))
            print('fitted KPI',mode,name,flush=True)
    dump(OUT/'partial_kpi_metrics.json',summaries);csvout(OUT/'partial_kpi_predictions.csv',predictions)
    forecasts=[];forecast_metrics=[]
    times,nf,_=data('netflix');ytimes,yt,_=data('youtube')
    for scenario,spec in CFG['scenarios'].items():
        def transform(loads):
            ix=np.arange(len(loads))
            if spec['kind']=='scale':return loads*spec['factor']
            return loads*np.array([spec['factors'][(i//96)%3] for i in ix])
        x=transform(nf);yx=transform(yt)
        for cutoff in CFG['forecast_prefixes']:
            scale=float(x[:cutoff].max());ix=np.arange(96,cutoff)
            features=tabular(windows(x,times,ix,scale));target=x[ix]/scale
            args={k:v for k,v in CFG['forecast_family'].items() if k!='name'}
            model=HistGradientBoostingRegressor(**args);model.fit(features,target)
            folder=MODELS/'forecast'/scenario/str(cutoff);folder.mkdir(parents=True,exist_ok=False)
            joblib.dump(model,folder/'model.joblib');assert np.array_equal(model.predict(features[:10]),joblib.load(folder/'model.joblib').predict(features[:10]))
            dump(folder/'manifest.json',dict(training_workload='netflix',fit_stop_exclusive=cutoff,
               fit_source_rows=ix.tolist(),scale_training_prefix_max=scale,scenario=spec,
               source_prefix_sha256=__import__('hashlib').sha256(x[:cutoff].tobytes()).hexdigest(),
               input='96 completed loads + 6 known calendar + 4 history-derived features',family=CFG['forecast_family'],
               model_sha256=sha256(folder/'model.joblib'),protocol_sha256=sha256(BASE/'protocol.json')))
            end=min(cutoff+336,2016) if cutoff<2016 else 2520
            for workload,loads,clock in [('netflix',x,times)]+([('youtube',yx,ytimes)] if cutoff==2016 else []):
                targets=np.arange(cutoff,end)
                p=model.predict(tabular(windows(loads,clock,targets,scale)))*scale
                assert np.isfinite(p).all() and (p>=0).all()
                for index,pred in zip(targets,p):
                    forecasts.append(dict(workload=workload,scenario=scenario,role='train' if index<2016 else 'validation',
                       source_index=int(index),timestamp=clock[index].isoformat(),observed_through=clock[index-1].isoformat(),
                       fit_stop_exclusive=cutoff,family='boosted_poisson',actual_mbps=float(loads[index]),forecast_mbps=float(pred)))
        for index in range(96,672):
            forecasts.append(dict(workload='netflix',scenario=scenario,role='train',source_index=index,
                 timestamp=times[index].isoformat(),observed_through=times[index-1].isoformat(),
                 fit_stop_exclusive=0,family='persistence_cold_start',actual_mbps=float(x[index]),forecast_mbps=float(x[index-1])))
        print('fitted forecasting',scenario,'5 prefix models',flush=True)
    forecasts.sort(key=lambda r:(r['workload'],r['scenario'],r['source_index']))
    csvout(OUT/'rebuilt_forecasts.csv',forecasts)
    for w in ['netflix','youtube']:
        for scenario in CFG['scenarios']:
            for role in ['train','validation']:
                sub=[r for r in forecasts if (r['workload'],r['scenario'],r['role'])==(w,scenario,role)]
                if sub:forecast_metrics.append(dict(workload=w,scenario=scenario,role=role,**metrics(np.array([r['actual_mbps'] for r in sub]),np.array([r['forecast_mbps'] for r in sub]))))
    dump(OUT/'rebuilt_forecast_metrics.json',forecast_metrics)


if __name__=='__main__':main()
