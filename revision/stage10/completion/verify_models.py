"""Read-only partial-KPI validation, all rows/loads/sweeps retained."""
import json
import joblib
import numpy as np
import pandas as pd
from revision.evaluation.twin import ROOT,sha256
from .independent_fits import OUT,MODELS,metrics,csvout,dump


def summarize_thresholds():
    frame=pd.read_csv(OUT/'all_partial_kpi_predictions.csv');details=[]
    for (mode,target),g in frame.groupby(['mode','target']):
        for role,sweeps in [('train',[1,2,3]),('validation',[4]),('internal_test',[5])]:
            s=g[g.sweep.isin(sweeps)];y=s.measured.to_numpy();p=s.predicted.to_numpy();b=float(s.budget.iloc[0]);lower=target=='throughput_fraction'
            unsafe=y<b if lower else y>b;safe=p>=b if lower else p<=b;fs=unsafe&safe
            severity=np.maximum(0,1-y/b if lower else y/b-1)
            band=abs(y-b)<=.1*b
            details.append(dict(mode=mode,target=target,role=role,**metrics(y,p),predicted_safe=int(safe.sum()),false_safe=int(fs.sum()),
                 false_unsafe=int((~safe&~unsafe).sum()),measured_violations=int(unsafe.sum()),
                 false_safe_given_predicted_safe=float(fs.sum()/safe.sum()) if safe.sum() else None,
                 false_safe_severity_mean=float(severity[fs].mean()) if fs.any() else None,
                 false_safe_severity_max=float(severity[fs].max()) if fs.any() else None,
                 measured_centered_10percent_budget_band_n=int(band.sum()),
                 band_rmse=float(np.sqrt(np.mean((p[band]-y[band])**2))) if band.any() else None,
                 band_false_safe=int((fs&band).sum())))
    dump(OUT/'partial_kpi_threshold_details.json',details)


def main():
    rows=[];groups=[];grids=[]
    raw=pd.read_csv(ROOT/'revision/reports/task2g/raw_row_checks.csv')
    for mode,prefix in [('dpdk','dpdk'),('usr','oai')]:
        source=__import__('pathlib').Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')/f'{prefix}_processed.csv'
        frame=pd.read_csv(source);load=frame['Objective Throughput (DL/UL)'].to_numpy()
        for target,column,budget,lower in [('delay_ms','downlink_delay_ms',4.,False),('jitter_us','RAN_jitter',1.,False),('throughput_fraction',None,.9,True)]:
            folder=MODELS/'kpi'/mode/target;m=json.loads((folder/'manifest.json').read_text())
            assert sha256(folder/'model.joblib')==m['model_sha256'] and sha256(source)==m['source_sha256']
            model=joblib.load(folder/'model.joblib');prediction=model.predict(np.log1p(load[:,None]))
            y=frame[column].to_numpy() if column else frame['DN_Rx_Mbits/s'].to_numpy()/load
            valid=np.ones(len(y),bool)
            if lower:
                rr=raw[raw['mode']==mode].reset_index(drop=True)
                valid=rr.raw_dn_rx_kbps.notna().to_numpy()&(abs(rr.stored_dn_rx_mbps-rr.raw_dn_rx_kbps/1000)<1e-12).to_numpy()
            for i in np.flatnonzero(valid):
                rows.append(dict(mode=mode,target=target,source_row=int(i),sweep=int(frame['Test Set'].iloc[i]),
                    load_mbps=float(load[i]),measured=float(y[i]),predicted=float(prediction[i]),budget=budget,
                    measured_violation=bool(y[i]<budget if lower else y[i]>budget),
                    predicted_violation=bool(prediction[i]<budget if lower else prediction[i]>budget)))
            for group in sorted(frame['Test Set'].unique()):
                for value in sorted(set(load)):
                    mask=(frame['Test Set'].to_numpy()==group)&(load==value)&valid
                    if not mask.any():continue
                    p=prediction[mask];yy=y[mask];safe=(p>=budget if lower else p<=budget);unsafe=(yy<budget if lower else yy>budget)
                    groups.append(dict(mode=mode,target=target,sweep=int(group),load_mbps=float(value),
                        **metrics(yy,p),predicted_safe=int(safe.sum()),false_safe=int((safe&unsafe).sum()),
                        false_unsafe=int((~safe&~unsafe).sum()),measured_violations=int(unsafe.sum())))
            x=np.geomspace(.01,5000,3000);p=model.predict(np.log1p(x[:,None]))
            assert np.isfinite(p).all() and (p>=0).all()
            grids.append(dict(mode=mode,target=target,n=len(x),min=float(p.min()),max=float(p.max()),finite_nonnegative=True))
    csvout(OUT/'all_partial_kpi_predictions.csv',rows);csvout(OUT/'kpi_errors_by_load_sweep.csv',groups)
    dump(OUT/'partial_kpi_domain_checks.json',grids)
    forecasts=pd.read_csv(OUT/'rebuilt_forecasts.csv');margins={}
    for name,g in forecasts[(forecasts.workload=='netflix')&(forecasts.source_index>=1680)&(forecasts.source_index<2016)].groupby('scenario'):
        assert len(g)==336 and (g.fit_stop_exclusive==1680).all()
        margins[name]=dict(k=1.,margin_mbps=float(abs(g.actual_mbps-g.forecast_mbps).mean()),
               source='Netlix training calibration rows1680..2015; provider fitted before1680',
               manual_margin_unchanged_mbps=26.,controller_outcomes_used=False)
    dump(OUT/'baseline_training_margin.json',margins)
    summarize_thresholds()


if __name__=='__main__':main()
