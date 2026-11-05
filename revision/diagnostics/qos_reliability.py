"""Read-only QoS reliability and label-provenance audit; no fitting or traffic."""

import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from revision.evaluation.twin import ROOT, FittedUPFTwin, sha256
from revision.evaluation.validate_static import write_csv
from revision.profiling.refit_usr_power import write_json

LOAD, TARGET = 'Objective Throughput (DL/UL)', 'QoS_w'
DATA = Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')
SCRIPT = Path('/home/ubuntu/xRL_UpfSelection/scripts/PowerPerformance/train_performance_models.py')
NOTEBOOK = Path('/home/ubuntu/xRL_UpfSelection/notebooks/03_QoS.ipynb')


def threshold_statistics(measured, predicted, tau=.90):
    y, p = np.asarray(measured, dtype=float), np.asarray(predicted, dtype=float)
    if len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Threshold audit requires matched finite labels and predictions')
    unsafe, safe = y < tau, p >= tau
    fs, fu = unsafe & safe, ~unsafe & ~safe
    def ratio(a, b): return float(a / b) if b else None
    def distribution(v):
        return {'mean': float(v.mean()), 'p95': float(np.quantile(v, .95)), 'max': float(v.max())} if len(v) else None
    return {'n': len(y), 'measured_unsafe': int(unsafe.sum()), 'predicted_safe': int(safe.sum()),
            'true_safe': int((~unsafe & safe).sum()), 'true_unsafe': int((unsafe & ~safe).sum()),
            'false_safe': int(fs.sum()), 'false_unsafe': int(fu.sum()),
            'false_safe_fraction_all': ratio(fs.sum(), len(y)),
            'false_safe_fraction_predicted_safe': ratio(fs.sum(), safe.sum()),
            'false_safe_fraction_measured_unsafe': ratio(fs.sum(), unsafe.sum()),
            'false_unsafe_fraction_all': ratio(fu.sum(), len(y)),
            'false_unsafe_fraction_measured_safe': ratio(fu.sum(), (~unsafe).sum()),
            'false_safe_measured_shortfall': distribution(tau - y[fs]),
            'false_safe_prediction_optimism': distribution(p[fs] - y[fs])}


def audit_metrics(frame):
    y, p = frame[TARGET].to_numpy(), frame.predicted_qos.to_numpy()
    if not len(y): return {'n': 0}
    e, variance = p - y, float(np.sum((y - y.mean()) ** 2))
    return {'n': len(y), 'mae': float(np.mean(abs(e))), 'rmse': float(np.sqrt(np.mean(e**2))),
            'r2': 1 - float(np.sum(e**2)) / variance if variance else None,
            'bias_prediction_minus_measurement': float(e.mean()),
            'target_mean': float(y.mean()), 'target_std_population': float(y.std()),
            'target_min': float(y.min()), 'target_max': float(y.max()),
            'target_q05': float(np.quantile(y,.05)), 'target_q50': float(np.quantile(y,.5)),
            'target_q95': float(np.quantile(y,.95)),
            'residual_quantiles': {str(q): float(np.quantile(e,q)) for q in (0,.01,.05,.25,.5,.75,.95,.99,1)},
            'nonfinite_predictions': int((~np.isfinite(p)).sum()),
            'out_of_range_predictions': int(((p<0)|(p>1)).sum()),
            **threshold_statistics(y,p)}


def variance_decomposition(frame):
    """Descriptive in-sample conditional means, not a fitted evaluation model."""
    y, p = frame[TARGET], frame.predicted_qos
    load_mean = frame.groupby(LOAD)[TARGET].transform('mean')
    block_mean = frame.groupby(['Test Set',LOAD])[TARGET].transform('mean')
    sst = float(((y-y.mean())**2).sum())
    within = float(((y-load_mean)**2).sum())
    within_block = float(((y-block_mean)**2).sum())
    across_sweeps = float(((block_mean-load_mean)**2).sum())
    load_fit = float(((p-load_mean)**2).sum())
    sse = float(((p-y)**2).sum())
    if not np.isclose(sse, within+load_fit, atol=1e-10, rtol=1e-12):
        raise ValueError('Load-only residual decomposition failed')
    return {'n':len(y),'total_target_sst':sst,'model_sse':sse,
            'within_load_sse':within,'load_mean_misfit_sse':load_fit,
            'within_load_case_sse':within_block,'between_sweeps_given_load_sse':across_sweeps,
            'within_load_fraction_target_variation':within/sst,
            'within_load_fraction_model_sse':within/sse,
            'squared_global_bias_fraction_model_sse':len(y)*float((p-y).mean())**2/sse,
            'in_sample_load_mean_r2_ceiling':1-within/sst,
            'within_load_case_fraction_within_load_sse':within_block/within,
            'interpretation':'In-sample same-load variability is not identified as measurement noise; no independent ceiling claim'}


def candidate_label_formulas(frame):
    lat, jit, loss = frame.downlink_delay_ms.to_numpy(), frame.RAN_jitter.to_numpy(), frame.RAN_PLR_interval.to_numpy()
    def score(x, cap): return 1-np.clip(x,0,cap)/cap
    return {
       'exploratory_notebook_0.6_0.2_0.2_D300_J30000raw_L1': .6*score(lat,300)+.2*score(jit,30000)+.2*score(loss,1),
       'manuscript_0.4_0.4_0.2_D300ms_J100ms_L1e-6_assuming_raw_jitter_us':
           .4*score(lat,300)+.4*score(jit/1000,100)+.2*score(loss,1e-6)}


def historical_reconstruction(twin, mode, frame, train_idx, val_idx):
    train, val = frame.iloc[train_idx], frame.iloc[val_idx]
    sx, sy = (twin._components[mode,'qos',k] for k in ('input_scaler','target_scaler'))
    tx, ty = np.log1p(train[LOAD].to_numpy()), train[TARGET].to_numpy()
    stats = {'stored_input_mean':float(sx.mean_[0]),'reconstructed_input_mean':float(tx.mean()),
             'stored_input_scale':float(sx.scale_[0]),'reconstructed_input_scale':float(tx.std()),
             'stored_target_mean':float(sy.mean_[0]),'reconstructed_target_mean':float(ty.mean()),
             'stored_target_scale':float(sy.scale_[0]),'reconstructed_target_scale':float(ty.std()),
             'stored_training_rows':int(sy.n_samples_seen_),'reconstructed_training_rows':len(train)}
    stats['matches'] = all(np.isclose(stats['stored_'+key], stats['reconstructed_'+key], rtol=0,atol=1e-12)
                           for key in ('input_mean','input_scale','target_mean','target_scale','training_rows'))
    if not stats['matches']: raise ValueError('Historical QoS scaler reconstruction mismatch')
    train_set = set(int(i) for i in train_idx)
    adjacent = sum(any(j in train_set and frame.iloc[j]['Test Case']==frame.iloc[i]['Test Case']
                       for j in (int(i)-1,int(i)+1)) for i in val_idx)
    nearest = []
    for case, block in frame.groupby('Test Case'):
        a = pd.to_datetime(block.loc[block.index.isin(train_idx),'timestamp']).astype('int64').to_numpy()
        b = pd.to_datetime(block.loc[block.index.isin(val_idx),'timestamp']).astype('int64').to_numpy()
        nearest.extend(float(np.min(abs(a-v)))/1e9 for v in b)
    return {'train_rows':len(train),'early_stopping_validation_rows':len(val),
       'test_size':.2,'seed':42,'shuffle':True,'scaler_statistics':stats,
       'case_blocks_shared':len(set(train['Test Case'])&set(val['Test Case'])),
       'sweeps_shared':len(set(train['Test Set'])&set(val['Test Set'])),
       'validation_rows_with_adjacent_training_row_same_case':int(adjacent),
       'median_nearest_same_case_training_seconds':float(np.median(nearest)),
       'independent_test':False,'early_stopping_and_reported_scores_same_partition':True}


def run(output):
    twin = FittedUPFTwin()
    summaries, rows, load_rows, sweep_rows, near_rows, labels, identities, architectures = {},[],[],[],[],[],[],{}
    catalog_path=Path('/home/ubuntu/UpfProfilingCampaign/UpfProfilingCampaign/configs/test_cases.csv')
    catalog=pd.read_csv(catalog_path).set_index('Test Case')
    for mode,prefix,action in [('dpdk','dpdk',0),('usr','oai',1)]:
        path=DATA/f'{prefix}_processed.csv';frame=pd.read_csv(path)
        if not np.isfinite(frame[[LOAD,TARGET]].to_numpy()).all(): raise ValueError('Nonfinite QoS source')
        frame['source_row']=np.arange(len(frame))
        predictions={float(x):twin.raw_prediction(action,float(x)).qos for x in frame[LOAD].unique()}
        frame['predicted_qos']=[predictions[float(x)] for x in frame[LOAD]]
        train_idx,val_idx=train_test_split(np.arange(len(frame)),test_size=.2,random_state=42,shuffle=True)
        frame['historical_role']='train';frame.loc[val_idx,'historical_role']='early_stopping_validation'
        frame['false_safe']=(frame.predicted_qos>=.9)&(frame[TARGET]<.9)
        frame['false_unsafe']=(frame.predicted_qos<.9)&(frame[TARGET]>=.9)
        identities.append({'role':'measurement_csv','mode':mode,'path':str(path),'sha256':sha256(path)})
        for project in ('xRL_UpfSelection','dynamic-upf-selection-rl'):
            copy=Path('/home/ubuntu')/project/'data/processed'/f'{prefix}_processed.csv'
            if sha256(copy)!=sha256(path): raise ValueError('Measurement copies differ')
            identities.append({'role':'identical_measurement_copy','mode':mode,'path':str(copy),'sha256':sha256(copy)})
        for case,block in frame.groupby('Test Case'):
            if block['Test Set'].nunique()!=1: raise ValueError('Case crosses sweeps')
            for column in ('Test Set',LOAD,'Packet Size','Packet Rate'):
                if not (block[column]==catalog.loc[case,column]).all():raise ValueError('Catalog mismatch')
        model_path=ROOT/f'saved_models/prediction_models/{prefix}_upf_performance_model.keras'
        with zipfile.ZipFile(model_path) as archive:
            architectures[mode]={'path':str(model_path),'sha256':sha256(model_path),
              'metadata':json.loads(archive.read('metadata.json')),'config':json.loads(archive.read('config.json'))}
        scopes={'all_rows_descriptive':frame,'historical_training':frame.iloc[train_idx],
                'historical_early_stopping_validation':frame.iloc[val_idx],
                'all_rows_load_le400_descriptive':frame[frame[LOAD]<=400],
                'historical_validation_load_le400':frame.iloc[val_idx][frame.iloc[val_idx][LOAD]<=400]}
        group_summary={name:audit_metrics(f) for name,f in scopes.items()}
        for name,f in scopes.items():
            for anchor,values in [('measured',f[TARGET]),('predicted',f.predicted_qos)]:
                near=f[abs(values-.9)<=.05]
                near_rows.append({'mode':mode,'scope':name,'band_anchor':anchor,'band_half_width':.05,
                                  **audit_metrics(near)})
        for load,f in frame.groupby(LOAD):
            for name,g in [('all_rows_descriptive',f),('historical_early_stopping_validation',f[f.historical_role=='early_stopping_validation'])]:
                load_rows.append({'mode':mode,'scope':name,'offered_load_mbps':float(load),**audit_metrics(g)})
        for sweep,f in frame.groupby('Test Set'):
            for domain,g in [('full',f),('le400',f[f[LOAD]<=400])]:
                sweep_rows.append({'mode':mode,'sweep':int(sweep),'load_domain':domain,
                  'first_timestamp':f.timestamp.min(),'last_timestamp':f.timestamp.max(),
                  **audit_metrics(g)})
        formula_results={}
        for name,p in candidate_label_formulas(frame).items():
            error=p-frame[TARGET].to_numpy()
            formula_results[name]={'mae':float(np.mean(abs(error))),'max_abs_error':float(np.max(abs(error))),
                'rows_matching_to_1e-10':int(np.sum(abs(error)<=1e-10)),
                'formula_source_not_authenticated_original_builder':True}
            for i,v in enumerate(p):labels.append({'mode':mode,'source_row':i,'formula':name,
               'stored_qos_w':float(frame.iloc[i][TARGET]),'candidate_score':float(v),'difference':float(error[i])})
        constant=float(frame.iloc[train_idx][TARGET].mean());v=frame.iloc[val_idx]
        baseline=v.copy();baseline['predicted_qos']=constant
        summaries[mode]={'source_path':str(path),'source_sha256':sha256(path),'rows':len(frame),
            'sweeps':int(frame['Test Set'].nunique()),'case_blocks':int(frame['Test Case'].nunique()),
            'packet_sizes':sorted(int(x) for x in frame['Packet Size'].unique()),
            'date_range':[frame.timestamp.min(),frame.timestamp.max()],
            'metrics':group_summary,'historical_reconstruction':historical_reconstruction(twin,mode,frame,train_idx,val_idx),
            'variance_all_rows':variance_decomposition(frame),'variance_historical_validation':variance_decomposition(v),
            'training_mean_constant_historical_validation':audit_metrics(baseline),
            'label_formula_checks':formula_results,
            'independent_profiling_evidence_available':False}
        for _,r in frame.iterrows():
            rows.append({'mode':mode,'source_row':int(r.source_row),'timestamp':r.timestamp,
               'sweep':int(r['Test Set']),'test_case':r['Test Case'],'offered_load_mbps':float(r[LOAD]),
               'historical_role':r.historical_role,'measured_qos':float(r[TARGET]),
               'predicted_qos':float(r.predicted_qos),'prediction_minus_measurement':float(r.predicted_qos-r[TARGET]),
               'false_safe':bool(r.false_safe),'false_unsafe':bool(r.false_unsafe)})
        print('Audited',mode,'historical val RMSE',group_summary['historical_early_stopping_validation']['rmse'],flush=True)
    for path,role in [(SCRIPT,'candidate_historical_training_script'),(NOTEBOOK,'exploratory_label_notebook'),
                      (SCRIPT.parents[2]/'config.yaml','candidate_historical_configuration'),
                      (catalog_path,'measurement_catalog'),(ROOT/'Paper artifacts/manuscript_energyawareupf.tex','submitted_methodology')]:
        identities.append({'role':role,'path':str(path),'sha256':sha256(path)})
    write_json(output/'summary.json',summaries)
    write_json(output/'artifact_provenance.json',{'inputs':identities,'original_twin':twin.manifest,
               'qos_units':'dimensionless stored QoS_w, not verified service violation probability',
               'no_new_model_fitting':True,'traffic_data_read':False})
    write_json(output/'architectures.json',architectures)
    write_csv(output/'predictions.csv',rows)
    write_csv(output/'errors_by_load.csv',load_rows)
    write_csv(output/'errors_by_sweep.csv',sweep_rows)
    write_json(output/'near_threshold.json',near_rows)
    write_csv(output/'label_formula_checks.csv',labels)
    twin.verify_artifacts()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT/'revision'):raise ValueError('Diagnostics must stay under revision/')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    run(args.output_dir)


if __name__=='__main__':main()
