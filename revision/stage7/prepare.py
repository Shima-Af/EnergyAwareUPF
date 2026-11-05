"""Inference-only cache and protocol input verification, no forecasting refit."""
from dataclasses import asdict
import json
import numpy as np
from revision.evaluation.revised_twin import RevisionTwin
from revision.evaluation.validate_static import write_csv
from revision.forecast_service.core import load_service,data_hash
from .common import ROOT,BASE,OUT,CFG,horizons,sha256,dump
from .pipeline import Ledger,feature_names


def main():
    twin=RevisionTwin();loads=set();inputs={};forecast_provenance={}
    for name in ['netflix','youtube']:
        train,val,contract=horizons(name);ledger=Ledger(ROOT/CFG['forecast_ledgers'][name])
        service=load_service(name);scope=[val] if name=='youtube' else [train,val]
        for horizon in scope:
            for i in horizon.indices:
                c=horizon.context(i);ledger.at(i,c);loads.add(float(horizon.partition.timeline.load_mbps[i]))
        loads.update(float(ledger.rows[i]['forecast_mbps']) for i in val.indices)
        manifest=json.loads((ROOT/f'revision/reports/forecast_service/{name}/data_manifest.json').read_text())
        timeline=val.partition.timeline
        if data_hash(np.array(timeline.load_mbps),timeline.timestamps,2520)!=manifest['authorized_prefix_sha256']:raise ValueError('Approved input prefix changed')
        inputs[name]={'train_horizon':train.identity if train else None,'validation_horizon':val.identity,
             'train_slots':len(train.indices) if train else 0,'validation_slots':len(val.indices),
             'validation_timestamps':[t.isoformat() for t in val.timestamps],
             'authorized_prefix_sha256':manifest['authorized_prefix_sha256'],'forecasts_sha256':ledger.digest,
             'service_sha256':sha256(ROOT/CFG['forecast_services'][name]),'provider_id':service.identity,'contract':asdict(contract)}
        forecast_provenance[name]={'selection':json.loads((ROOT/f'revision/reports/forecast_service/{name}/selection_frozen.json').read_text()),
            'training_provider_counts':{family:sum(r['family']==family for i,r in ledger.rows.items() if i<2016) for family in ['persistence','boosted_poisson']},
            'cold_start_rows':sum(r['kind']=='minimum_history_cold_start' for i,r in ledger.rows.items() if i<2016)}
    rows=[]
    for j,load in enumerate(sorted(loads)):
        for action in range(3):
            o=twin.outcome(action,load);rows.append({'load_mbps':load,'action':action,'power_w':o.power_w,'qos':o.qos})
        if not j%1000:print('Verified real frozen-twin inputs',j,'/',len(loads),flush=True)
    write_csv(OUT/'outcome_cache.csv',rows);twin.verify_artifacts()
    dump(OUT/'prepared.json',{'twin_id':twin.identity,'twin_manifest':twin.manifest,'cache_sha256':sha256(OUT/'outcome_cache.csv'),
      'unique_input_loads':len(loads),'outcomes':len(rows),'inputs':inputs,'feature_order':{s:feature_names(s) for s in CFG['schemas']},
      'forecast_provenance':forecast_provenance,'cache_kind':'Exact frozen-twin scalar inference; read-only numerical optimization, no fitted substitute or physical-output repair',
      'final_test_inputs':False,'forecast_refits':0})

if __name__=='__main__':main()
