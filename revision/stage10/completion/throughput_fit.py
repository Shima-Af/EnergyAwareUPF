"""Authenticate active per-direction offer and fit the approved DN-receive KPI.

Adds the previously gated third KPI after flow evidence, with the exact same
frozen family; does not repeat or reselect delay/jitter models.
"""
import io,json,zipfile
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from revision.evaluation.twin import ROOT,sha256
from .independent_fits import CFG,OUT,MODELS,dump,csvout,metrics
from .counter_audit import RAW


def main():
    dump(OUT/'throughput_provenance_amendment.json',dict(reason='Active flow and offered packet-population evidence now examined; not a response to model performance',
         family=CFG['kpi_family'],fits=2,search=False,definition='DN receive-side uplink throughput / per-direction configured offered Mbps; not delivered downlink or application goodput',
         row_policy='Use exact raw receive-counter matches only; no fill or zero substitution',
         implementation_sha256=sha256(__import__('pathlib').Path(__file__))))
    checks=[];scores=[]
    rawchecks=pd.read_csv(ROOT/'revision/reports/task2g/raw_row_checks.csv')
    for mode,prefix in [('dpdk','dpdk'),('usr','oai')]:
        source=__import__('pathlib').Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')/f'{prefix}_processed.csv'
        frame=pd.read_csv(source);frame['time']=pd.to_datetime(frame.timestamp,utc=True)
        for path in sorted((RAW/mode).glob('*.zip')):
            with zipfile.ZipFile(path) as outer:
                times=pd.to_datetime(pd.read_csv(io.BytesIO(outer.read('Fullcoreapplicationtraffic_GTPuPackets_NGRAN.csv'))).timestamp,utc=True)
                case=frame[frame.time.isin(times)]
                if not len(case):raise ValueError('No exact processed case match')
                objective=case['Objective Throughput (DL/UL)'].unique()
                if len(objective)!=1:raise ValueError('Ambiguous source offer')
                nested=next(n for n in outer.namelist() if n.endswith('.zip') and not n.startswith('agent'))
                with zipfile.ZipFile(io.BytesIO(outer.read(nested))) as inner:
                    cfg=json.loads(inner.read('config-data.bin'))['Config']
                    for side,expected in [('dn','downlink'),('ue','uplink')]:
                        active=[(r,o['statelessUDP']) for r in cfg['nodes'][side]['ranges'] for o in r['userPlane']['tigerObjective']
                                if r['enable'] and o['enable'] and o['type']=='StatelessUDP']
                        if len(active)!=1:raise ValueError('Ambiguous active flow')
                        r,u=active[0]
                        inferred_offer=u['rate']*u['payloadSize']*8/1e6
                        ok=(u['direction']==expected and int(r['count'])==1 and abs(inferred_offer-float(objective[0]))<1e-9)
                        checks.append(dict(mode=mode,archive=path.name,side=side,direction=u['direction'],
                             configured_rate=u['rate'],payload_bytes=u['payloadSize'],configured_offer_mbps=inferred_offer,
                             recorded_offer_mbps=float(objective[0]),matches=ok))
                        if not ok:raise ValueError('Configured per-direction offer not authenticated')
        raw=rawchecks[rawchecks['mode']==mode].reset_index(drop=True)
        assert len(raw)==len(frame) and raw.timestamp.tolist()==frame.timestamp.tolist()
        valid=raw.raw_dn_rx_kbps.notna().to_numpy() & (abs(raw.stored_dn_rx_mbps-raw.raw_dn_rx_kbps/1000)<1e-12).to_numpy()
        offered=frame['Objective Throughput (DL/UL)'].to_numpy();target=frame['DN_Rx_Mbits/s'].to_numpy()/offered
        train=frame['Test Set'].isin([1,2,3]).to_numpy()&valid
        args={k:v for k,v in CFG['kpi_family'].items() if k!='name'}
        model=RandomForestRegressor(**args);model.fit(np.log1p(offered[train,None]),target[train])
        folder=MODELS/'kpi'/mode/'throughput_fraction';folder.mkdir(parents=True,exist_ok=False)
        joblib.dump(model,folder/'model.joblib');prediction=model.predict(np.log1p(offered[:,None]))
        assert np.array_equal(prediction,joblib.load(folder/'model.joblib').predict(np.log1p(offered[:,None])))
        dump(folder/'manifest.json',dict(target='throughput_fraction',direction='uplink received at DN',
             definition='DN_Rx_Mbits/s / Objective Throughput (DL/UL)',not_application_goodput=True,
             source_sha256=sha256(source),train_source_rows=np.flatnonzero(train).tolist(),
             excluded_unmatched_raw_source_rows=np.flatnonzero(~valid).tolist(),
             input_transform='log1p(per-instance Mbps)',domain_mbps=[.01,5000],family=CFG['kpi_family'],
             model_sha256=sha256(folder/'model.joblib'),selection='none; same frozen model family, one seed',
             row_provenance_sha256=sha256(ROOT/'revision/reports/task2g/raw_row_checks.csv')))
        for role,sweeps in CFG['group_split'].items():
            mask=frame['Test Set'].isin(sweeps).to_numpy()&valid;y=target[mask];p=prediction[mask]
            m=metrics(y,p);safe=p>=.9;unsafe=y<.9
            m.update(mode=mode,role=role,predicted_safe=int(safe.sum()),false_safe=int((safe&unsafe).sum()),false_unsafe=int((~safe&~unsafe).sum()),
                     measured_violations=int(unsafe.sum()),max_false_safe_shortfall=float((.9-y[safe&unsafe]).max()) if (safe&unsafe).any() else None)
            scores.append(m)
    csvout(OUT/'active_offer_mapping.csv',checks);dump(OUT/'throughput_metrics.json',scores)


if __name__=='__main__':main()
