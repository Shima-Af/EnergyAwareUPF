"""Targeted active-flow and same-interval population feasibility, no new labels."""
from pathlib import Path
from datetime import datetime
import csv,io,json,zipfile
import numpy as np
import pandas as pd
from revision.evaluation.twin import ROOT,sha256
from .intervals import difference,candidate_fraction

OUT=ROOT/'revision/reports/stage10/completion'
RAW=Path('/home/ubuntu/UpfProfilingCampaign/UpfProfilingCampaign/data/raw/loadcore')


def dump(name,data):
    with (OUT/name).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)


def main():
    allrows=[];flows=[];stats={}
    for mode,prefix in [('dpdk','dpdk'),('usr','oai')]:
        source=Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')/f'{prefix}_processed.csv'
        processed=pd.read_csv(source);processed['time']=pd.to_datetime(processed.timestamp,utc=True)
        indices=dict(zip(processed.time,range(len(processed))))
        for path in sorted((RAW/mode).glob('*.zip')):
            with zipfile.ZipFile(path) as outer:
                nested=next(n for n in outer.namelist() if n.endswith('.zip') and not n.startswith('agent'))
                with zipfile.ZipFile(io.BytesIO(outer.read(nested))) as inner:
                    rawconfig=inner.read('config-data.bin');cfg=json.loads(rawconfig)['Config']
                    active=[]
                    for side in ['dn','ue']:
                        for ran in cfg['nodes'][side]['ranges']:
                            for obj in ran.get('userPlane',{}).get('tigerObjective',[]):
                                if not ran.get('enable') or not obj.get('enable') or obj.get('type')!='StatelessUDP':continue
                                udp=obj['statelessUDP']
                                active.append(dict(side=side,direction=udp['direction'],range_count=ran['count'],
                                    flow_id=udp.get('flowId'),dnn_id=udp.get('dnnId'),payload_bytes=udp.get('payloadSize'),
                                    configured_rate=udp.get('rate'),fallback_to_default_flow=udp.get('fallbackToDefaultFlow')))
                    flows.append(dict(mode=mode,archive=str(path),config_sha256=__import__('hashlib').sha256(rawconfig).hexdigest(),
                                      active=active))
                mapping={'ran_tx':'GTPu Packets Tx','ran_rx':'GTPu Packets Rx','lost':'Lost Packets'}
                tables=[]
                for filename,columns in [('Fullcoreapplicationtraffic_GTPuPackets_NGRAN.csv',mapping),
                       ('Fullcoreapplicationtraffic_GTPuPackets_DN.csv',{'dn_tx':'Packets Tx','dn_rx':'Packets Rx'})]:
                    frame=pd.read_csv(io.BytesIO(outer.read(filename)))
                    frame['time']=pd.to_datetime(frame.timestamp,utc=True)
                    if frame.time.duplicated().any():raise ValueError('Duplicate raw timestamp')
                    frame=frame[['time',*columns.values()]].rename(columns={v:k for k,v in columns.items()})
                    tables.append(frame)
                merged=tables[0].merge(tables[1],how='outer',on='time',validate='one_to_one').sort_values('time')
                records=merged.to_dict('records');prev=None
                for row in records:
                    result={};reasons=[]
                    for field in ['ran_tx','ran_rx','lost','dn_tx','dn_rx']:
                        value,reason=difference(prev[field] if prev is not None else None,row[field],
                                                prev['time'] if prev is not None else None,row['time'])
                        result[field+'_delta']=value
                        if reason!='valid':reasons.append(field+':'+reason)
                    lost=result['lost_delta'];rx=result['ran_rx_delta'];tx=result['dn_tx_delta']
                    expected_rx=None if lost is None or rx is None else lost+rx
                    receive_fraction,receive_status=candidate_fraction(lost,expected_rx)
                    sent_fraction,sent_status=candidate_fraction(lost,tx)
                    residual=None if expected_rx is None or tx is None else tx-expected_rx
                    allrows.append(dict(mode=mode,archive=path.name,timestamp=row['time'].isoformat(),
                        source_row=indices.get(row['time']),raw_previous_timestamp=prev['time'].isoformat() if prev is not None else None,
                        **result,unknown_reasons=';'.join(reasons),
                        candidate_receiver_expected_fraction=receive_fraction,receiver_status=receive_status,
                        candidate_sender_fraction=sent_fraction,sender_status=sent_status,
                        dn_sent_minus_ran_received_and_lost=residual,
                        physical_loss_fraction=None,loss_population_authenticated=False))
                    prev=row
        sub=[r for r in allrows if r['mode']==mode and r['source_row'] is not None]
        vals=[r['dn_sent_minus_ran_received_and_lost'] for r in sub if r['dn_sent_minus_ran_received_and_lost'] is not None]
        stats[mode]=dict(processed_source_sha256=sha256(source),processed_rows=len(processed),matched_processed_rows=len(sub),
              valid_numeric_receiver_candidates=sum(r['candidate_receiver_expected_fraction'] is not None for r in sub),
              valid_numeric_sender_candidates=sum(r['candidate_sender_fraction'] is not None for r in sub),
              nonzero_conservation_residuals=sum(v!=0 for v in vals),conservation_comparable=len(vals),
              max_abs_conservation_residual=max(abs(v) for v in vals),
              physical_labels_released=0,interpretation='Numerically aligned candidate denominators are not authenticated packet cohorts; sender/receiver counters are not conservative at arbitrary sample boundaries')
    with (OUT/'interval_feasibility.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    dump('active_flows.json',flows);dump('counter_feasibility_summary.json',stats)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
