"""Read-only reconstruction of stored loss and embedded tester configurations.

Does not create corrected loss labels or fit a KPI model. All export matches
are timestamp-exact, with no nearest-neighbour repair or imputation.
"""
import csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from revision.evaluation.twin import ROOT,sha256

OUT=ROOT/'revision/reports/stage10/reconciliation_v2'
RAW=Path('/home/ubuntu/UpfProfilingCampaign/UpfProfilingCampaign/data/raw/loadcore')
SOURCE=Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')


def dump(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)


def csvout(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    summary={};configs=[];exports=[];examples=[]
    for mode,prefix in [('dpdk','dpdk'),('usr','oai')]:
        path=SOURCE/f'{prefix}_processed.csv';d=pd.read_csv(path)
        rates=[];packets=[]
        for p in sorted((RAW/mode).glob('*.zip')):
            with zipfile.ZipFile(p) as a:
                for entry,destination in [('Fullcoreapplicationtraffic_GTPuPacketss_NGRAN.csv',rates),
                                          ('Fullcoreapplicationtraffic_GTPuPackets_NGRAN.csv',packets)]:
                    b=a.read(entry)
                    if b:
                        frame=pd.read_csv(io.BytesIO(b));destination.append(frame)
                        exports.append(dict(archive=str(p),entry=entry,sha256=hashlib.sha256(b).hexdigest(),columns=frame.columns.tolist()))
                for n in a.namelist():
                    if n.endswith('.zip') and not n.startswith('agent-'):
                        with zipfile.ZipFile(io.BytesIO(a.read(n))) as inner:
                            if 'config-data.bin' not in inner.namelist():continue
                            b=inner.read('config-data.bin');c=json.loads(b)
                            directions=[]
                            def walk(x,path=''):
                                if isinstance(x,dict):
                                    for k,v in x.items():
                                        if k=='direction' and 'statelessUDP' in path:directions.append({'field':path+'.'+k,'value':v})
                                        walk(v,path+'.'+k)
                                elif isinstance(x,list):
                                    for i,v in enumerate(x):walk(v,path+f'[{i}]')
                            walk(c)
                            configs.append(dict(mode=mode,archive=str(p),nested_entry=n,config_sha256=hashlib.sha256(b).hexdigest(),stateless_udp_directions=directions,
                                 note='Configured entries including possible inactive protocol objects; not full tester metric semantics'))
        base=d.copy();base['time']=pd.to_datetime(base.timestamp,utc=True)
        for name,frames,cols in [('rate',rates,['GTPu Packets Tx/s','GTPu Packets Rx/s']),
                                  ('counter',packets,['GTPu Packets Tx','GTPu Packets Rx','Lost Packets'])]:
            frame=pd.concat(frames,ignore_index=True);frame['time']=pd.to_datetime(frame.timestamp,utc=True)
            assert not frame.time.duplicated().any()
            base=base.merge(frame[['time',*cols]],on='time',how='left',validate='one_to_one',sort=False)
        numerator=base.RAN_pks_lost_interval.to_numpy();tx=base.RAN_Tx_pks.to_numpy();den=tx+numerator
        reproduced=np.divide(numerator,den,out=np.zeros(len(d)),where=den>0)
        error=abs(reproduced-base.RAN_PLR_interval.to_numpy())
        delta=base.groupby(['Test Set','Test Case'],sort=False).time.diff().dt.total_seconds()
        raw_delta=base.groupby(['Test Set','Test Case'],sort=False)['Lost Packets'].diff()
        first=~base[['Test Set','Test Case']].duplicated()
        # First stored interval is zero; report separately, not a measured zero.
        comparable=raw_delta.notna()&~first
        summary[mode]=dict(source=str(path),sha256=sha256(path),n=len(d),
            reconstructed_formula='RAN_pks_lost_interval / (RAN_Tx_pks + RAN_pks_lost_interval), zero denominator -> zero',
            matches_at_1e_12=int((error<1e-12).sum()),max_abs_error=float(error.max()),
            raw_rate_header='GTPu Packets Tx/s',rate_comparable=int(base['GTPu Packets Tx/s'].notna().sum()),
            stored_RAN_Tx_equals_raw_packet_rate=int((abs(base.RAN_Tx_pks-base['GTPu Packets Tx/s'])<1e-12).sum()),
            raw_interval_lost_comparable=int(comparable.sum()),
            raw_loss_delta_matches=int(((abs(raw_delta-numerator)<1e-12)&comparable).sum()),
            first_rows_with_zero_interval_numerator=int((first&(numerator==0)).sum()),
            comparable_sample_seconds={str(k):int(v) for k,v in delta[comparable].value_counts().items()},
            physical_ratio_authenticated=False,
            limitation='Addition mixes packets/second with packets per sample; same-time/direction expected-packet denominator not established. Numerical reconstruction is not authenticated original code or physically valid loss ratio.')
        failed=base[(base.RAN_PLR_interval>0)&(base.RAN_PLR_interval<1)].head(8)
        for i,r in failed.iterrows():
            examples.append(dict(mode=mode,source_row=int(i),timestamp=r.timestamp,sweep=int(r['Test Set']),case=r['Test Case'],
                 interval_lost_packets=float(r.RAN_pks_lost_interval),stored_tx=float(r.RAN_Tx_pks),raw_tx_packets_per_second=float(r['GTPu Packets Tx/s']) if pd.notna(r['GTPu Packets Tx/s']) else None,
                 source_interval_seconds=float(delta.iloc[i]) if pd.notna(delta.iloc[i]) else None,
                 stored_loss_fraction=float(r.RAN_PLR_interval),reconstructed_loss_fraction=float(reproduced[i])))
    dump('loss_reconstruction.json',summary);dump('tester_config_inventory.json',configs)
    dump('raw_export_inventory.json',exports);csvout('loss_examples.csv',examples)
    refs=[ROOT/'revision/reference/1571355221 paper.pdf',ROOT/'revision/reference/Thesis_ShimaAfshar_Final.pdf']
    dump('authoritative_pdf_manifest.json',[dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for p in refs])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
