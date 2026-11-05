"""Read-only raw-counter/unit checks for the frozen recovered QoS definition."""

import argparse
import hashlib
import io
from pathlib import Path
import numpy as np
import pandas as pd

from revision.evaluation.twin import ROOT, sha256
from revision.evaluation.validate_static import write_csv
from revision.profiling.refit_usr_power import write_json

import zipfile


def run(output):
    inventories,details,loss_checks,summary=[],[],[],{}
    for mode,prefix in [('dpdk','dpdk'),('usr','oai')]:
        source=Path('/home/ubuntu/DDT_UPF_Selection_DRL/data/processed')/f'{prefix}_processed.csv'
        f=pd.read_csv(source)
        f['datetime']=pd.to_datetime(f.timestamp,format='ISO8601')
        base=Path('/home/ubuntu/UpfProfilingCampaign/UpfProfilingCampaign/data/raw/loadcore')/mode
        tables={k:[] for k in ['throughput','jitter','ran_packets','dn_packets']}
        entries={'throughput':'Fullcoreapplicationtraffic_GTPuKbitss_DN.csv',
                 'jitter':'Fullcoreapplicationtraffic_DelayVariationJitterAverage_RAN.csv',
                 'ran_packets':'Fullcoreapplicationtraffic_GTPuPackets_NGRAN.csv',
                 'dn_packets':'Fullcoreapplicationtraffic_GTPuPackets_DN.csv'}
        for path in sorted(base.glob('*.zip')):
            with zipfile.ZipFile(path) as archive:
                for kind,name in entries.items():
                    b=archive.read(name)
                    if not b:continue
                    t=pd.read_csv(io.BytesIO(b))
                    t['datetime']=pd.to_datetime(t.timestamp,format='ISO8601',utc=True).dt.tz_localize(None)
                    if t.datetime.duplicated().any():raise ValueError('Duplicate raw timestamps inside one export')
                    tables[kind].append(t)
                    inventories.append({'mode':mode,'zip_path':str(path),'entry':name,
                      'entry_sha256':hashlib.sha256(b).hexdigest(),'rows':len(t),'columns':list(t.columns)})
        merged=f.copy()
        maps={'throughput':{'Kbits Rx/s':'raw_dn_rx_kbps'},'jitter':{'Avg us':'raw_ran_jitter_us'},
              'ran_packets':{'GTPu Packets Tx':'raw_ran_tx_cumulative','GTPu Packets Rx':'raw_ran_rx_cumulative',
                             'Lost Packets':'raw_ran_lost_cumulative'},
              'dn_packets':{'Packets Tx':'raw_dn_tx_cumulative','Packets Rx':'raw_dn_rx_cumulative',
                            'Packets Lost':'raw_dn_lost_cumulative'}}
        for kind,parts in tables.items():
            table=pd.concat(parts,ignore_index=True)
            if table.datetime.duplicated().any():raise ValueError('Duplicate timestamps across raw exports')
            cols=maps[kind]
            table=table[['datetime',*cols]].rename(columns=cols)
            merged=merged.merge(table,on='datetime',how='left',validate='one_to_one',sort=False)
        missing={v:int(merged[v].isna().sum()) for m in maps.values() for v in m.values()}
        throughput_error=abs(merged['DN_Rx_Mbits/s']-merged.raw_dn_rx_kbps/1000)
        jitter_error=abs(merged.RAN_jitter-merged.raw_ran_jitter_us)
        mode_summary={'source_path':str(source),'source_sha256':sha256(source),'rows':len(f),
          'raw_zip_count':len(list(base.glob('*.zip'))),'unmatched_raw_fields':missing,
          'dn_rx_mbps_equals_raw_kbps_div1000':int((throughput_error<=1e-12).sum()),
          'max_throughput_conversion_difference_mbps':float(throughput_error.max()),
          'ran_jitter_equals_raw_Avg_us':int((jitter_error==0).sum()),
          'max_jitter_difference_us':float(jitter_error.max())}
        for side in ['RAN','DN']:
            key=side.lower();lost=merged[side+'_pks_lost_interval'].to_numpy()
            raw_lost=merged['raw_'+key+'_lost_cumulative']
            first=~merged['Test Case'].duplicated()
            lost_diff=raw_lost.groupby(merged['Test Case'],sort=False).diff()
            lost_diff.loc[first & raw_lost.notna()]=0
            lost_delta=lost_diff.clip(lower=0).to_numpy()
            mode_summary[key+'_stored_cumulative_loss_matches_raw']=int((abs(merged[side+'_pks_lost']-raw_lost)<=1e-10).sum())
            mode_summary[key+'_lost_interval_equals_casewise_nonnegative_raw_counter_diff']=int((abs(lost-lost_delta)<=1e-10).sum())
            for received in ['tx','rx']:
                counter=merged['raw_'+key+'_'+received+'_cumulative']
                differences=counter.groupby(merged['Test Case'],sort=False).diff()
                differences.loc[first & counter.notna()]=0
                delta=differences.clip(lower=0).to_numpy()
                for add_lost in [False,True]:
                    denominator=delta+lost if add_lost else delta
                    rate=np.divide(lost,denominator,out=np.zeros(len(f)),where=denominator>0)
                    rate=np.clip(rate,0,1)
                    error=abs(rate-merged[side+'_PLR_interval'].to_numpy())
                    comparable=np.isfinite(denominator)
                    loss_checks.append({'mode':mode,'side':side,'denominator':'delta_raw_'+received+('_plus_lost' if add_lost else ''),
                       'numerator':'nonnegative casewise delta lost; first row zero',
                       'zero_denominator_behavior':'zero','output_behavior':'clip ratio to [0,1]',
                       'rows':len(f),'comparable_rows':int(comparable.sum()),
                       'matches_at_1e-10':int(((error<=1e-10)&comparable).sum()),
                       'mae':float(error[comparable].mean()),'max_abs_difference':float(error[comparable].max())})
        def number(x):return float(x) if pd.notna(x) else None
        for _,row in merged.iterrows():
            details.append({'mode':mode,'timestamp':row.timestamp,'test_case':row['Test Case'],
                'stored_dn_rx_mbps':float(row['DN_Rx_Mbits/s']),'raw_dn_rx_kbps':number(row.raw_dn_rx_kbps),
                'stored_ran_jitter':float(row.RAN_jitter),'raw_ran_jitter_us':number(row.raw_ran_jitter_us),
                'stored_ran_plr':float(row.RAN_PLR_interval),
                'stored_ran_interval_lost':float(row.RAN_pks_lost_interval),
                'raw_ran_cumulative_tx':number(row.raw_ran_tx_cumulative),
                'raw_ran_cumulative_rx':number(row.raw_ran_rx_cumulative),
                'raw_ran_cumulative_lost':number(row.raw_ran_lost_cumulative)})
        summary[mode]=mode_summary
        print(mode,mode_summary,flush=True)
    write_json(output/'raw_unit_counter_summary.json',summary)
    write_json(output/'raw_export_entries.json',inventories)
    write_csv(output/'raw_row_checks.csv',details)
    write_csv(output/'loss_denominator_checks.csv',loss_checks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();path=args.output_dir.resolve()
    if not path.is_relative_to(ROOT/'revision'):raise ValueError('Diagnostics must remain under revision/')
    path.mkdir(parents=True,exist_ok=False);run(path)


if __name__=='__main__':main()
