"""Additional label consistency and raw-export inventory; no label replacement."""

import io
from pathlib import Path
import zipfile
import pandas as pd

from .qos_reliability import DATA, LOAD, TARGET
from revision.evaluation.twin import ROOT, sha256
from revision.evaluation.validate_static import write_csv
from revision.profiling.refit_usr_power import write_json


def main():
    output=ROOT/'revision/reports/task2e'
    groups,examples,inventory,summary=[],[],[],{}
    columns=[LOAD,'downlink_delay_ms','RAN_jitter','RAN_PLR_interval']
    for mode,prefix in [('dpdk','dpdk'),('usr','oai')]:
        f=pd.read_csv(DATA/f'{prefix}_processed.csv')
        count,rows,maxspan=0,0,0.
        for key,block in f.groupby(columns):
            span=float(block[TARGET].max()-block[TARGET].min())
            if len(block)<2 or span<=1e-8:continue
            count+=1;rows+=len(block);maxspan=max(maxspan,span)
            groups.append({'mode':mode,**dict(zip(columns,key)),'rows':len(block),
                'minimum_qos':float(block[TARGET].min()),'maximum_qos':float(block[TARGET].max()),'span':span})
            if span>=.39:
                for i in [block[TARGET].idxmin(),block[TARGET].idxmax()]:
                    r=f.loc[i]
                    examples.append({'mode':mode,'source_row':int(i),'timestamp':r.timestamp,
                        'test_case':r['Test Case'],'sweep':int(r['Test Set']),
                        **{k:float(r[k]) for k in columns},'stored_qos_w':float(r[TARGET])})
        summary[mode]={'same_recorded_load_and_DL_KPIs_but_different_labels_groups':count,
                       'rows_in_those_groups':rows,'maximum_qos_span':maxspan,
                       'interpretation':'Omitted components, hidden precision or pipeline mismatch remain possible; this does not authenticate a different scoring formula'}
        base=Path('/home/ubuntu/UpfProfilingCampaign/UpfProfilingCampaign/data/raw/loadcore')/mode
        paths=sorted(base.glob('*.zip'))
        # Representative raw container only: not an authenticated all-row KPI join.
        p=paths[0]
        with zipfile.ZipFile(p) as z:
            headers=[]
            for name in z.namelist():
                if name.endswith('.csv') and ('Delay' in name or 'Jitter' in name):
                    data=z.read(name)
                    if data:headers.append({'entry':name,'header':data.decode(errors='replace').splitlines()[0]})
            inventory.append({'mode':mode,'directory':str(base),'zip_files':len(paths),
                'representative_zip':str(p),'sha256':sha256(p),'nonempty_delay_jitter_headers':headers,
                'authenticated_processed_KPI_join':False})
    write_csv(output/'same_kpi_label_discrepancies.csv',groups)
    write_csv(output/'label_discrepancy_examples.csv',examples)
    write_json(output/'label_consistency.json',{'groups':summary,'raw_export_inventory':inventory,
        'manuscript_formula_check_note':'Manuscript gives weights/thresholds but not a complete component-scoring implementation. The numerical check applies notebook-style bounded linear scores and an explicitly assumed microseconds-to-ms jitter conversion, not every possible bounded function.',
        'newer_campaign_pipeline_not_substituted':True})


if __name__=='__main__':main()
