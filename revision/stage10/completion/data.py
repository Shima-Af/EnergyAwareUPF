"""Frozen scenario horizons and newly fitted causal ledgers; never final targets."""
import csv
from datetime import datetime,timedelta
import numpy as np
from revision.evaluation.data import Timeline,Partition,DataSpec,EvaluationHorizon
from revision.evaluation.contract import Forecast
from revision.forecast_service.core import data as prefix,data_hash
from revision.evaluation.twin import sha256
from .independent_fits import CFG,OUT


def horizons(workload,scenario):
    if workload not in ('netflix','youtube') or scenario not in CFG['scenarios']:raise ValueError('Unknown workload/scenario')
    times,loads,_=prefix(workload);spec=CFG['scenarios'][scenario]
    if spec['kind']=='scale':loads=loads*spec['factor']
    else:loads=loads*np.array([spec['factors'][(i//96)%3] for i in range(len(loads))])
    timeline=Timeline(workload,times,tuple(float(v) for v in loads),timedelta(seconds=900),data_hash(loads,times,2520),
                      'development' if workload=='netflix' else 'transfer')
    s=DataSpec(96,96)
    train=EvaluationHorizon(Partition(timeline,'train',0,2016),s) if workload=='netflix' else None
    val=EvaluationHorizon(Partition(timeline,'validation',2016,2520),s)
    return train,val


class Ledger:
    def __init__(self,workload,scenario):
        if workload not in ('netflix','youtube') or scenario not in CFG['scenarios']:raise ValueError('Unknown ledger')
        path=OUT/'rebuilt_forecasts.csv';self.digest=sha256(path);self.rows={}
        with path.open() as f:
            for row in csv.DictReader(f):
                if row['workload']==workload and row['scenario']==scenario:
                    i=int(row['source_index'])
                    if i in self.rows:raise ValueError('Duplicate forecast target')
                    # Actual targets are deliberately not retained by the provider.
                    self.rows[i]={k:v for k,v in row.items() if k!='actual_mbps'}
        expected=range(96,2520) if workload=='netflix' else range(2016,2520)
        if sorted(self.rows)!=list(expected):raise ValueError('Incomplete or out-of-scope forecast ledger')

    def at(self,index,context):
        r=self.rows[index]
        if int(r['fit_stop_exclusive'])>index:raise ValueError('Forecast parameters use future targets')
        issued=datetime.fromisoformat(r['timestamp']);observed=datetime.fromisoformat(r['observed_through'])
        if issued!=context.decision_at or observed!=context.completed_interval_at:raise ValueError('Forecast clock mismatch')
        return Forecast(issued,observed,issued,float(r['forecast_mbps']))
