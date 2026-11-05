import json
from datetime import datetime,timedelta
from dataclasses import asdict
import numpy as np
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.data import Timeline,SplitPlan,LoadScale,DataSpec,prepare
from revision.evaluation.contract import Contract

BASE=ROOT/'revision/stage3'
OUT=ROOT/'revision/reports/stage3'
MODELS=ROOT/'revision/models/stage3'
CFG=json.loads((BASE/'protocol.json').read_text())

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def inputs():
    cfg=json.loads((ROOT/'revision/configs/evaluation_v1.json').read_text())
    raw=Timeline.read_csv(ROOT/cfg['datasets']['netflix'],'netflix',timedelta(minutes=15))
    plan=SplitPlan(**{k:datetime.fromisoformat(v) for k,v in cfg['split'].items()})
    scale=LoadScale.fit_training(plan.partition(raw)['train'],400.)
    data=prepare(raw,plan,DataSpec(**cfg['data_spec']),scale=scale)
    # Return only explicitly authorized controller partitions.
    return data.horizons['train'],data.horizons['validation'],Contract(**cfg['contract'])

def forecast_metrics(y,p):
    y,p=np.asarray(y),np.asarray(p);e=p-y
    return {'n':len(y),'mae_mbps':float(abs(e).mean()),'rmse_mbps':float(np.sqrt((e**2).mean())),
      'bias_mbps':float(e.mean()),'r2':float(1-(e**2).sum()/((y-y.mean())**2).sum())}
