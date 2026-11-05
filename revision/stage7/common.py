import json
from datetime import timedelta
from pathlib import Path
import numpy as np
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.data import Timeline,Partition,DataSpec,EvaluationHorizon
from revision.evaluation.contract import Contract
from revision.stage3.common import dump
from revision.forecast_service.core import data,data_hash
BASE=ROOT/'revision/stage7';OUT=ROOT/'revision/reports/stage7';MODELS=ROOT/'revision/models/stage7'
CFG=json.loads((BASE/'protocol.json').read_text())


def horizons(workload):
    if workload not in ('netflix','youtube'):raise ValueError('Unknown workload')
    times,loads,scale=data(workload)
    source=json.loads((ROOT/f'revision/reports/forecast_service/{workload}/data_manifest.json').read_text())
    if data_hash(loads,times,2520)!=source['authorized_prefix_sha256']:raise ValueError('Authorized workload prefix changed')
    timeline=Timeline(workload,times,tuple(loads),timedelta(minutes=15),source['authorized_prefix_sha256'],
                      'development' if workload=='netflix' else 'transfer')
    spec=DataSpec(96,96,1)
    val=EvaluationHorizon(Partition(timeline,'validation',2016,2520),spec)
    train=EvaluationHorizon(Partition(timeline,'train',0,2016),spec) if workload=='netflix' else None
    contract=Contract(**json.loads((ROOT/'revision/configs/evaluation_v1.json').read_text())['contract'])
    return train,val,contract


def run_name(algorithm,schema,seed):return f'{algorithm}_{schema}_{seed}'

def directory(algorithm,schema,seed):return MODELS/algorithm/schema/f'seed_{seed}'
