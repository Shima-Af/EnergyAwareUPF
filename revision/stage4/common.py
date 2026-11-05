import json
from revision.evaluation.twin import ROOT
from revision.stage3.common import inputs,dump,forecast_metrics
BASE=ROOT/'revision/stage4'
OUT=ROOT/'revision/reports/stage4'
MODELS=ROOT/'revision/models/stage4'
CFG=json.loads((BASE/'protocol.json').read_text())
