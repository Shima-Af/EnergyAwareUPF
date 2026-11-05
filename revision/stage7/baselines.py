"""Unchanged Stage5 families plus separately versioned training-calibrated A."""
import json,time
import numpy as np
from revision.stage5.baselines import SlicesExtendedManual,ComcomHistorical
from revision.stage3.pipeline import RuleController
from revision.evaluation.static import StaticController
from .common import ROOT,CFG


def definitions(twin,workload):
    f=json.loads((ROOT/'revision/reports/stage5/baseline_frozen.json').read_text())
    def manual(margin):return SlicesExtendedManual(twin,f['slices_regions'],f['normalization_min_power_w'],f['normalization_max_power_w'],margin)
    return [('slices_extended_manual_26',manual(26.)),('slices_recalibrated_v2',manual(CFG['recalibrated_margins'][workload])),
            ('strong_rule_stage4',RuleController(twin)),('comcom_historical',ComcomHistorical(twin)),
            *[(f'static_{a}',StaticController(a)) for a in range(3)]]


class Timed:
    def __init__(self,controller):self.controller=controller
    def reset(self,*,seed):self.controller.reset(seed=seed);self.latency_ns=[]
    def predict(self,o,*,deterministic):
        start=time.perf_counter_ns();action=self.controller.predict(o,deterministic=deterministic)
        self.latency_ns.append(time.perf_counter_ns()-start);return action
