"""Post-pilot frozen-policy outcome/telemetry sensitivity. No model selection."""
import json
from pathlib import Path
import numpy as np
import torch
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.data import fingerprint
from revision.evaluation.contract import Outcome
from revision.evaluation.static import StaticController
from revision.evaluation.revised_twin import RevisionTwin
from revision.profiling.neural_power import NeuralPowerModel
from revision.profiling.neural_qos import SimpleQoSModel
from .common import CFG,OUT,MODELS,inputs,dump
from .pipeline import ForecastLedger,RuleController,run_controller,summarize
from .pilot import load_bundle,export_steps

class AlternativeTwin:
    def __init__(self,scenario):
        self.base=RevisionTwin();self.scenario=scenario;self.cache={}
        self.identity=fingerprint({'baseline':self.base.identity,'scenario':scenario,'alternative_manifest':sha256(ROOT/'revision/reports/task2d/selection_frozen.json'),'qos_reference_hashes':{m:sha256(ROOT/f'revision/models/task2h/{m}/training_load_mean_pwl.json') for m in ['dpdk','usr']}})
        selection=json.loads((ROOT/'revision/reports/task2d/selection_frozen.json').read_text())
        self.power=NeuralPowerModel(Path(selection['selected_manifest_path']))
        self.qos={m:SimpleQoSModel(json.loads((ROOT/f'revision/models/task2h/{m}/training_load_mean_pwl.json').read_text())) for m in ['dpdk','usr']}
    def outcome(self,a,load):
        if (a,load) in self.cache:return self.cache[a,load]
        original=self.base.outcome(a,load);k=2 if a==2 else 1;per=load/k
        power=k*float(self.power.predict([per])[0]) if a>0 and self.scenario in ('usr_neural_power','both') else original.power_w
        qos=float(self.qos['dpdk' if a==0 else 'usr'].predict([per])[0]) if self.scenario in ('load_mean_qos','both') else original.qos
        out=Outcome(power,qos);self.cache[a,load]=out;return out


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    _,val,contract=inputs();ledger=ForecastLedger();base=json.loads((OUT/'controller_comparison.json').read_text())['controllers']
    comparison={};stepdir=OUT/'sensitivity';stepdir.mkdir()
    for scenario in ['usr_neural_power','load_mean_qos','both']:
        twin=AlternativeTwin(scenario);metrics={}
        for name,entry in base.items():
            if name.startswith('static_'):controller=StaticController(int(name[-1]))
            elif name=='rule_three_action':controller=RuleController(twin)
            else:
                algorithm='recurrent_ppo' if name.startswith('recurrent_ppo') else 'ppo'
                controller=load_bundle(ROOT/entry['selected_checkpoint'],algorithm,val,twin.base,ledger,contract)
            # Bundle identity is verified against the frozen baseline; scenario gets its own outcome identity.
            # Scenario identity is recorded separately and never becomes a newly selected model.
            r=run_controller(val,controller,twin,ledger,contract,name,sha256(ROOT/'revision/stage3/protocol.json'))
            metrics[name]=summarize(r)
            metrics[name]['energy_delta_from_base_wh']=metrics[name]['energy_wh']-entry['energy_wh']
            baseline_requests=__import__('pandas').read_csv(OUT/f'{name}_steps.csv').requested.to_numpy()
            requests=np.asarray([s.transition.requested for s in r.steps])
            metrics[name]['request_changes_from_base']=int((requests!=baseline_requests).sum())
            export_steps(f'{scenario}_{name}',r,stepdir)
        comparison[scenario]=metrics
    dump(OUT/'twin_sensitivity.json',{'validation_only':True,'no_retraining_or_reselection':True,
        'meaning':'Fixed policies re-executed with alternative outcomes and resulting past telemetry; rule curves also use alternative twin',
        'scenarios':comparison,
        'alternatives':{'usr_power':'Task 2D validation-selected original_hidden_softplus_seed_42',
                        'qos':'Task 2H training-only per-load mean curves'},
        'not_measured_risk_or_physical_safety':True})
    print('Sensitivity complete',flush=True)

if __name__=='__main__':main()
