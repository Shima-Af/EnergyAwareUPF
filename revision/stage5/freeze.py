"""Freeze profiling-only regions and artifact identities before validation scoring."""
from dataclasses import asdict
import json,subprocess
from pathlib import Path
import numpy as np
from revision.evaluation.revised_twin import RevisionTwin
from revision.evaluation.twin import ROOT,sha256
from revision.stage3.common import dump
from .baselines import utility,profile_regions

BASE=ROOT/'revision/stage5'
OUT=ROOT/'revision/reports/stage5'
CFG=json.loads((BASE/'protocol.json').read_text())


def main():
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='revision/comcom-clean-evaluation':
        raise ValueError('Wrong working branch')
    twin=RevisionTwin()
    if twin.identity!=CFG['frozen_twin']:raise ValueError('Wrong twin')
    lo,hi=CFG['profile_domain_mbps'];step=CFG['profile_grid_step_mbps']
    grid=np.r_[lo,np.arange(step,hi+step/2,step)]
    outcomes=[]
    for i,load in enumerate(grid):
        outcomes.append([twin.outcome(a,float(load)) for a in range(3)])
        if not i%400:print('Profiling-only grid',i,'/',len(grid),flush=True)
    minimum=min(o.power_w for row in outcomes for o in row)
    maximum=max(o.power_w for row in outcomes for o in row)
    def au(load):
        return [utility(o.qos,o.power_w,performance_weight=.2,energy_weight=.8,minimum=minimum,maximum=maximum)
                for o in [twin.outcome(a,load) for a in range(3)]]
    def cu(load):
        return [.3*o.qos+.7*100.*(3.58-o.power_w)/3.58 for o in [twin.outcome(a,load) for a in (0,1)]]
    values=[[utility(o.qos,o.power_w,performance_weight=.2,energy_weight=.8,minimum=minimum,maximum=maximum) for o in row] for row in outcomes]
    regions=profile_regions(grid,values,au,CFG['root_xtol_mbps'])
    cvalues=[[.3*o.qos+.7*100.*(3.58-o.power_w)/3.58 for o in row[:2]] for row in outcomes]
    cregions=profile_regions(grid,cvalues,cu,CFG['root_xtol_mbps'])
    from revision.evaluation.validate_static import write_csv
    rows=[]
    for load,row,utilities,cutilities in zip(grid,outcomes,values,cvalues):
        for action,(o,u) in enumerate(zip(row,utilities)):
            rows.append({'load_mbps':float(load),'action':action,'power_w':o.power_w,'qos':o.qos,
                         'slices_utility':u,'comcom_utility':cutilities[action] if action<2 else None})
    write_csv(OUT/'profiling_curves.csv',rows)
    selections={}
    for algorithm in CFG['algorithms']:
        for seed in CFG['seeds']:
            p=ROOT/CFG['policy_selections'].format(algorithm=algorithm,seed=seed)
            selection=json.loads(p.read_text());checkpoint=ROOT/selection['selected_checkpoint']
            selections[f'{algorithm}_{seed}']={'selection_path':str(p.relative_to(ROOT)),'selection_sha256':sha256(p),
                'selected_checkpoint':selection['selected_checkpoint'],
                'artifacts':{str(f.relative_to(ROOT)):sha256(f) for f in [checkpoint/n for n in ('bundle.json','policy.zip','normalizer.pkl')]}}
    inputs=[BASE/'protocol.json',*BASE.glob('*.py'),ROOT/CFG['forecasts'],ROOT/'revision/stage3/pipeline.py',
            ROOT/'revision/stage3/protocol.json',ROOT/'revision/stage4/train.py',ROOT/'revision/evaluation/contract.py',
            *[p for p in (OUT/'sources').glob('*') if p.is_file()]]
    dump(OUT/'baseline_frozen.json',{'version':CFG['version'],'configuration':CFG,'twin_id':twin.identity,
         'twin_manifest':twin.manifest,'normalization_min_power_w':minimum,'normalization_max_power_w':maximum,
         'slices_regions':[asdict(r) for r in regions],'comcom_utility_regions':[asdict(r) for r in cregions],
         'profiling_grid_points':len(grid),'profiling_grid_sha256':sha256(OUT/'profiling_curves.csv'),
         'construction_uses_workload_outcomes':False,'validation_used_for_rule_selection':False,
         'improved_strong_rule_selected':False,'existing_policy_selections':selections,
         'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in inputs},
         'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'main_head':subprocess.check_output(['git','rev-parse','main'],text=True).strip(),
         'paper_url':'https://networking.ifip.org/2025/images/Net25_papers/1571142002.pdf',
         'root_precision_note':'Finite .25 Mbps scan, bracket refinement to 1e-5 Mbps; not proof that narrower unsampled regions are absent'})
    twin.verify_artifacts()
    print('FROZEN SLICES REGIONS',regions,'COMCOM',cregions,flush=True)

if __name__=='__main__':main()
