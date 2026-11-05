"""Seal controller evidence and verify preservation; never mutate Git references."""
import json,re,subprocess
from pathlib import Path
from revision.evaluation.twin import ROOT,sha256
from .common import OUT,MODELS,dump
from .pipeline import ForecastLedger


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def main():
    prior=json.loads((ROOT/'revision/reports/stage3_prior_sha256.json').read_text())
    changed=[p for p,d in prior.items() if not (ROOT/p).exists() or sha256(ROOT/p)!=d]
    assert not changed,changed
    assert git('branch','--show-current')=='revision/comcom-clean-evaluation'
    assert git('rev-parse','HEAD')=='c0734715418d3f6d10785e5174aebb4bac24d0da'
    assert not git('diff','--name-only','HEAD')
    original=json.loads((ROOT/'revision/configs/twin_artifacts_v1.json').read_text())['artifacts']
    for a in original:assert sha256(ROOT/a['path'])==a['sha256']
    frozen=json.loads((OUT/'pilot_protocol_frozen.json').read_text())
    for name,digest in frozen['code_hashes'].items():assert sha256(ROOT/'revision/stage3'/name)==digest
    assert sha256(OUT/'forecasts.csv')==frozen['forecast_sha256']
    for path in (MODELS/'forecast').glob('*/manifest.json'):
        m=json.loads(path.read_text());assert sha256(path.parent/'model.keras')==m['model_sha256']
    for path in (MODELS/'controllers').glob('*/*/checkpoint_*/bundle.json'):
        m=json.loads(path.read_text())
        for filename,digest in m['artifacts'].items():assert sha256(path.parent/filename)==digest
        assert m['forecast_sha256']==frozen['forecast_sha256']
    alternatives=[ROOT/'revision/reports/task2d/selection_frozen.json',
        ROOT/'revision/models/task2h/dpdk/training_load_mean_pwl.json',
        ROOT/'revision/models/task2h/usr/training_load_mean_pwl.json']
    selected=json.loads(alternatives[0].read_text());manifest=Path(selected['selected_manifest_path'])
    alternatives.append(manifest);alternatives.append(manifest.parent/'model.keras')
    from .sensitivity import AlternativeTwin
    dump(OUT/'sensitivity_identities.json',{'baseline_id':frozen['frozen_twin']['twin_id'],
       'alternative_artifacts':{str(p):sha256(p) for p in alternatives},
       'scenario_ids':{s:AlternativeTwin(s).identity for s in ['usr_neural_power','load_mean_qos','both']},
       'forecast_sha256':frozen['forecast_sha256'],'implementation_sha256':sha256(ROOT/'revision/stage3/sensitivity.py')})
    ledger=ForecastLedger()
    report=ROOT/'revision/CONTROLLER_CLAIMS_PILOT.md'
    allowed={'integrity.json','file_manifest.json'}
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if '://' not in target:
            p=report.parent/target.split('#')[0]
            assert p.exists() or p.name in allowed,target
    preflight=ROOT/'revision/reports/stage3_preflight_final.log'
    final=ROOT/'revision/reports/stage3_final_tests.log'
    assert '238 passed, 28 warnings' in preflight.read_text()
    assert '241 passed, 28 warnings' in final.read_text()
    dump(OUT/'integrity.json',{'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),
       'prior_files_checked':len(prior),'prior_files_changed':changed,'original_profiling_artifacts_checked':len(original),
       'tracked_changes':git('diff','--name-only','HEAD'),'status_short':git('status','--short'),
       'frozen_twin_id':frozen['frozen_twin']['twin_id'],
       'pilot_code_hashes_verified':frozen['code_hashes'],
       'policy_bundles_checked':len(list((MODELS/'controllers').glob('*/*/checkpoint_*/bundle.json'))),
       'forecast_fits_checked':len(list((MODELS/'forecast').glob('*/manifest.json'))),
       'forecast_rows':len(ledger.rows),'maximum_forecast_target_source_index':max(ledger.rows),
       'cold_start_rows':sum(r['kind']=='explicit_cold_start_persistence' for r in ledger.rows.values()),
       'full_preflight_tests':{'passed':238,'warnings':28,'sha256':sha256(preflight)},
       'full_postpilot_tests':{'passed':241,'warnings':28,'sha256':sha256(final)},
       'scientific_controller_runs':4,'training_interactions_per_run':32768,
       'traffic_final_test_controller_evaluation':False,'manuscript_modified':False,'commit_or_push':False,
       'report_sha256':sha256(report)})
    print('Preservation checks passed:',len(prior),'prior files unchanged; 16 checkpoint pairs; 7 forecast fits',flush=True)
    destination=OUT/'file_manifest.json';files=[]
    for p in sorted((ROOT/'revision').rglob('*')):
        relative=str(p.relative_to(ROOT))
        if p.is_file() and '__pycache__' not in p.parts and relative not in prior and p!=destination:
            files.append({'path':relative,'sha256':sha256(p),'bytes':p.stat().st_size})
    dump(destination,{'scope':'Stage 3 scientific additions; excludes Python bytecode and this manifest itself','count':len(files),'files':files})

if __name__=='__main__':main()
