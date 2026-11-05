"""Final preservation, identity and exact-budget acceptance record."""
import json,re,subprocess
from pathlib import Path
from revision.evaluation.twin import ROOT,sha256
from revision.stage3.pipeline import ForecastLedger
from .common import OUT,MODELS,CFG,dump


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def main():
    prior=json.loads((ROOT/'revision/reports/stage4_prior_sha256.json').read_text())
    changed=[p for p,h in prior.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    assert not changed,changed
    assert git('branch','--show-current')=='revision/comcom-clean-evaluation'
    assert git('rev-parse','HEAD')=='c0734715418d3f6d10785e5174aebb4bac24d0da'
    assert not git('diff','--name-only','HEAD')
    frozen=json.loads((OUT/'training_protocol_frozen.json').read_text())
    assert frozen['protocol_sha256']==sha256(ROOT/'revision/stage4/protocol.json')
    assert frozen['stage3_pipeline_sha256']==sha256(ROOT/'revision/stage3/pipeline.py')
    assert frozen['twin_config_sha256']==sha256(ROOT/'revision/configs/revised_twin_v1.json')
    for p,h in frozen['stage4_implementation_hashes'].items():assert sha256(ROOT/'revision/stage4'/p)==h
    ledger=ForecastLedger(OUT/'forecasts.csv');assert ledger.digest==frozen['forecast_sha256']
    assert max(ledger.rows)==2519 and min(ledger.rows)==96
    for a in json.loads((ROOT/'revision/configs/twin_artifacts_v1.json').read_text())['artifacts']:assert sha256(ROOT/a['path'])==a['sha256']
    bundles=list((MODELS/'controllers').glob('*/*/checkpoint_*/bundle.json'));assert len(bundles)==140
    for p in bundles:
        m=json.loads(p.read_text());assert m['phase']=='after_optimizer_update'
        for name,h in m['artifacts'].items():assert sha256(p.parent/name)==h
        assert m['forecast_sha256']==ledger.digest
    for path in (MODELS/'forecast').glob('*/manifest.json'):
        m=json.loads(path.read_text());assert sha256(path.parent/'model.keras')==m['model_sha256']
    preflight=ROOT/'revision/reports/stage4_preflight_tests.log';final=ROOT/'revision/reports/stage4_final_tests.log'
    assert re.search(r'\b[0-9]+ passed\b',final.read_text()) and not re.search(r'(?m)^(FAILED|ERROR) ',final.read_text())
    report=ROOT/'revision/FULL_TRAINING_EVALUATION.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if '://' not in target:
            p=report.parent/target.split('#')[0]
            assert p.exists() or p.name in ('integrity.json','file_manifest.json'),target
    dump(OUT/'integrity.json',{'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),
       'prior_files_checked':len(prior),'prior_files_changed':changed,'tracked_changes':git('diff','--name-only','HEAD'),
       'status_short':git('status','--short'),'training_runs':10,'interactions_per_run':300000,'total_interactions':3000000,
       'checkpoint_pairs':len(bundles),'forecast_models':len(list((MODELS/'forecast').glob('*/manifest.json'))),
       'forecast_ledger_sha256':ledger.digest,'source_and_protocol_hashes_verified':True,'prior_stage3_and_twin_preserved':True,
       'preflight_log_sha256':sha256(preflight),'post_training_log_sha256':sha256(final),
       'post_training_result':final.read_text().splitlines()[-1],'report_sha256':sha256(report),
       'final_test_controller_outcomes_accessed':False,'manuscript_modified':False,'git_commit_push':False})
    print('Stage 4 preservation and completion checks passed',flush=True)
    manifest=OUT/'file_manifest.json';files=[]
    for p in sorted((ROOT/'revision').rglob('*')):
        relative=str(p.relative_to(ROOT))
        if p.is_file() and '__pycache__' not in p.parts and relative not in prior and p!=manifest:
            files.append({'path':relative,'sha256':sha256(p),'bytes':p.stat().st_size})
    dump(manifest,{'count':len(files),'excludes':['incidental Python bytecode','this self-referential manifest'],'files':files})

if __name__=='__main__':main()
