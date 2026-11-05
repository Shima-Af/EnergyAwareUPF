"""Seal the consolidated corrected Stage4 evidence and both preserved campaigns."""
import json,re,subprocess
from pathlib import Path
from revision.evaluation.twin import ROOT,sha256
from .common import CFG,dump
OUT=ROOT/'revision/reports/stage4_rng_isolated'
MODELS=ROOT/'revision/models/stage4_rng_isolated'


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def main():
    snapshots={}
    for name in ['stage4_prior_sha256.json','stage4_initial_campaign_sha256.json']:
        p=ROOT/'revision/reports'/name;records=json.loads(p.read_text())
        changed=[path for path,h in records.items() if not (ROOT/path).is_file() or sha256(ROOT/path)!=h]
        assert not changed,changed
        snapshots[name]={'checked':len(records),'changed':changed,'sha256':sha256(p)}
    assert git('branch','--show-current')=='revision/comcom-clean-evaluation'
    assert git('rev-parse','HEAD')=='c0734715418d3f6d10785e5174aebb4bac24d0da'
    assert not git('diff','--name-only','HEAD')
    frozen=json.loads((OUT/'training_protocol_frozen.json').read_text())
    assert frozen['protocol_sha256']==sha256(ROOT/'revision/stage4/protocol.json')
    assert frozen['stage3_pipeline_sha256']==sha256(ROOT/'revision/stage3/pipeline.py')
    assert frozen['twin_config_sha256']==sha256(ROOT/'revision/configs/revised_twin_v1.json')
    for name,h in frozen['stage4_implementation_hashes'].items():assert sha256(ROOT/'revision/stage4'/name)==h
    for root in ['stage4','stage4_rng_isolated']:
        for alg in CFG['algorithms']:
            for seed in CFG['seeds']:
                p=ROOT/f'revision/models/{root}/controllers/{alg}/seed_{seed}'
                s=json.loads((p/'selection.json').read_text());assert s['actual_timesteps']==300000
                assert [x['timesteps'] for x in s['validation_scores']]==CFG['checkpoint_timesteps']
                assert json.loads((p/'result.json').read_text())['deterministic_repeat_equal']
        for p in (ROOT/f'revision/models/{root}/controllers').glob('*/*/checkpoint_*/bundle.json'):
            m=json.loads(p.read_text())
            for filename,h in m['artifacts'].items():assert sha256(p.parent/filename)==h
    for name in ['forecasts.csv','persistence_forecasts.csv','forecast_selection.json']:
        assert sha256(OUT/name)==sha256(ROOT/'revision/reports/stage4'/name)
    for p in (MODELS/'forecast').rglob('*'):
        if p.is_file():assert sha256(p)==sha256(ROOT/'revision/models/stage4/forecast'/p.relative_to(MODELS/'forecast'))
    original=json.loads((ROOT/'revision/configs/twin_artifacts_v1.json').read_text())['artifacts']
    for a in original:assert sha256(ROOT/a['path'])==a['sha256']
    report=ROOT/'revision/FULL_TRAINING_EVALUATION.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if '://' not in target:
            p=report.parent/target.split('#')[0]
            assert p.exists() or p.name in ('integrity.json','file_manifest.json'),target
    final=ROOT/'revision/reports/stage4_final_tests.log';preflight=ROOT/'revision/reports/stage4_isolated_preflight_tests.log'
    assert re.search(r'\b[0-9]+ passed\b',final.read_text()) and not re.search(r'(?m)^(FAILED|ERROR) ',final.read_text())
    dump(OUT/'integrity.json',{'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),
       'tracked_changes':git('diff','--name-only','HEAD'),'status_short':git('status','--short'),
       'preservation_snapshots':snapshots,'primary_runs':10,'initial_rng_affected_runs_preserved':10,
       'primary_interactions':3000000,'all_scientific_controller_interactions_including_affected':6000000,
       'primary_checkpoint_pairs':140,'all_checkpoint_pairs':280,
       'scientific_forecast_fits':7,'corrected_forecast_artifacts_identical_copies':True,
       'rng_isolation_implementation_sha256':sha256(ROOT/'revision/stage4/rng_isolation.py'),
       'full_preflight_log_sha256':sha256(preflight),'full_posttraining_log_sha256':sha256(final),
       'posttraining_result':final.read_text().splitlines()[-1],'report_sha256':sha256(report),
       'final_test_controller_outcomes_accessed':False,'manuscript_edit':False,'commit_or_push':False})
    print('Preservation and corrected Stage4 completion verified',flush=True)
    prior=json.loads((ROOT/'revision/reports/stage4_prior_sha256.json').read_text());destination=OUT/'file_manifest.json'
    files=[]
    for p in sorted((ROOT/'revision').rglob('*')):
        relative=str(p.relative_to(ROOT))
        if p.is_file() and '__pycache__' not in p.parts and relative not in prior and p!=destination:
            files.append({'path':relative,'sha256':sha256(p),'bytes':p.stat().st_size})
    dump(destination,{'count':len(files),'scope':'All Stage4 scientific additions, including affected campaign and RNG correction',
                      'excludes':['incidental bytecode','this self-referential manifest'],'files':files})

if __name__=='__main__':main()
