"""Read-only preservation verification and package freeze; excludes its own manifest."""
import hashlib,json,subprocess
from pathlib import Path
from revision.evaluation.twin import ROOT,sha256
from revision.evaluation.revised_twin import RevisionTwin
from revision.stage3.common import dump
from .evaluate import verify_frozen

OUT=ROOT/'revision/reports/stage5'

def main():
    prior_path=ROOT/'revision/reports/stage5_prior_sha256.json'
    prior=json.loads(prior_path.read_text())
    failures=[p for p,h in prior.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    if failures:raise ValueError('Prior scientific files changed: '+repr(failures))
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tracked=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
    if branch!='revision/comcom-clean-evaluation' or head!='c0734715418d3f6d10785e5174aebb4bac24d0da' or tracked:
        raise ValueError('Git branch/HEAD/tracked state changed')
    frozen=json.loads((OUT/'baseline_frozen.json').read_text());verify_frozen(frozen)
    twin=RevisionTwin();twin.verify_artifacts()
    if twin.identity!=frozen['twin_id']:raise ValueError('Twin changed')
    uploaded=json.loads((OUT/'uploaded_source_reconciliation.json').read_text())
    if sha256(ROOT/uploaded['uploaded_path'])!=uploaded['uploaded_sha256']:raise ValueError('Uploaded reference changed')
    results=json.loads((OUT/'comparison.json').read_text())
    if not results['same_horizon'] or results['final_test_accessed']:raise ValueError('Evaluation scope mismatch')
    dump(OUT/'integrity.json',{'prior_scientific_files_checked':len(prior),'preservation_failures':failures,
       'preservation_snapshot_sha256':sha256(prior_path),'branch':branch,'head':head,'tracked_changes':tracked,
       'baseline_definition_sha256':sha256(OUT/'baseline_frozen.json'),'twin_identity':twin.identity,
       'selected_policy_bundles_unchanged':True,'uploaded_reference_sha256':uploaded['uploaded_sha256'],
       'scientific_or_fixture_neural_rl_training_performed':False,'default_twin_changed':False,
       'new_controller_comparisons':len(results['controllers']),'tests_log_sha256':sha256(OUT/'tests.log'),
       'test_result':'272 passed, 3 training-fixture cases deselected','final_test_evaluation':False,
       'commit_or_push':False})
    # Collect before opening the manifest; no self-reference and no streaming log hash.
    added={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'revision').rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
           and str(p.relative_to(ROOT)) not in prior}
    payload={'version':'comcom.baseline_freeze.v1','definition_freeze_sha256':sha256(OUT/'baseline_frozen.json'),
       'definition_frozen_before_validation_replay':True,'git_branch':branch,'git_head':head,
       'source_and_input_hashes':frozen['source_hashes'],
       'user_uploaded_source':{'path':uploaded['uploaded_path'],'sha256':uploaded['uploaded_sha256'],'origin':'User uploaded during Stage 5; inspected read-only'},
       'existing_selected_policy_bundles':frozen['existing_policy_selections'],
       'new_files':added,'new_file_count_excluding_this_manifest':len(added),
       'self_hash_note':'This manifest excludes itself. Freeze identity hashes canonical payload before adding freeze_id.'}
    identity=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dump(OUT/'baseline_freeze_manifest.json',{'freeze_id':identity,**payload})
    print('Verified prior files:',len(prior),'new files:',len(added),'freeze:',identity,flush=True)

if __name__=='__main__':main()
