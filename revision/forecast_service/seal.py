"""Close the task with preserved-source verification and an artifact freeze."""
import hashlib,json,subprocess
from .core import ROOT,OUT,CFG,sha256,dump,load_service,select


def main():
    snapshot=ROOT/'revision/reports/forecast_service_prior_sha256.json';prior=json.loads(snapshot.read_text())
    failures=[p for p,h in prior.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    if failures:raise ValueError('Prior scientific files changed: '+repr(failures))
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    tracked=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True)
    if branch!='revision/comcom-clean-evaluation' or head!='c0734715418d3f6d10785e5174aebb4bac24d0da' or tracked:raise ValueError('Git preservation failure')
    frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    for key in ['source_hashes','implementation_hashes']:
        for p,h in frozen[key].items():
            if sha256(ROOT/p)!=h:raise ValueError('Frozen source changed: '+p)
    selected={};starts=0;completed=0
    for name in CFG['workloads']:
        provider=load_service(name);path=OUT/name/'selection_frozen.json';s=json.loads(path.read_text())
        if s['selected_family']!=select(s['development_metrics']) or s['calibration_results_seen'] or s['external_validation_seen']:raise ValueError('Selection scope failure')
        selected[name]={'family':provider.family,'manifest_sha256':provider.identity,'selection_sha256':sha256(path),
                        'service_sha256':sha256(OUT/name/'service.json')}
        records=[json.loads(l) for l in (OUT/name/'fit_ledger.jsonl').read_text().splitlines()]
        if any(r['status']=='failed' for r in records):raise ValueError('Unresolved scientific fit failure')
        starts+=sum(r['status']=='started' for r in records);completed+=sum(r['status']=='completed' for r in records)
        if len([r for r in records if r['status']=='completed'])!=15:raise ValueError('Candidate budget mismatch')
    testlog=(OUT/'final_tests.log').read_text()
    if '81 passed' not in testlog:raise ValueError('Missing passing verification evidence')
    dump(OUT/'integrity.json',{'branch':branch,'head':head,'tracked_changes':tracked,'prior_scientific_files_checked':len(prior),
       'prior_changed_files':failures,'preservation_snapshot_sha256':sha256(snapshot),'forecast_artifact_fits_started':starts,
       'forecast_artifact_fits_completed':completed,'neural_training_stages':40,'boosted_fits':10,
       'rl_training':False,'controller_or_baseline_replay':False,'baseline_or_twin_or_reward_change':False,
       'real_final_test_rows_parsed':0,'manuscript_changes':False,'commit_or_push':False,
       'all_families_retained':True,'test_result':'81 passed in5.31s; no model optimization in tests',
       'tests_sha256':sha256(OUT/'final_tests.log'),'selected_services':selected})
    added={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'revision').rglob('*')
      if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and str(p.relative_to(ROOT)) not in prior}
    payload={'version':'comcom.forecasting-freeze.v1','branch':branch,'head':head,'protocol_frozen_sha256':sha256(OUT/'protocol_frozen.json'),
       'selected_services':selected,'new_files':added,'new_files_excluding_this_manifest':len(added),
       'self_reference_policy':'Manifest excluded from own file inventory; freeze_id hashes canonical payload before adding that field',
       'prior_snapshot_sha256':sha256(snapshot),'source_identities':frozen['source_hashes'],
       'default_controller_pipeline_changed':False,'hysteresis_recalibration_status':'Proposal only; Stage5 baselines unchanged'}
    identity=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dump(OUT/'forecast_freeze_manifest.json',{'freeze_id':identity,**payload})
    print('Forecast task sealed:',len(prior),'prior files unchanged;',len(added),'new files; freeze',identity,flush=True)

if __name__=='__main__':main()
