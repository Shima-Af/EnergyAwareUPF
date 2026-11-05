"""Preservation and full artifact freeze, including the access-boundary exception."""
import hashlib,json,subprocess
from pathlib import Path
from .common import ROOT,OUT,MODELS,CFG,dump,sha256,directory,run_name
from .train import verify_inputs


def main():
    frozen=verify_inputs();snapshot=ROOT/'revision/reports/stage7_prior_sha256.json';prior=json.loads(snapshot.read_text())
    changed=[p for p,h in prior.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    if changed:raise ValueError('Prior scientific files changed: '+repr(changed))
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip();head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    tracked=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True)
    if branch!='revision/comcom-clean-evaluation' or head!='c0734715418d3f6d10785e5174aebb4bac24d0da' or tracked:raise ValueError('Git preservation failure')
    workers=json.loads((OUT/'worker_results.json').read_text());assert len(workers)==40 and all(r['exit_code']==0 for r in workers)
    selections=json.loads((OUT/'all_netflix_selections_frozen.json').read_text());pairs=0;timesteps=0
    for item in selections['selections'].values():
        path=directory(item['algorithm'],item['schema'],item['seed']);s=json.loads((path/'selection.json').read_text())
        assert sha256(path/'selection.json')==item['selection_sha256'] and not s['youtube_used'] and not s['final_test_used']
        timesteps+=s['actual_timesteps']
        for score in s['validation_scores']:
            p=ROOT/score['checkpoint'];m=json.loads((p/'bundle.json').read_text());pairs+=1
            for file,h in m['artifacts'].items():assert sha256(p/file)==h
    assert pairs==560 and timesteps==12000000
    assert '106 passed' in (OUT/'final_prefix_guard_tests.log').read_text()
    comparison=json.loads((OUT/'comparison.json').read_text());assert not comparison['final_test_used'] and not comparison['youtube_used_for_selection']
    incident=OUT/'DATA_ACCESS_ADDENDUM.json';qualification=json.loads(incident.read_text())
    assert qualification['literal_no_target_access_requirement_satisfied'] is False
    dump(OUT/'integrity.json',{'branch':branch,'head':head,'tracked_changes':tracked,'prior_scientific_files_checked':len(prior),
      'changed_prior_files':changed,'prior_snapshot_sha256':sha256(snapshot),'scientific_policy_fits':40,
      'scientific_interactions':timesteps,'checkpoint_normalizer_pairs':pairs,'scientific_fit_failures_or_retries':0,
      'forecast_refits':0,'original_twin_reward_and_baseline_files_changed':False,'all_selected_replays_deterministic':True,
      'youtube_checkpoint_selection':False,'final_test_performance_evaluated':False,
      'scientific_pipeline_source_prefix_limit_exclusive':2520,'scientific_pipeline_final_test_rows_parsed':0,
      'whole_task_literal_no_target_access_requirement_satisfied':False,
      'legacy_preflight_full_csv_parsing_disclosed':True,'data_access_addendum_sha256':sha256(incident),
      'test_result':'106 passed in8.87s, prefix-restricted verification','test_log_sha256':sha256(OUT/'final_prefix_guard_tests.log'),
      'manuscript_modified':False,'commit_or_push':False})
    # Collect before opening the manifest: no self-reference or streaming-log hash.
    added={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'revision').rglob('*')
      if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and str(p.relative_to(ROOT)) not in prior}
    payload={'version':'comcom.observation-freeze.v1','branch':branch,'head':head,
      'protocol_frozen_sha256':sha256(OUT/'protocol_frozen.json'),'netflix_selection_freeze_sha256':sha256(OUT/'all_netflix_selections_frozen.json'),
      'comparison_sha256':sha256(OUT/'comparison.json'),'source_hashes':frozen['source_hashes'],
      'new_files':added,'new_files_excluding_manifest':len(added),'prior_snapshot_sha256':sha256(snapshot),
      'scientific_policy_fits':40,'scientific_interactions':12000000,'checkpoint_pairs':560,
      'exception':'Legacy preflight parsed the full Netflix CSV. No final-test performance or scientific selection/gradient use; literal no-access restriction was not satisfied.',
      'self_reference_policy':'Manifest excluded from own file inventory; freeze_id hashes canonical payload before adding that field.'}
    identity=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dump(OUT/'observation_freeze_manifest.json',{'freeze_id':identity,**payload})
    print('Stage7 sealed:',len(prior),'prior files unchanged;',len(added),'new files;',pairs,'checkpoint pairs; freeze',identity,flush=True)

if __name__=='__main__':main()
