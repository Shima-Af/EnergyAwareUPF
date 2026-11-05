"""Freeze full factorial, sources, runtime and preflight before40 policy fits."""
import json,subprocess,re
from .common import ROOT,BASE,OUT,CFG,dump,sha256
from .pipeline import feature_names


def main():
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    if branch!='revision/comcom-clean-evaluation':raise ValueError('Wrong branch')
    log=OUT/'preflight_tests.log'
    if not re.search(r'\b\d+ passed\b',log.read_text()) or re.search(r'(?m)^FAILED',log.read_text()):raise ValueError('Preflight failed')
    sources=[ROOT/'Paper artifacts/manuscript_energyawareupf.tex',ROOT/'src/environment.py',
      ROOT/'revision/FULL_TRAINING_EVALUATION.md',ROOT/'revision/FINAL_BASELINE_COMPARISON.md',ROOT/'revision/FINAL_FORECASTING_VALIDATION.md',
      ROOT/'revision/stage3/pipeline.py',ROOT/'revision/forecast_service/core.py',ROOT/'revision/evaluation/contract.py',ROOT/'revision/evaluation/data.py',ROOT/'revision/stage4/train.py',ROOT/'revision/stage4/rng_isolation.py',
      ROOT/'revision/stage5/baselines.py',ROOT/'revision/reports/stage5/baseline_frozen.json',
      ROOT/'revision/reports/forecast_service/forecast_freeze_manifest.json',OUT/'prepared.json',OUT/'outcome_cache.csv']
    for name in ['netflix','youtube']:
        sources += [ROOT/CFG['forecast_ledgers'][name],ROOT/CFG['forecast_services'][name],
          ROOT/f'revision/reports/forecast_service/{name}/selection_frozen.json',ROOT/f'revision/reports/forecast_service/{name}/hysteresis_recalibration_proposal.json']
    prepared=json.loads((OUT/'prepared.json').read_text())
    sources += [__import__('pathlib').Path(p) for p in prepared['twin_manifest']['pinned_paths_and_hashes']]
    from revision.evaluation.twin import _runtime
    import torch,stable_baselines3,sb3_contrib
    dump(OUT/'protocol_frozen.json',{'configuration':CFG,'branch':branch,
      'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'implementation_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in BASE.glob('*') if p.is_file()},
      'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in sources},'preflight_sha256':sha256(log),
      'feature_order':{s:feature_names(s) for s in CFG['schemas']},'prepared_sha256':sha256(OUT/'prepared.json'),
      'runtime':{'twin_cpu_profile':_runtime()[3],'torch':torch.__version__,'sb3':stable_baselines3.__version__,'sb3_contrib':sb3_contrib.__version__,
      'torch_threads':1,'device':'cpu','worker_concurrency':CFG['workers']},
      'scientific_policy_fits_started':False,'no_final_test_access':True})
    print('Prescribed40 scientific fits frozen before outcome inspection',flush=True)

if __name__=='__main__':main()
