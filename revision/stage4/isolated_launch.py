"""Fixed-seed implementation-correction rerun; no search or hyperparameter change."""
import concurrent.futures,json,os,re,subprocess,sys
from revision.evaluation.twin import ROOT,sha256
from .common import CFG,inputs,dump
from .train import append
OUT=ROOT/'revision/reports/stage4_rng_isolated'


def job(algorithm,seed):
    log_path=OUT/f'{algorithm}_seed_{seed}.log'
    with log_path.open('x') as log:
        result=subprocess.run([sys.executable,'-m','revision.stage4.isolated_run','--algorithm',algorithm,'--seed',str(seed)],cwd=ROOT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT)
    record={'algorithm':algorithm,'seed':seed,'exit_code':result.returncode,'log':str(log_path.relative_to(ROOT))}
    append(OUT/'worker_completion.jsonl',record);print(record,flush=True);return record


def main():
    _,_,contract=inputs();test=ROOT/'revision/reports/stage4_isolated_preflight_tests.log'
    assert re.search(r'\b[0-9]+ passed\b',test.read_text()) and not re.search(r'(?m)^(FAILED|ERROR) ',test.read_text())
    dump(OUT/'training_protocol_frozen.json',{'config':CFG,'contract':contract.__dict__,
       'protocol_sha256':sha256(ROOT/'revision/stage4/protocol.json'),
       'stage3_pipeline_sha256':sha256(ROOT/'revision/stage3/pipeline.py'),
       'twin_config_sha256':sha256(ROOT/'revision/configs/revised_twin_v1.json'),
       'forecast_sha256':sha256(OUT/'forecasts.csv'),'forecast_selection_sha256':sha256(OUT/'forecast_selection.json'),
       'stage4_implementation_hashes':{p.name:sha256(p) for p in (ROOT/'revision/stage4').glob('*.py')},
       'preflight_sha256':sha256(test),'workers':4,'all_final_tests_prohibited':True,
       'execution_amendment':{'id':'comcom.stage4.validation-rng-isolation.v1','change':'Checkpoint capture saves/restores Python, NumPy and CPU Torch RNG states, including on exceptions',
          'reason':'Confirmed SB3 checkpoint loading reseeds all three generators; diagnostic independent of favorable/unfavorable results',
          'scientific_hyperparameters_changed':False,'forecast_refits':0,'initial_affected_runs_preserved':10,
          'initial_campaign_snapshot_sha256':sha256(ROOT/'revision/reports/stage4_initial_campaign_sha256.json')}})
    jobs=[(alg,s) for s in CFG['seeds'] for alg in CFG['algorithms']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=[f.result() for f in [pool.submit(job,*args) for args in jobs]]
    dump(OUT/'worker_results.json',results)
    if any(r['exit_code'] for r in results):raise RuntimeError('Corrected seed failure; no automatic retry')

if __name__=='__main__':main()
