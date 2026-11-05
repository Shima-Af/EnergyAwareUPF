"""Execute the fixed independent seed jobs in separate CPU processes, preserving failures."""
import concurrent.futures,json,os,subprocess,sys,re
from pathlib import Path
from revision.evaluation.twin import ROOT,sha256
from .common import CFG,OUT,inputs,dump
from .train import append


def job(algorithm,seed):
    path=OUT/f'{algorithm}_seed_{seed}.log'
    with path.open('x') as log:
        process=subprocess.run([sys.executable,'-m','revision.stage4.train','--algorithm',algorithm,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    result={'algorithm':algorithm,'seed':seed,'exit_code':process.returncode,'log':str(path.relative_to(ROOT))}
    append(OUT/'worker_completion.jsonl',result);print(result,flush=True);return result


def main():
    _,_,contract=inputs()
    preflight=ROOT/'revision/reports/stage4_preflight_tests.log'
    if not preflight.exists() or not re.search(r'\b[0-9]+ passed\b',preflight.read_text()) or re.search(r'(?m)^(FAILED|ERROR) ',preflight.read_text()):raise ValueError('Passing full preflight log required')
    dump(OUT/'training_protocol_frozen.json',{'config':CFG,'contract':contract.__dict__,
       'protocol_sha256':sha256(ROOT/'revision/stage4/protocol.json'),
       'stage3_pipeline_sha256':sha256(ROOT/'revision/stage3/pipeline.py'),
       'twin_config_sha256':sha256(ROOT/'revision/configs/revised_twin_v1.json'),
       'forecast_sha256':sha256(OUT/'forecasts.csv'),'forecast_selection_sha256':sha256(OUT/'forecast_selection.json'),
       'stage4_implementation_hashes':{p.name:sha256(p) for p in (ROOT/'revision/stage4').glob('*.py')},
       'preflight_sha256':sha256(preflight),'workers':4,'all_final_tests_prohibited':True})
    jobs=[(alg,seed) for seed in CFG['seeds'] for alg in CFG['algorithms']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(job,*args) for args in jobs]
        results=[f.result() for f in futures]
    dump(OUT/'worker_results.json',results)
    if any(r['exit_code'] for r in results):raise RuntimeError('Some prespecified seeds failed; retain logs, no automatic retries')

if __name__=='__main__':main()
