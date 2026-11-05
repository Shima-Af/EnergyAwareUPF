"""Prescribed 40-job launch, fixed concurrency; failures retained without retries."""
import concurrent.futures,json,os,subprocess,sys
from .common import CFG,OUT,ROOT,dump
from .train import append,verify_inputs


def job(algorithm,schema,seed):
    name=f'{algorithm}_{schema}_{seed}';path=OUT/f'{name}.log'
    with path.open('x') as log:
        result=subprocess.run([sys.executable,'-m','revision.stage7.train','--algorithm',algorithm,'--schema',schema,'--seed',str(seed)],
          cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT)
    record={'algorithm':algorithm,'schema':schema,'seed':seed,'exit_code':result.returncode,'log':str(path.relative_to(ROOT))}
    append(OUT/'worker_completion.jsonl',record);print(record,flush=True);return record


def main():
    verify_inputs()
    # Interleave seed blocks and schemas/algorithms so no family is systematically
    # assigned a late all-idle or high-contention period. No outcome-dependent jobs.
    jobs=[(algorithm,schema,seed) for seed in CFG['seeds'] for schema in CFG['schemas'] for algorithm in CFG['algorithms']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=CFG['workers']) as pool:
        futures=[pool.submit(job,*j) for j in jobs];results=[f.result() for f in futures]
    dump(OUT/'worker_results.json',results)
    if len(results)!=40 or any(r['exit_code'] for r in results):raise RuntimeError('Prescribed job failure; no automatic retries')

if __name__=='__main__':main()
