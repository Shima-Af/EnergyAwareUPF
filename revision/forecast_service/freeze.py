"""Protocol commitment before candidate fitting or result comparisons."""
import subprocess
from pathlib import Path
from .core import BASE,OUT,CFG,ROOT,sha256,dump,_runtime

def main():
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    if branch!='revision/comcom-clean-evaluation':raise ValueError('Wrong branch')
    sources=[ROOT/'revision/FULL_TRAINING_EVALUATION.md',ROOT/'revision/FINAL_BASELINE_COMPARISON.md',
       ROOT/'Paper artifacts/manuscript_energyawareupf.tex',ROOT/'revision/reference/Dynamic_UPF_Cyprus_2025_SLICES.pdf',
       ROOT/'revision/stage4/forecast.py',ROOT/'revision/stage3/forecast.py',ROOT/'revision/stage4/protocol.json',
       ROOT/'revision/reports/stage5/baseline_frozen.json']
    implementation=[BASE/n for n in ['__init__.py','core.py','run.py','freeze.py','protocol.json']]
    dump(OUT/'protocol_frozen.json',{'configuration':CFG,'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
      'implementation_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in implementation},
      'preflight_log_sha256':sha256(OUT/'preflight_tests.log'),'runtime':_runtime()[3],
      'git_branch':branch,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'training_started':False,'candidate_results_seen':False,'no_final_test_rows_requested':True,
      'source_direction_and_units':'Read raw Traffic_bps/1e6, preserve common Netflix-training workload scale. No physically invented cap or resampling.'})
    print('Forecast protocol frozen before scientific fits',flush=True)

if __name__=='__main__':main()
